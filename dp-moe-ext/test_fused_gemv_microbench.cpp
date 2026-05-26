// DPMoE — fused MoE GEMV microbench.
//
// This isolates qwen3::naver_gemv_moe_launch from llama-server,
// routing, cache, and prefill.  It builds synthetic per-expert AnyBCQ
// plane pointer tables, launches the fused GEMV decode shape, and can
// dump output buffers so different kernel modes can be compared.
//
// Mode is selected by DP_MOE_FUSED_MIXED_MODE:
//   loop   : original one-block loops over all planes for its expert
//   flat   : plane id is flattened into grid.z
//   bucket : one loop-kernel launch per precision bucket
//   auto   : production default
//
// Flat grouping is selected by DP_MOE_FUSED_PLANE_GROUP:
//   1      : legacy one plane per block
//   2      : two planes per block (current optimized default)
//   4      : four planes per block

#include "fused_kernels.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int kMaxPlanes = 8;
constexpr int kGroupSize = 128;

void check_cuda(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s: %s\n", what,
                     cudaGetErrorString(e));
        std::exit(1);
    }
}

uint16_t half_bits(float x) {
    __half h = __float2half(x);
    uint16_t u;
    std::memcpy(&u, &h, sizeof(u));
    return u;
}

uint64_t fnv1a(const void * data, size_t bytes) {
    const uint8_t * p = static_cast<const uint8_t *>(data);
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < bytes; ++i) {
        h ^= (uint64_t) p[i];
        h *= 1099511628211ull;
    }
    return h;
}

std::string mix_name(const std::vector<int> & p) {
    std::string s;
    for (size_t i = 0; i < p.size(); ++i) {
        if (i) s += "-";
        s += std::to_string(p[i]);
    }
    return s;
}

struct CaseSpec {
    const char * name;
    int M;
    int K;
    int n_tokens;
    int n_used;
    std::vector<int> active_precisions;
};

struct DeviceCase {
    CaseSpec spec;
    int n_experts = 0;

    std::vector<uint32_t *> d_qw;
    std::vector<uint16_t *> d_alpha;
    std::vector<uint16_t *> d_qbias;
    std::vector<void **> d_qw_ptrs;
    std::vector<void **> d_alpha_ptrs;

    dp_moe_ext::MoeExpertTable table;

    int32_t * d_ids = nullptr;
    int * d_prec = nullptr;
    uint16_t * d_x = nullptr;
    float * d_y = nullptr;
};

void free_case(DeviceCase & c) {
    dp_moe_ext::qwen3::free_moe_expert_table(c.table);
    for (auto * p : c.d_qw) cudaFree(p);
    for (auto * p : c.d_alpha) cudaFree(p);
    for (auto * p : c.d_qbias) cudaFree(p);
    for (auto * p : c.d_qw_ptrs) cudaFree(p);
    for (auto * p : c.d_alpha_ptrs) cudaFree(p);
    cudaFree(c.d_ids);
    cudaFree(c.d_prec);
    cudaFree(c.d_x);
    cudaFree(c.d_y);
    c = DeviceCase{};
}

DeviceCase make_case(const CaseSpec & spec, uint32_t seed) {
    DeviceCase c;
    c.spec = spec;
    c.n_experts = std::max(128, spec.n_tokens * spec.n_used);

    const int M = spec.M;
    const int K = spec.K;
    if (K % 64 != 0 || kGroupSize % 64 != 0 || K % kGroupSize != 0) {
        std::fprintf(stderr, "bad shape M=%d K=%d group=%d\n",
                     M, K, kGroupSize);
        std::exit(1);
    }
    const size_t qw_words = (size_t)(K / 32) * M;
    const size_t alpha_elems = (size_t)(K / kGroupSize) * M;

    std::mt19937 rng(seed);
    std::uniform_int_distribution<uint32_t> bits(0, UINT32_MAX);
    std::uniform_real_distribution<float> small(-0.06f, 0.06f);
    std::uniform_real_distribution<float> xdist(-1.0f, 1.0f);

    c.d_qw.resize((size_t)c.n_experts * kMaxPlanes, nullptr);
    c.d_alpha.resize((size_t)c.n_experts * kMaxPlanes, nullptr);
    c.d_qbias.resize(c.n_experts, nullptr);
    c.d_qw_ptrs.resize(c.n_experts, nullptr);
    c.d_alpha_ptrs.resize(c.n_experts, nullptr);

    std::vector<const void *> table_qw(c.n_experts);
    std::vector<const void *> table_alpha(c.n_experts);
    std::vector<const void *> table_qbias(c.n_experts);

    for (int e = 0; e < c.n_experts; ++e) {
        std::vector<void *> qw_plane(kMaxPlanes, nullptr);
        std::vector<void *> alpha_plane(kMaxPlanes, nullptr);

        for (int p = 0; p < kMaxPlanes; ++p) {
            std::vector<uint32_t> h_qw(qw_words);
            for (auto & v : h_qw) v = bits(rng);
            uint32_t * d_qw = nullptr;
            check_cuda(cudaMalloc(&d_qw, h_qw.size() * sizeof(uint32_t)),
                       "cudaMalloc qw");
            check_cuda(cudaMemcpy(d_qw, h_qw.data(),
                                  h_qw.size() * sizeof(uint32_t),
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy qw");
            c.d_qw[(size_t)e * kMaxPlanes + p] = d_qw;
            qw_plane[p] = d_qw;

            std::vector<uint16_t> h_alpha(alpha_elems);
            for (auto & v : h_alpha) v = half_bits(small(rng));
            uint16_t * d_alpha = nullptr;
            check_cuda(cudaMalloc(&d_alpha, h_alpha.size() * sizeof(uint16_t)),
                       "cudaMalloc alpha");
            check_cuda(cudaMemcpy(d_alpha, h_alpha.data(),
                                  h_alpha.size() * sizeof(uint16_t),
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy alpha");
            c.d_alpha[(size_t)e * kMaxPlanes + p] = d_alpha;
            alpha_plane[p] = d_alpha;
        }

        std::vector<uint16_t> h_qbias(alpha_elems);
        for (auto & v : h_qbias) v = half_bits(small(rng));
        uint16_t * d_qbias = nullptr;
        check_cuda(cudaMalloc(&d_qbias, h_qbias.size() * sizeof(uint16_t)),
                   "cudaMalloc qbias");
        check_cuda(cudaMemcpy(d_qbias, h_qbias.data(),
                              h_qbias.size() * sizeof(uint16_t),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy qbias");
        c.d_qbias[e] = d_qbias;

        void ** d_qw_ptrs = nullptr;
        void ** d_alpha_ptrs = nullptr;
        check_cuda(cudaMalloc(&d_qw_ptrs, kMaxPlanes * sizeof(void *)),
                   "cudaMalloc qw ptrs");
        check_cuda(cudaMalloc(&d_alpha_ptrs, kMaxPlanes * sizeof(void *)),
                   "cudaMalloc alpha ptrs");
        check_cuda(cudaMemcpy(d_qw_ptrs, qw_plane.data(),
                              kMaxPlanes * sizeof(void *),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy qw ptrs");
        check_cuda(cudaMemcpy(d_alpha_ptrs, alpha_plane.data(),
                              kMaxPlanes * sizeof(void *),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy alpha ptrs");
        c.d_qw_ptrs[e] = d_qw_ptrs;
        c.d_alpha_ptrs[e] = d_alpha_ptrs;
        table_qw[e] = d_qw_ptrs;
        table_alpha[e] = d_alpha_ptrs;
        table_qbias[e] = d_qbias;
    }

    dp_moe_ext::qwen3::alloc_moe_expert_table(
        c.table, table_qw.data(), table_alpha.data(),
        table_qbias.data(), c.n_experts);

    const int n_tu = spec.n_tokens * spec.n_used;
    std::vector<int32_t> h_ids(n_tu);
    for (int i = 0; i < n_tu; ++i) h_ids[i] = i;
    check_cuda(cudaMalloc(&c.d_ids, h_ids.size() * sizeof(int32_t)),
               "cudaMalloc ids");
    check_cuda(cudaMemcpy(c.d_ids, h_ids.data(),
                          h_ids.size() * sizeof(int32_t),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy ids");

    std::vector<int> h_prec(c.n_experts, kMaxPlanes);
    for (int i = 0; i < n_tu; ++i) {
        h_prec[i] = spec.active_precisions[(size_t)i %
                                           spec.active_precisions.size()];
    }
    check_cuda(cudaMalloc(&c.d_prec, h_prec.size() * sizeof(int)),
               "cudaMalloc prec");
    check_cuda(cudaMemcpy(c.d_prec, h_prec.data(),
                          h_prec.size() * sizeof(int),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy prec");

    std::vector<uint16_t> h_x((size_t)spec.n_tokens * K);
    for (auto & v : h_x) v = half_bits(xdist(rng));
    check_cuda(cudaMalloc(&c.d_x, h_x.size() * sizeof(uint16_t)),
               "cudaMalloc x");
    check_cuda(cudaMemcpy(c.d_x, h_x.data(),
                          h_x.size() * sizeof(uint16_t),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy x");

    check_cuda(cudaMalloc(&c.d_y, (size_t)n_tu * M * sizeof(float)),
               "cudaMalloc y");
    check_cuda(cudaMemset(c.d_y, 0, (size_t)n_tu * M * sizeof(float)),
               "cudaMemset y");

    return c;
}

void launch(DeviceCase & c, cudaStream_t stream) {
    const auto mm = std::minmax_element(
        c.spec.active_precisions.begin(),
        c.spec.active_precisions.end());
    const int span_hint = *mm.second - *mm.first;
    const int max_precision = *mm.second;
    dp_moe_ext::qwen3::naver_gemv_moe_launch(
        c.d_x, c.d_y, c.d_ids, c.table,
        c.spec.M, c.spec.K, c.spec.n_tokens, c.spec.n_used,
        max_precision, c.d_prec, kGroupSize,
        /*shared_x=*/1,
        span_hint,
        (dp_moe_ext::StreamHandle)stream);
}

struct Result {
    double us = 0.0;
    double plane_mb = 0.0;
    double effective_gbps = 0.0;
    uint64_t hash = 0;
    double sum = 0.0;
    double max_abs = 0.0;
};

Result run_case(DeviceCase & c, int warmup, int iters,
                const std::string & dump_prefix) {
    cudaStream_t stream = nullptr;
    check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate");

    const size_t y_bytes = (size_t)c.spec.n_tokens * c.spec.n_used *
                           c.spec.M * sizeof(float);
    check_cuda(cudaMemsetAsync(c.d_y, 0, y_bytes, stream), "warm y zero");
    for (int i = 0; i < warmup; ++i) {
        launch(c, stream);
    }
    check_cuda(cudaStreamSynchronize(stream), "warmup sync");

    cudaEvent_t t0, t1;
    check_cuda(cudaEventCreate(&t0), "event create 0");
    check_cuda(cudaEventCreate(&t1), "event create 1");
    check_cuda(cudaEventRecord(t0, stream), "event record 0");
    for (int i = 0; i < iters; ++i) {
        launch(c, stream);
    }
    check_cuda(cudaEventRecord(t1, stream), "event record 1");
    check_cuda(cudaEventSynchronize(t1), "event sync");
    float ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&ms, t0, t1), "event elapsed");
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);

    check_cuda(cudaMemsetAsync(c.d_y, 0, y_bytes, stream), "dump y zero");
    launch(c, stream);
    check_cuda(cudaStreamSynchronize(stream), "dump sync");

    std::vector<float> y(y_bytes / sizeof(float));
    check_cuda(cudaMemcpy(y.data(), c.d_y, y_bytes, cudaMemcpyDeviceToHost),
               "copy y");

    Result r;
    r.us = (double)ms * 1000.0 / (double)iters;
    r.hash = fnv1a(y.data(), y_bytes);
    for (float v : y) {
        r.sum += (double)v;
        r.max_abs = std::max(r.max_abs, std::fabs((double)v));
    }

    const int K = c.spec.K;
    const int M = c.spec.M;
    const int kgroups = K / kGroupSize;
    double bytes = 0.0;
    for (int p : c.spec.active_precisions) {
        bytes += (double)(K / 32) * M * p * sizeof(uint32_t);
        bytes += (double)kgroups * M * p * sizeof(uint16_t);
        bytes += (double)kgroups * M * sizeof(uint16_t);
    }
    r.plane_mb = bytes / 1024.0 / 1024.0;
    r.effective_gbps = bytes / (r.us * 1.0e-6) / 1.0e9;

    if (!dump_prefix.empty()) {
        std::string path = dump_prefix + "." + c.spec.name + ".bin";
        std::ofstream f(path, std::ios::binary);
        f.write(reinterpret_cast<const char *>(y.data()), (std::streamsize)y_bytes);
    }

    cudaStreamDestroy(stream);
    return r;
}

std::vector<CaseSpec> make_cases() {
    return {
        // Same average K budget, different per-expert tail.
        // Tailed cases keep sum(K_e) equal to uniform but introduce
        // K_max-K_min >= 3 imbalance across the 8 routed experts.
        {"gate_k3_uniform", 768, 2048, 1, 8, {3,3,3,3,3,3,3,3}},
        {"gate_k3_tail",    768, 2048, 1, 8, {2,2,2,2,2,2,6,6}},
        {"gate_k4_uniform", 768, 2048, 1, 8, {4,4,4,4,4,4,4,4}},
        {"gate_k4_tail",    768, 2048, 1, 8, {2,2,2,2,6,6,6,6}},
        {"gate_k5_uniform", 768, 2048, 1, 8, {5,5,5,5,5,5,5,5}},
        {"gate_k5_tail",    768, 2048, 1, 8, {2,2,4,4,7,7,7,7}},
        {"gate_k6_uniform", 768, 2048, 1, 8, {6,6,6,6,6,6,6,6}},
        {"gate_k6_tail",    768, 2048, 1, 8, {2,2,6,6,8,8,8,8}},

        {"down_k3_uniform", 2048, 768, 1, 8, {3,3,3,3,3,3,3,3}},
        {"down_k3_tail",    2048, 768, 1, 8, {2,2,2,2,2,2,6,6}},
        {"down_k4_uniform", 2048, 768, 1, 8, {4,4,4,4,4,4,4,4}},
        {"down_k4_tail",    2048, 768, 1, 8, {2,2,2,2,6,6,6,6}},
        {"down_k5_uniform", 2048, 768, 1, 8, {5,5,5,5,5,5,5,5}},
        {"down_k5_tail",    2048, 768, 1, 8, {2,2,4,4,7,7,7,7}},
        {"down_k6_uniform", 2048, 768, 1, 8, {6,6,6,6,6,6,6,6}},
        {"down_k6_tail",    2048, 768, 1, 8, {2,2,6,6,8,8,8,8}},

        // Batched/prefill-like mixed cases. These exercise the
        // DP_MOE_OPTIMIZED_BATCH_KERNEL opt-in path, whose flat grid
        // places the plane-group axis on grid.y so grid.z remains
        // n_tokens * n_used rather than n_tokens * n_used * groups.
        {"batch_gate_k3_uniform", 768, 2048, 16, 8, {3,3,3,3,3,3,3,3}},
        {"batch_gate_k3_tail",    768, 2048, 16, 8, {2,2,2,2,2,2,6,6}},
        {"batch_down_k3_uniform", 2048, 768, 16, 8, {3,3,3,3,3,3,3,3}},
        {"batch_down_k3_tail",    2048, 768, 16, 8, {2,2,2,2,2,2,6,6}},
    };
}

} // namespace

int main(int argc, char ** argv) {
    const int iters = argc > 1 ? std::atoi(argv[1]) : 512;
    const std::string dump_prefix = argc > 2 ? argv[2] : "";
    const int warmup = 32;

    const char * mode = std::getenv("DP_MOE_FUSED_MIXED_MODE");
    if (mode == nullptr || mode[0] == '\0') mode = "auto";
    const char * plane_group = std::getenv("DP_MOE_FUSED_PLANE_GROUP");
    if (plane_group == nullptr || plane_group[0] == '\0') plane_group = "default";

    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, dev);

    std::printf("mode=%s plane_group=%s device=%d %s warmup=%d iters=%d\n",
                mode, plane_group, dev, prop.name, warmup, iters);
    std::printf("%-16s %7s %7s %7s %18s %8s %10s %10s %12s %14s\n",
                "case", "M", "K", "tu", "precisions",
                "us", "MB/call", "GB/s", "hash", "sum");

    for (const auto & spec : make_cases()) {
        DeviceCase c = make_case(spec, 0xC0FFEEu);
        Result r = run_case(c, warmup, iters, dump_prefix);
        std::printf("%-16s %7d %7d %7d %18s %8.2f %10.2f %10.2f %012llx %14.4f\n",
                    spec.name, spec.M, spec.K,
                    spec.n_tokens * spec.n_used,
                    mix_name(spec.active_precisions).c_str(),
                    r.us, r.plane_mb, r.effective_gbps,
                    (unsigned long long)r.hash, r.sum);
        free_case(c);
    }

    return 0;
}
