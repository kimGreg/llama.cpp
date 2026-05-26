// DPMoE — fused MoE GEMV mixed-precision batch correctness test.
//
// Validates qwen3::naver_gemv_moe_launch for n_tokens > 1 with
// per-expert precision variation.  The selected kernel mode is controlled
// by the normal runtime env vars, so this executable can validate both:
//
//   DP_MOE_FUSED_MIXED_MODE=loop
//   DP_MOE_KERNEL_OPTIMIZED=1 DP_MOE_OPTIMIZED_BATCH_KERNEL=1

#include "fused_kernels.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

float half_to_float(uint16_t u) {
    __half h;
    std::memcpy(&h, &u, sizeof(h));
    return __half2float(h);
}

struct DeviceCase {
    int M = 128;
    int K = 128;
    int n_tokens = 5;
    int n_used = 4;
    int n_experts = 16;

    std::vector<int32_t> h_ids;
    std::vector<int> h_prec;
    std::vector<uint16_t> h_x;
    std::vector<std::vector<uint32_t>> h_qw;
    std::vector<std::vector<uint16_t>> h_alpha;
    std::vector<std::vector<uint16_t>> h_qbias;

    std::vector<uint32_t *> d_qw;
    std::vector<uint16_t *> d_alpha;
    std::vector<uint16_t *> d_qbias;
    std::vector<void **> d_qw_ptrs;
    std::vector<void **> d_alpha_ptrs;

    int32_t * d_ids = nullptr;
    int * d_prec = nullptr;
    uint16_t * d_x = nullptr;
    float * d_y = nullptr;
    dp_moe_ext::MoeExpertTable table;
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

DeviceCase make_case(bool shared_x) {
    DeviceCase c;
    const int M = c.M;
    const int K = c.K;
    const int n_tu = c.n_tokens * c.n_used;
    const size_t qw_words = (size_t)(K / 32) * M;
    const size_t alpha_elems = (size_t)(K / kGroupSize) * M;

    std::mt19937 rng(0xBADC0DEu);
    std::uniform_int_distribution<uint32_t> bits(0, UINT32_MAX);
    std::uniform_real_distribution<float> small(-0.025f, 0.025f);
    std::uniform_real_distribution<float> xdist(-1.0f, 1.0f);

    c.h_ids.resize(n_tu);
    for (int i = 0; i < n_tu; ++i) c.h_ids[i] = i % c.n_experts;

    const int active_prec[] = {2, 6, 3, 8, 4, 7, 2, 5};
    c.h_prec.assign(c.n_experts, kMaxPlanes);
    for (int e = 0; e < c.n_experts; ++e) {
        c.h_prec[e] = active_prec[e % (int)(sizeof(active_prec) / sizeof(active_prec[0]))];
    }

    const size_t x_elems = (size_t)(shared_x ? c.n_tokens : n_tu) * K;
    c.h_x.resize(x_elems);
    for (auto & v : c.h_x) v = half_bits(xdist(rng));

    c.h_qw.resize((size_t)c.n_experts * kMaxPlanes);
    c.h_alpha.resize((size_t)c.n_experts * kMaxPlanes);
    c.h_qbias.resize(c.n_experts);
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
            auto & hq = c.h_qw[(size_t)e * kMaxPlanes + p];
            hq.resize(qw_words);
            for (auto & v : hq) v = bits(rng);
            uint32_t * dq = nullptr;
            check_cuda(cudaMalloc(&dq, hq.size() * sizeof(uint32_t)),
                       "cudaMalloc qw");
            check_cuda(cudaMemcpy(dq, hq.data(), hq.size() * sizeof(uint32_t),
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy qw");
            c.d_qw[(size_t)e * kMaxPlanes + p] = dq;
            qw_plane[p] = dq;

            auto & ha = c.h_alpha[(size_t)e * kMaxPlanes + p];
            ha.resize(alpha_elems);
            for (auto & v : ha) v = half_bits(small(rng));
            uint16_t * da = nullptr;
            check_cuda(cudaMalloc(&da, ha.size() * sizeof(uint16_t)),
                       "cudaMalloc alpha");
            check_cuda(cudaMemcpy(da, ha.data(), ha.size() * sizeof(uint16_t),
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy alpha");
            c.d_alpha[(size_t)e * kMaxPlanes + p] = da;
            alpha_plane[p] = da;
        }

        auto & hb = c.h_qbias[e];
        hb.resize(alpha_elems);
        for (auto & v : hb) v = half_bits(small(rng));
        uint16_t * db = nullptr;
        check_cuda(cudaMalloc(&db, hb.size() * sizeof(uint16_t)),
                   "cudaMalloc qbias");
        check_cuda(cudaMemcpy(db, hb.data(), hb.size() * sizeof(uint16_t),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy qbias");
        c.d_qbias[e] = db;

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
        table_qbias[e] = db;
    }

    dp_moe_ext::qwen3::alloc_moe_expert_table(
        c.table, table_qw.data(), table_alpha.data(),
        table_qbias.data(), c.n_experts);

    check_cuda(cudaMalloc(&c.d_ids, c.h_ids.size() * sizeof(int32_t)),
               "cudaMalloc ids");
    check_cuda(cudaMemcpy(c.d_ids, c.h_ids.data(),
                          c.h_ids.size() * sizeof(int32_t),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy ids");
    check_cuda(cudaMalloc(&c.d_prec, c.h_prec.size() * sizeof(int)),
               "cudaMalloc prec");
    check_cuda(cudaMemcpy(c.d_prec, c.h_prec.data(),
                          c.h_prec.size() * sizeof(int),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy prec");
    check_cuda(cudaMalloc(&c.d_x, c.h_x.size() * sizeof(uint16_t)),
               "cudaMalloc x");
    check_cuda(cudaMemcpy(c.d_x, c.h_x.data(),
                          c.h_x.size() * sizeof(uint16_t),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy x");
    check_cuda(cudaMalloc(&c.d_y, (size_t)n_tu * M * sizeof(float)),
               "cudaMalloc y");
    return c;
}

std::vector<float> cpu_reference(const DeviceCase & c, bool shared_x) {
    const int n_tu = c.n_tokens * c.n_used;
    std::vector<float> y((size_t)n_tu * c.M, 0.0f);
    for (int tu = 0; tu < n_tu; ++tu) {
        const int t = tu / c.n_used;
        const int e = c.h_ids[tu];
        const int P = c.h_prec[e];
        const uint16_t * x = c.h_x.data() + (size_t)(shared_x ? t : tu) * c.K;
        for (int m = 0; m < c.M; ++m) {
            double acc = 0.0;
            for (int k = 0; k < c.K; ++k) {
                const int g = k / kGroupSize;
                const int kt = k / 32;
                const int bit = k & 31;
                double w = half_to_float(c.h_qbias[e][(size_t)g * c.M + m]);
                for (int p = 0; p < P; ++p) {
                    const auto & qw = c.h_qw[(size_t)e * kMaxPlanes + p];
                    const auto & al = c.h_alpha[(size_t)e * kMaxPlanes + p];
                    const uint32_t word = qw[(size_t)kt * c.M + m];
                    const double sign = ((word >> bit) & 1u) ? 1.0 : -1.0;
                    w += sign * half_to_float(al[(size_t)g * c.M + m]);
                }
                acc += w * half_to_float(x[k]);
            }
            y[(size_t)tu * c.M + m] = (float)acc;
        }
    }
    return y;
}

bool run_case(const char * name, bool shared_x) {
    DeviceCase c = make_case(shared_x);
    const int n_tu = c.n_tokens * c.n_used;
    const size_t y_bytes = (size_t)n_tu * c.M * sizeof(float);
    check_cuda(cudaMemset(c.d_y, 0, y_bytes), "cudaMemset y");

    dp_moe_ext::qwen3::naver_gemv_moe_launch(
        c.d_x, c.d_y, c.d_ids, c.table,
        c.M, c.K, c.n_tokens, c.n_used,
        kMaxPlanes, c.d_prec, kGroupSize,
        shared_x ? 1 : 0,
        /*routed_precision_span_hint=*/6,
        /*stream=*/nullptr);
    check_cuda(cudaDeviceSynchronize(), "kernel sync");

    std::vector<float> got(y_bytes / sizeof(float));
    check_cuda(cudaMemcpy(got.data(), c.d_y, y_bytes, cudaMemcpyDeviceToHost),
               "copy y");
    std::vector<float> ref = cpu_reference(c, shared_x);

    double max_abs = 0.0;
    double sum_abs2 = 0.0;
    double sum_ref2 = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double d = (double)got[i] - (double)ref[i];
        max_abs = std::max(max_abs, std::fabs(d));
        sum_abs2 += d * d;
        sum_ref2 += (double)ref[i] * (double)ref[i];
    }
    const double rel_l2 = std::sqrt(sum_abs2) /
        std::max(1.0e-12, std::sqrt(sum_ref2));
    const bool ok = max_abs < 2.0e-3 && rel_l2 < 5.0e-5;
    std::printf("%-14s shared_x=%d max_abs=%.8g rel_l2=%.8g %s\n",
                name, shared_x ? 1 : 0, max_abs, rel_l2,
                ok ? "PASS" : "FAIL");
    free_case(c);
    return ok;
}

} // namespace

int main() {
    const char * mode = std::getenv("DP_MOE_FUSED_MIXED_MODE");
    const char * opt = std::getenv("DP_MOE_KERNEL_OPTIMIZED");
    const char * opt_batch = std::getenv("DP_MOE_OPTIMIZED_BATCH_KERNEL");
    const char * opt_batch_alias = std::getenv("DP_MOE_KERNEL_OPTIMIZED_BATCH");
    std::printf("mode=%s optimized=%s optimized_batch=%s optimized_batch_alias=%s\n",
                mode ? mode : "auto",
                opt ? opt : "0",
                opt_batch ? opt_batch : "0",
                opt_batch_alias ? opt_batch_alias : "0");

    bool ok = true;
    ok &= run_case("mixed_batch", true);
    ok &= run_case("mixed_batch", false);
    return ok ? 0 : 1;
}
