// streamllm-ext — full-model StreamllmRuntime test.
//
// Loads every managed tensor of a streamllm GGUF into a StreamllmRuntime
// (== upload all upstream layouts into a VramChunkPool), then spot-
// checks the kernel output for N tensors against Python references.
//
// Usage:
//   streamllm-runtime-test <gguf> <refs_dir> <tensor_name_1> [tensor_name_2 ...]
//
// For each <tensor_name_i>, expects the refs directory to contain
//   <tensor_name_i>.x.bin     — fp16 activations (length K)
//   <tensor_name_i>.y.bin     — fp16 expected output (length M)
// produced by streamllm/scripts/dump_gpu_reference.py.
//
// Exit 0 on all spot-checks PASS + full-model load success.

#include "stream_reader.h"
#include "runtime.h"
#include "naver_gemv.h"

#include <gguf.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using streamllm_ext::StreamReader;
using streamllm_ext::StreamllmRuntime;
using streamllm_ext::UpstreamLayoutDevice;

namespace {

void check_cuda(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

std::vector<uint8_t> read_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(1); }
    auto sz = f.tellg();
    f.seekg(0);
    std::vector<uint8_t> buf((size_t)sz);
    f.read((char *)buf.data(), sz);
    return buf;
}

inline float f16_to_f32(uint16_t h) {
    __half v;
    std::memcpy(&v, &h, 2);
    return __half2float(v);
}

// Estimate a conservative VRAM footprint for a streamllm GGUF so the
// pool can be sized without manual tuning. 2× margin for alignment +
// allocator fragmentation.
size_t estimate_pool_bytes(const StreamReader & r) {
    size_t total = 0;
    for (const auto & name : r.managed_tensor_names()) {
        const auto * L = r.layout(name);
        if (!L) continue;
        int32_t M  = (int32_t)L->shape[0];
        int32_t K  = L->padded_m;
        int32_t P  = (int32_t)L->chunk_bytes.size();
        int32_t Kg = (int32_t)L->n_groups_per_row;
        total += (size_t)(K / 32) * P * M * 4;  // q_weight u32
        total += (size_t)Kg * P * M * 2;        // alpha fp16
        total += (size_t)Kg * M * 2;            // q_bias fp16
    }
    return total * 2;
}

} // anon


int main(int argc, char ** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
            "usage: %s <gguf> <refs_dir> <tensor_1> [tensor_2 ...]\n", argv[0]);
        return 1;
    }
    const std::string gguf_path = argv[1];
    const std::string refs_dir  = argv[2];
    std::vector<std::string> check_names;
    for (int i = 3; i < argc; ++i) check_names.emplace_back(argv[i]);

    // --- 1. Parse GGUF + streamllm metadata ---
    gguf_init_params p{ true, nullptr };
    gguf_context * ctx = gguf_init_from_file(gguf_path.c_str(), p);
    if (!ctx) { std::fprintf(stderr, "open GGUF failed\n"); return 1; }
    auto reader_opt = StreamReader::from_gguf(ctx, gguf_path.c_str());
    if (!reader_opt) { std::fprintf(stderr, "no streamllm metadata\n"); return 1; }
    const auto & reader = *reader_opt;
    std::printf("GGUF: %s\n", gguf_path.c_str());
    std::printf("  version=%u encoder=%s group_size=%u P=%u managed=%zu\n",
                reader.global().version, reader.global().encoder.c_str(),
                reader.global().group_size, reader.global().target_precision,
                reader.managed_tensor_names().size());
    gguf_free(ctx);

    // --- 2. Size the pool + build runtime + bulk load ---
    size_t cap = estimate_pool_bytes(reader);
    std::printf("  estimated pool capacity: %.1f MB\n", (double)cap / 1024.0 / 1024.0);

    const char * sched_name = getenv("STREAMLLM_SCHEDULER");
    StreamllmRuntime runtime(cap, /*device=*/0, /*copy_stream=*/false,
                             sched_name);
    runtime.install(reader, gguf_path);
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(load)");

    std::printf("  scheduler=%s, VRAM %.1f / %.1f MB pool used\n",
                runtime.scheduler().name(),
                (double)runtime.pool().used_bytes() / 1024.0 / 1024.0,
                (double)runtime.pool().capacity_bytes() / 1024.0 / 1024.0);

    // --- 3. Spot-check kernel outputs for the requested tensors ---
    std::printf("\nSpot-check kernel output on %zu tensors:\n", check_names.size());
    int n_pass = 0;
    for (const auto & name : check_names) {
        // Drive the public plan → move → chunk_matmul path the same way
        // the hook does. Lazy's first-touch moves fire here; Eager's
        // plan carries no moves since everything uploaded at install.
        const streamllm_ext::Plan * plan = runtime.scheduler().plan(
            name, /*compute_stream=*/nullptr);
        if (plan == nullptr) {
            std::fprintf(stderr, "  %s: not in runtime (managed tensor name?)\n",
                         name.c_str());
            continue;
        }
        for (const auto & mv : plan->moves) {
            runtime.move_chunk(mv.wid, mv.cid, mv.src, mv.dst);
        }
        const UpstreamLayoutDevice * L = runtime.layout(name);
        if (L == nullptr || L->q_bias_fp16 == nullptr ||
            L->chunk_ptrs[0] == nullptr) {
            std::fprintf(stderr, "  %s: layout missing after plan\n",
                         name.c_str());
            continue;
        }

        // Load reference X + Y from disk.
        auto x_path = refs_dir + "/" + name + ".x.bin";
        auto y_path = refs_dir + "/" + name + ".y.bin";
        auto x_bytes = read_file(x_path);
        auto y_ref_bytes = read_file(y_path);
        if ((int)x_bytes.size() != L->K * 2) {
            std::fprintf(stderr,
                "  %s: X size %zu != K*2 (%d); skipped\n",
                name.c_str(), x_bytes.size(), L->K * 2);
            continue;
        }
        if ((int)y_ref_bytes.size() != L->M * 2) {
            std::fprintf(stderr,
                "  %s: Y size %zu != M*2 (%d); skipped\n",
                name.c_str(), y_ref_bytes.size(), L->M * 2);
            continue;
        }

        // Allocate X/Y on device + run kernel.
        void * d_x = nullptr;
        void * d_y = nullptr;
        check_cuda(cudaMalloc(&d_x, L->K * sizeof(__half)), "cudaMalloc(x)");
        check_cuda(cudaMalloc(&d_y, L->M * sizeof(__half)), "cudaMalloc(y)");
        check_cuda(cudaMemcpy(d_x, x_bytes.data(),
                              L->K * sizeof(__half), cudaMemcpyHostToDevice),
                   "cudaMemcpy(x)");

        // Exercise the public primitive rather than the raw kernel —
        // same math, goes through the runtime's chunk_matmul path.
        if (!runtime.chunk_matmul(name, plan->chunks, d_x, d_y,
                                  /*n_tokens=*/1,
                                  /*x_stride=*/0, /*y_stride=*/0,
                                  /*stream=*/nullptr)) {
            std::fprintf(stderr, "  %s: chunk_matmul returned false\n",
                         name.c_str());
            cudaFree(d_x); cudaFree(d_y);
            continue;
        }
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(kernel)");

        std::vector<uint16_t> y_host(L->M);
        check_cuda(cudaMemcpy(y_host.data(), d_y,
                              L->M * sizeof(__half), cudaMemcpyDeviceToHost),
                   "cudaMemcpy(y back)");
        cudaFree(d_x);
        cudaFree(d_y);

        const uint16_t * y_ref = reinterpret_cast<const uint16_t *>(y_ref_bytes.data());
        double sum2 = 0, ref2 = 0;
        for (int i = 0; i < L->M; ++i) {
            float fc = f16_to_f32(y_host[i]);
            float fr = f16_to_f32(y_ref[i]);
            double d = (double)fc - (double)fr;
            sum2 += d * d;
            ref2 += (double)fr * (double)fr;
        }
        double rel_l2 = std::sqrt(sum2) / std::max(1e-12, std::sqrt(ref2));
        bool pass = rel_l2 < 0.01;
        std::printf("  %-30s  rel_L2=%.6f  %s\n",
                    name.c_str(), rel_l2, pass ? "PASS" : "FAIL");
        if (pass) ++n_pass;
    }

    std::printf("\n%d / %zu spot-checks passed.\n", n_pass, check_names.size());
    return (n_pass == (int)check_names.size()) ? 0 : 1;
}
