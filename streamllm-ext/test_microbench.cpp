// streamllm-ext — microbench for the two primitives.
//
// Measures:
//   1. Kernel execution: μs per naver_gemv_launch, effective GB/s
//      over the q_weight buffer read.
//   2. Chunk H2D transfer: μs per move_chunk (RAM→VRAM), effective
//      GB/s.
//
// Usage:
//   streamllm-microbench <gguf> [n_iters]
//
// Reports per representative tensor size (M × K × P). For Qwen3-1.7B:
//   attn_q        (M=2048, K=2048, P=8)   — small square
//   attn_output   (M=2048, K=2048, P=8)   — same
//   ffn_gate/up   (M=6144, K=2048, P=8)   — tall rectangle
//   ffn_down      (M=2048, K=6144, P=8)   — wide rectangle

#include "stream_reader.h"
#include "upstream_layout.h"
#include "vram_pool.h"
#include "naver_gemv.h"

#include <gguf.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

using streamllm_ext::StreamReader;
using streamllm_ext::TensorLayout;
using streamllm_ext::UpstreamLayoutHost;
using streamllm_ext::VramChunkPool;
using streamllm_ext::build_upstream_layout_host;

namespace {

void check_cuda(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

std::vector<uint8_t> pread_range(const std::string & path, int64_t off, int64_t n) {
    std::ifstream f(path, std::ios::binary);
    f.seekg(off);
    std::vector<uint8_t> buf((size_t)n);
    f.read((char *)buf.data(), n);
    return buf;
}

struct KernelStats {
    int    M, K, P;
    double us_per_call;
    double q_weight_bytes_per_call;  // what the kernel reads each call
    double gbps;                     // effective BW over q_weight read
};

struct TransferStats {
    size_t chunk_bytes;
    double us_per_call;
    double gbps;
};

KernelStats bench_kernel(VramChunkPool & pool,
                          const std::string & tensor_name,
                          const UpstreamLayoutHost & host,
                          int n_iters)
{
    const int M = host.n, K = host.padded_m, P = host.n_chunks;

    // Upload the plane chunks + q_bias. Each plane chunk is
    // (signs + alpha) packed contiguously; the launcher splits at
    // qw_bytes_per_chunk.
    const size_t chunk_bytes  = host.bytes_per_chunk;
    const size_t q_bias_bytes = host.q_bias.size() * sizeof(uint16_t);

    auto h_qb = pool.load(tensor_name, 0, host.q_bias.data(), q_bias_bytes);
    std::vector<const void *> qw_ptrs(P), a_ptrs(P);
    for (int p = 0; p < P; ++p) {
        auto h = pool.load(tensor_name, 100 + p, host.chunks[p].data(), chunk_bytes);
        qw_ptrs[p] = h.device_ptr;
        a_ptrs[p]  = (const uint8_t *) h.device_ptr + host.qw_bytes_per_chunk;
    }

    // Allocate X / Y on device.
    void * d_x; void * d_y;
    check_cuda(cudaMalloc(&d_x, (size_t)K * sizeof(__half)), "cudaMalloc(x)");
    check_cuda(cudaMalloc(&d_y, (size_t)M * sizeof(__half)), "cudaMalloc(y)");
    check_cuda(cudaMemset(d_x, 0, (size_t)K * sizeof(__half)), "cudaMemset(x)");

    // Warm up.
    for (int i = 0; i < 10; ++i) {
        streamllm_ext::naver_gemv_launch(
            d_x, d_y,
            qw_ptrs.data(), a_ptrs.data(), h_qb.device_ptr,
            M, K, P, host.group_size, nullptr);
    }
    check_cuda(cudaDeviceSynchronize(), "warmup sync");

    // Time n_iters.
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < n_iters; ++i) {
        streamllm_ext::naver_gemv_launch(
            d_x, d_y,
            qw_ptrs.data(), a_ptrs.data(), h_qb.device_ptr,
            M, K, P, host.group_size, nullptr);
    }
    cudaEventRecord(t1);
    check_cuda(cudaEventSynchronize(t1), "eventSync");
    float ms = 0.f;
    cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);

    cudaFree(d_x); cudaFree(d_y);

    KernelStats s{};
    s.M = M; s.K = K; s.P = P;
    s.us_per_call = (double)ms * 1000.0 / n_iters;
    // Each kernel reads all P planes' q_weight (K/32 × M uint32 per plane)
    // plus P planes of alpha (K_groups × M fp16) plus one q_bias load.
    // The q_weight dominates; report BW over that.
    const double qw_bytes = (double)(K / 32) * M * P * sizeof(uint32_t);
    const double alpha_bytes = (double)host.K_groups * M * P * sizeof(uint16_t);
    s.q_weight_bytes_per_call = qw_bytes + alpha_bytes;
    s.gbps = (s.q_weight_bytes_per_call) / (s.us_per_call * 1e-6) / 1e9;
    return s;
}

TransferStats bench_move(VramChunkPool & pool,
                          const std::string & tensor_name,
                          const UpstreamLayoutHost & host,
                          int n_iters)
{
    // Time the "evict + load one plane chunk" cycle. Use a throwaway
    // cid so we don't disturb the kernel bench's uploads.
    const size_t chunk_bytes = host.bytes_per_chunk;
    const int cid = 999;
    const std::vector<uint8_t> & src = host.chunks[0];

    // Warm up: make sure the slot is allocated/freed a few times
    // so the pool's free list is stable.
    for (int i = 0; i < 4; ++i) {
        pool.load(tensor_name, cid, src.data(), chunk_bytes);
        pool.evict(tensor_name, cid);
    }
    check_cuda(cudaDeviceSynchronize(), "move warmup");

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < n_iters; ++i) {
        pool.load(tensor_name, cid, src.data(), chunk_bytes);
        pool.evict(tensor_name, cid);
    }
    cudaEventRecord(t1);
    check_cuda(cudaEventSynchronize(t1), "move eventSync");
    float ms = 0.f;
    cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);

    TransferStats s{};
    s.chunk_bytes = chunk_bytes;
    s.us_per_call = (double)ms * 1000.0 / n_iters;
    s.gbps = ((double)chunk_bytes) / (s.us_per_call * 1e-6) / 1e9;
    return s;
}

void print_kernel_header() {
    std::printf("\n== Kernel execution (naver_gemv_launch) ==\n");
    std::printf("%-35s %6s %6s %3s  %9s %10s %9s\n",
                "tensor", "M", "K", "P",
                "μs/call", "MB read", "GB/s");
}
void print_kernel_row(const std::string & name, const KernelStats & s) {
    std::printf("%-35s %6d %6d %3d  %9.2f %10.2f %9.2f\n",
                name.c_str(), s.M, s.K, s.P,
                s.us_per_call,
                s.q_weight_bytes_per_call / 1024.0 / 1024.0,
                s.gbps);
}

void print_move_header() {
    std::printf("\n== Chunk H2D (VramChunkPool::load, pageable src) ==\n");
    std::printf("%-35s %10s %9s %9s\n",
                "tensor", "chunk MB", "μs/call", "GB/s");
}
void print_move_row(const std::string & name, const TransferStats & s) {
    std::printf("%-35s %10.2f %9.2f %9.2f\n",
                name.c_str(),
                (double)s.chunk_bytes / 1024.0 / 1024.0,
                s.us_per_call, s.gbps);
}

}  // anon


int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
            "usage: %s <gguf> [n_iters=256] [tensor1 tensor2 ...]\n",
            argv[0]);
        return 1;
    }
    const std::string gguf_path = argv[1];
    const int n_iters = (argc >= 3) ? std::atoi(argv[2]) : 256;

    gguf_init_params p{ /*no_alloc=*/true, nullptr };
    gguf_context * ctx = gguf_init_from_file(gguf_path.c_str(), p);
    if (!ctx) { std::fprintf(stderr, "failed to open %s\n", gguf_path.c_str()); return 1; }

    auto reader_opt = StreamReader::from_gguf(ctx, gguf_path.c_str());
    gguf_free(ctx);
    if (!reader_opt) {
        std::fprintf(stderr, "no streamllm.* block in %s\n", gguf_path.c_str());
        return 1;
    }
    const auto & reader = *reader_opt;

    // Default tensor set: one of each representative shape.
    std::vector<std::string> tensors;
    if (argc > 3) {
        for (int i = 3; i < argc; ++i) tensors.emplace_back(argv[i]);
    } else {
        tensors = {
            "blk.13.attn_q.weight",        // M=square×1
            "blk.13.attn_output.weight",   // M=square×1
            "blk.13.ffn_gate.weight",      // tall
            "blk.13.ffn_up.weight",        // tall
            "blk.13.ffn_down.weight",      // wide
        };
    }

    // Pool sized generously: enough for every benched tensor's full
    // plane set + one scratch slot for the move benchmark.
    size_t cap = 0;
    for (const auto & n : tensors) {
        const auto * L = reader.layout(n);
        if (!L) continue;
        int tm = (int)L->shape[0];
        int tk = (int)L->padded_m;
        int tp = (int)L->chunk_bytes.size();
        int tkg = (int)L->n_groups_per_row;
        size_t per = (size_t)(tk / 32) * tm * tp * 4 +
                     (size_t)tkg * tm * tp * 2 +
                     (size_t)tkg * tm * 2;
        cap += per;
    }
    cap += 64 * 1024 * 1024;  // scratch
    cap = cap * 3 / 2;        // 1.5× margin

    VramChunkPool pool(cap, /*device=*/0, /*copy_stream=*/false);

    std::printf("== Setup ==\n");
    std::printf("gguf   : %s\n", gguf_path.c_str());
    std::printf("iters  : %d per config\n", n_iters);
    std::printf("pool   : %zu MB\n", cap / 1024 / 1024);

    std::vector<std::pair<std::string, UpstreamLayoutHost>> built;
    for (const auto & name : tensors) {
        const auto * lay = reader.layout(name);
        if (!lay) {
            std::fprintf(stderr, "skip: %s (not in GGUF)\n", name.c_str());
            continue;
        }
        auto raw = pread_range(gguf_path, lay->tensor_offset, lay->tensor_nbytes);
        auto host = build_upstream_layout_host(*lay, raw.data(), reader.global().group_size);
        built.emplace_back(name, std::move(host));
    }

    print_kernel_header();
    for (auto & [name, host] : built) {
        auto s = bench_kernel(pool, name, host, n_iters);
        print_kernel_row(name, s);
    }

    print_move_header();
    for (auto & [name, host] : built) {
        auto s = bench_move(pool, name, host, n_iters);
        print_move_row(name, s);
    }

    std::printf("\n");
    return 0;
}
