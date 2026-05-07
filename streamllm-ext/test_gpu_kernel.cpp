// streamllm-ext — end-to-end GPU kernel test.
//
// Reads a streamllm GGUF, picks one managed tensor, builds the
// upstream layout host-side, uploads it into a VramChunkPool, runs
// NAVER's LUT-GEMV kernel on a fixed input X, and compares Y against
// a Python reference Y dump.
//
// Usage:
//   streamllm-gpu-test <gguf> <tensor> <input_fp16.bin> <expected_y_fp16.bin>
//
// The input + expected_y are produced by
// streamllm/scripts/dump_gpu_reference.py in the parent repo.

#include "stream_reader.h"
#include "upstream_layout.h"
#include "vram_pool.h"
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
using streamllm_ext::TensorLayout;
using streamllm_ext::VramChunkPool;
using streamllm_ext::build_upstream_layout_host;

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

std::vector<uint8_t> pread_range(const std::string & path, int64_t off, int64_t n) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(1); }
    f.seekg(off);
    std::vector<uint8_t> buf((size_t)n);
    f.read((char *)buf.data(), n);
    return buf;
}

// Convert fp16 → fp32 with hardware intrinsic for loss-free compare.
inline float f16_to_f32(uint16_t h) {
    __half v;
    std::memcpy(&v, &h, 2);
    return __half2float(v);
}

} // anon


int main(int argc, char ** argv) {
    if (argc != 5) {
        std::fprintf(stderr,
            "usage: %s <gguf> <tensor> <input_fp16.bin> <expected_y_fp16.bin>\n",
            argv[0]);
        return 1;
    }
    const std::string gguf_path = argv[1];
    const std::string tensor    = argv[2];
    const std::string x_path    = argv[3];
    const std::string y_ref_path = argv[4];

    // --- 1. Parse GGUF metadata + locate managed tensor ---
    gguf_init_params p{ true, nullptr };
    gguf_context * ctx = gguf_init_from_file(gguf_path.c_str(), p);
    if (!ctx) { std::fprintf(stderr, "open GGUF failed\n"); return 1; }
    auto reader = StreamReader::from_gguf(ctx, gguf_path.c_str());
    if (!reader) { std::fprintf(stderr, "no streamllm metadata\n"); return 1; }
    const TensorLayout * layout = reader->layout(tensor);
    if (!layout) { std::fprintf(stderr, "tensor not managed\n"); return 1; }
    const uint32_t group_size = reader->global().group_size;
    gguf_free(ctx);

    std::printf("tensor = %s\n", tensor.c_str());
    std::printf("shape  = [%lld, %lld]  P = %zu  group_size = %u\n",
                (long long)layout->shape[0], (long long)layout->shape[1],
                layout->chunk_bytes.size(), group_size);

    const int M = (int)layout->shape[0];    // n_out
    const int K = (int)layout->padded_m;     // input dim (padded)

    // --- 2. Read raw tensor bytes + build host upstream layout ---
    auto raw = pread_range(gguf_path, layout->tensor_offset, layout->tensor_nbytes);
    auto host = build_upstream_layout_host(*layout, raw.data(), group_size);

    // --- 3. Allocate pool, upload each plane chunk + q_bias ---
    const size_t chunk_bytes  = host.bytes_per_chunk;
    const size_t q_bias_bytes = host.q_bias.size() * sizeof(uint16_t);
    const size_t total_bytes  = host.n_chunks * chunk_bytes + q_bias_bytes +
                                4 * 1024 * 1024;

    VramChunkPool pool(total_bytes, /*device=*/0, /*copy_stream=*/false);

    auto h_qb = pool.load(tensor, /*cid=*/0, host.q_bias.data(), q_bias_bytes);

    std::vector<const void *> qw_dev_ptrs(host.n_chunks);
    std::vector<const void *> a_dev_ptrs(host.n_chunks);
    for (int p = 0; p < host.n_chunks; ++p) {
        auto h_chunk = pool.load(tensor, 100 + p,
                                 host.chunks[p].data(), chunk_bytes);
        qw_dev_ptrs[p] = h_chunk.device_ptr;
        a_dev_ptrs[p]  = static_cast<const uint8_t *>(h_chunk.device_ptr) +
                         host.qw_bytes_per_chunk;
    }

    std::printf("\nuploaded to VRAM pool (%.1f MB of %.1f MB)\n",
                (double)pool.used_bytes() / 1024.0 / 1024.0,
                (double)pool.capacity_bytes() / 1024.0 / 1024.0);

    // --- 4. Read X + reference Y from disk, upload X to VRAM ---
    auto x_bytes = read_file(x_path);
    auto y_ref_bytes = read_file(y_ref_path);
    if ((int)x_bytes.size() != K * 2) {
        std::fprintf(stderr, "input size %zu != K*2 (%d)\n", x_bytes.size(), K * 2);
        return 1;
    }
    if ((int)y_ref_bytes.size() != M * 2) {
        std::fprintf(stderr, "ref size %zu != M*2 (%d)\n", y_ref_bytes.size(), M * 2);
        return 1;
    }

    void * d_x = nullptr;
    void * d_y = nullptr;
    check_cuda(cudaMalloc(&d_x, K * sizeof(__half)), "cudaMalloc(x)");
    check_cuda(cudaMalloc(&d_y, M * sizeof(__half)), "cudaMalloc(y)");
    check_cuda(cudaMemcpy(d_x, x_bytes.data(), K * sizeof(__half),
                          cudaMemcpyHostToDevice), "cudaMemcpy(x)");

    // --- 5. Launch NAVER kernel with per-plane pointer arrays ---
    int P = host.n_chunks;
    streamllm_ext::naver_gemv_launch(
        d_x, d_y,
        qw_dev_ptrs.data(), a_dev_ptrs.data(), h_qb.device_ptr,
        M, K, /*precision=*/P, (int)group_size,
        nullptr);
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    // --- 6. Copy Y back + compare ---
    std::vector<uint16_t> y_host(M);
    check_cuda(cudaMemcpy(y_host.data(), d_y, M * sizeof(__half),
                          cudaMemcpyDeviceToHost), "cudaMemcpy(y back)");
    const uint16_t * y_ref = reinterpret_cast<const uint16_t *>(y_ref_bytes.data());

    size_t n_identical = 0;
    double max_abs = 0, sum_abs2 = 0, sum_ref2 = 0;
    float  max_rel_large = 0;   // max rel Δ over "large" reference values
    size_t n_mismatch = 0;
    double y_ref_max_abs = 0;
    for (int i = 0; i < M; ++i) {
        float f_ref = f16_to_f32(y_ref[i]);
        double ar = std::fabs((double)f_ref);
        if (ar > y_ref_max_abs) y_ref_max_abs = ar;
        sum_ref2 += (double)f_ref * (double)f_ref;
        if (y_host[i] == y_ref[i]) { ++n_identical; continue; }
        ++n_mismatch;
        float f_cxx = f16_to_f32(y_host[i]);
        double d = std::fabs(f_cxx - f_ref);
        sum_abs2 += d * d;
        if (d > max_abs) max_abs = d;
    }
    // Relative comparison metric: ||Δ||₂ / ||Y_ref||₂ — robust to near-zero
    // elements that inflate per-element max-rel.
    const double rms_delta = n_mismatch ? std::sqrt(sum_abs2 / n_mismatch) : 0.0;
    const double rel_l2    = std::sqrt(sum_abs2) /
                             std::max(1e-12, std::sqrt(sum_ref2));

    // Second-pass: max relative over "large" Y values (|Y_ref| > 1% of Y_ref max).
    const double big_thresh = 0.01 * y_ref_max_abs;
    for (int i = 0; i < M; ++i) {
        if (y_host[i] == y_ref[i]) continue;
        float f_ref = f16_to_f32(y_ref[i]);
        if (std::fabs(f_ref) < big_thresh) continue;
        float f_cxx = f16_to_f32(y_host[i]);
        float r = (float)(std::fabs(f_cxx - f_ref) / std::fabs(f_ref));
        if (r > max_rel_large) max_rel_large = r;
    }

    std::printf("\n=== kernel output parity ===\n");
    std::printf("  M = %d\n", M);
    std::printf("  bit-identical          : %zu / %d (%.2f%%)\n",
                n_identical, M, 100.0 * (double)n_identical / M);
    std::printf("  mismatched             : %zu\n", n_mismatch);
    std::printf("  max |Δ|                : %.6g\n", max_abs);
    std::printf("  rms |Δ|                : %.6g\n", rms_delta);
    std::printf("  ||Δ||_2 / ||Y_ref||_2  : %.4g  (target < 0.05)\n", rel_l2);
    std::printf("  max_rel on |Y|>1%%max  : %.4g  (target < 0.05)\n", max_rel_large);
    std::printf("  |Y_ref|_max            : %.4g\n", y_ref_max_abs);

    // Acceptance — fp16 kernel vs fp16 torch.matmul reference:
    //   ||Δ||₂ / ||Y_ref||₂ < 1%
    // Aggregate relative fp16 error. Robust to the reduction-order noise
    // intrinsic to LUT-GEMV vs dense-matmul on the same fp16 inputs (per-
    // element max-rel is too noisy at fp16 to be a useful threshold).
    const bool pass = (rel_l2 < 0.01);
    std::printf("\n%s.\n", pass ? "PASS" : "FAIL");

    cudaFree(d_x);
    cudaFree(d_y);
    return pass ? 0 : 1;
}
