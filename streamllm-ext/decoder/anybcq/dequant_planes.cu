// streamllm-ext — AnyBCQ plane → F16 dense weight reconstruction.
// See dequant_planes.h for the formula + layout.

#include "dequant_planes.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace streamllm_ext {

namespace {

// One thread per output element W[m, k]. precision ≤ 16 so the inner
// loop is short; unrolling is bounded. Weight planes are read along
// the M-innermost stride (qw[kt*M + m]) which matches the kernel's
// own reader and keeps the 128-byte sector fully utilised across a
// warp striding in m.
__global__ void k_dequant_planes_f16(
    const uint32_t * const * __restrict__ q_weight_planes,
    const __half   * const * __restrict__ alpha_planes,
    const __half   * __restrict__ q_bias,
    __half *         __restrict__ W_out,
    int M, int K, int precision, int group_size)
{
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M || k >= K) return;

    const int g  = k / group_size;
    const int kt = k / 32;
    const int s  = k % 32;

    float w = __half2float(q_bias[(size_t)g * M + m]);
    for (int p = 0; p < precision; ++p) {
        uint32_t word = q_weight_planes[p][(size_t)kt * M + m];
        int      bit  = (int)((word >> s) & 1u);
        float    sign = bit ? 1.0f : -1.0f;
        float    a    = __half2float(alpha_planes[p][(size_t)g * M + m]);
        w += sign * a;
    }
    W_out[(size_t)m * K + k] = __float2half(w);
}

inline void check_cuda(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "dequant_planes: CUDA error in %s: %s\n",
                     what, cudaGetErrorString(e));
        std::exit(1);
    }
}

} // anonymous

void launch_dequant_planes_f16(
    const void * const * d_q_weight_planes,
    const void * const * d_alpha_planes,
    const void *         d_q_bias,
    void *               d_W_out,
    int M, int K, int precision, int group_size,
    cudaStream_t         stream)
{
    // Upload host pointer arrays. Same pattern as naver_gemv_launch:
    // cudaMallocAsync + cudaMemcpyAsync on the caller's stream, then
    // cudaFreeAsync after the kernel finishes. All ordered on ``stream``.
    const size_t ptr_bytes = (size_t)precision * sizeof(void *);
    void ** d_qw_ptrs = nullptr;
    void ** d_a_ptrs  = nullptr;
    check_cuda(cudaMallocAsync((void **)&d_qw_ptrs, ptr_bytes, stream),
               "cudaMallocAsync(qw_ptrs)");
    check_cuda(cudaMallocAsync((void **)&d_a_ptrs,  ptr_bytes, stream),
               "cudaMallocAsync(a_ptrs)");
    check_cuda(
        cudaMemcpyAsync(d_qw_ptrs, d_q_weight_planes, ptr_bytes,
                        cudaMemcpyHostToDevice, stream),
        "cudaMemcpyAsync(qw_ptrs)");
    check_cuda(
        cudaMemcpyAsync(d_a_ptrs, d_alpha_planes, ptr_bytes,
                        cudaMemcpyHostToDevice, stream),
        "cudaMemcpyAsync(a_ptrs)");

    // 32 × 8 threads per block: x sweeps k contiguously (coalesces on
    // W_out writes since W is row-major with K innermost), y sweeps m.
    dim3 block(32, 8);
    dim3 grid((K + block.x - 1) / block.x,
              (M + block.y - 1) / block.y);

    k_dequant_planes_f16<<<grid, block, 0, stream>>>(
        (const uint32_t * const *) d_qw_ptrs,
        (const __half   * const *) d_a_ptrs,
        (const __half *)           d_q_bias,
        (__half *)                 d_W_out,
        M, K, precision, group_size);

    check_cuda(cudaFreeAsync(d_qw_ptrs, stream), "cudaFreeAsync(qw_ptrs)");
    check_cuda(cudaFreeAsync(d_a_ptrs,  stream), "cudaFreeAsync(a_ptrs)");
}

} // namespace streamllm_ext
