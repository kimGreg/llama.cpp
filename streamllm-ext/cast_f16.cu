// streamllm-ext — minimal F32 ↔ F16 cast kernels.
//
// Needed because Qwen3 (and most llama.cpp-loaded archs) flow
// activations through the graph as F32, but NAVER's LUT-GEMV kernel
// only accepts F16 X and produces F16 Y. The runtime hook allocates
// two small scratch buffers per managed mul_mat — K fp16 for X,
// M fp16 for Y — and uses these kernels to bridge.
//
// A fused version would skip the round-trip and live inside the LUT
// kernel itself; for MVR the extra global-memory touch is negligible
// compared to the kernel's LUT build + accumulate.

#include "cast_f16.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace streamllm_ext {

namespace {

__global__ void k_f32_to_f16(const float * __restrict__ src,
                             __half      * __restrict__ dst,
                             int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

__global__ void k_f16_to_f32(const __half * __restrict__ src,
                             float        * __restrict__ dst,
                             int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __half2float(src[i]);
}

} // anonymous

void launch_f32_to_f16(const void * src_f32, void * dst_f16,
                       int n, cudaStream_t stream) {
    const int block = 256;
    const int grid  = (n + block - 1) / block;
    k_f32_to_f16<<<grid, block, 0, stream>>>(
        (const float *) src_f32, (__half *) dst_f16, n);
}

void launch_f16_to_f32(const void * src_f16, void * dst_f32,
                       int n, cudaStream_t stream) {
    const int block = 256;
    const int grid  = (n + block - 1) / block;
    k_f16_to_f32<<<grid, block, 0, stream>>>(
        (const __half *) src_f16, (float *) dst_f32, n);
}

} // namespace streamllm_ext
