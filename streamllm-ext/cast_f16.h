// streamllm-ext — F32 ↔ F16 cast launchers for the runtime hook.
//
// Thin wrappers around two trivial CUDA kernels; called from
// runtime_hook.cpp (plain C++, no CUDA compiler), which means these
// launchers have to live in a .cu TU compiled by nvcc.

#pragma once

#include <cuda_runtime.h>

namespace streamllm_ext {

// ``dst_f16 = __float2half(src_f32)``. ``n`` elements. No-op on n == 0.
void launch_f32_to_f16(const void * src_f32, void * dst_f16,
                       int n, cudaStream_t stream);

// ``dst_f32 = __half2float(src_f16)``. ``n`` elements.
void launch_f16_to_f32(const void * src_f16, void * dst_f32,
                       int n, cudaStream_t stream);

} // namespace streamllm_ext
