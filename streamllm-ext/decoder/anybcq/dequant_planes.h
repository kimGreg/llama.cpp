// streamllm-ext — dequantize AnyBCQ bit-plane weights back to a dense
// F16 tensor, usable as a standard-shape operand for cuBLAS GEMM.
//
// The per-token GEMV path (naver_gemv_launch) is strictly one kernel
// launch per input vector. For prefill (n_tokens ≫ 1) that's 2048×
// kernel-launch overhead *and* the same weight re-read from VRAM every
// token. This launcher materialises W_fp16 once, letting the hook fall
// back to a single cuBLAS F16 GEMM for batched inputs.
//
// The reconstruction exactly matches the Python reference
// (streamllm/encoders/impl/anybcq.py::reconstruct_weight):
//
//   W[m, k] = q_bias[g, m] + Σ_p α[p, g, m] · (2·bit_p(m,k) − 1)
//
// where g = k / group_size, bit_p(m,k) is the plane-p sign bit for
// output row m at input-col k (packed LSB-first into uint32 tiles of 32
// consecutive k values, as laid out by upstream_layout.cpp).
//
// Runtime precision dial is preserved: the caller passes ``precision``
// ≤ P_total and only the first ``precision`` planes contribute. A
// smaller precision simply reconstructs the truncated approximation the
// prefix-decodable format was designed to yield.

#pragma once

#include <cuda_runtime.h>

namespace streamllm_ext {

// Fills ``d_W_out`` with the F16 dense weight matrix at the requested
// precision. Shape: row-major [M, K] (K innermost), which is identical
// to the memory layout ggml uses for a 2-D tensor with
// ``ne = {K, M}``. The caller owns the allocation (must be at least
// ``M × K × sizeof(__half)`` bytes).
//
// ``d_q_weight_planes`` / ``d_alpha_planes`` are host-side arrays of
// ``precision`` device pointers, one per plane, laid out as described in
// naver_gemv.h. ``d_q_bias`` is the F16 bias array [K_groups × M].
//
// All work is enqueued on ``stream``. The launcher allocates two small
// device-side scratch buffers for the pointer arrays via
// ``cudaMallocAsync`` on the same stream and frees them at the end —
// same pattern naver_gemv_launch uses.
void launch_dequant_planes_f16(
    const void * const * d_q_weight_planes,
    const void * const * d_alpha_planes,
    const void *         d_q_bias,
    void *               d_W_out,
    int M, int K, int precision, int group_size,
    cudaStream_t        stream);

} // namespace streamllm_ext
