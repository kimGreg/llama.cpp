// streamllm-ext / decoder / anybcq — public GEMM interface (prefill).
//
// For n_tokens > 1 the per-token GEMV is wasteful — a 2048-token
// prefill would launch 2048 kernels per tensor and re-read the weight
// from VRAM 2048 times. This header consolidates the prefill path:
//
//   1. ``launch_dequant_planes_f16`` — materialise W[M,K] in fp16 from
//      the bitplanes + α + β, honoring the runtime precision dial.
//   2. ``batched_gemm_f16``           — Y = W @ X via cublasGemmEx.
//
// Plus the f32 ↔ f16 cast launchers used by the hook to bridge ggml's
// f32 activations into the kernel's f16 inputs.

#pragma once

#include <cuda_runtime.h>

namespace streamllm_ext {

// =====================================================================
// Plane → fp16 dense weight (precision-dialed).
//
//   W[m, k] = q_bias[g, m] + Σ_p α[p, g, m] · (2·bit_p(m,k) − 1)
//
// where g = k / group_size, exactly matching the Python reference
// (streamllm/encoders/impl/anybcq.py::reconstruct_weight).
//
// ``d_W_out`` is row-major [M, K] (K innermost) = same layout ggml
// uses for ne = {K, M}. Caller owns at least M × K × sizeof(__half) bytes.
// Setting ``precision`` < P_total reconstructs the truncated prefix.
// =====================================================================
void launch_dequant_planes_f16(
    const void * const * d_q_weight_planes,
    const void * const * d_alpha_planes,
    const void *         d_q_bias,
    void *               d_W_out,
    int M, int K, int precision, int group_size,
    cudaStream_t         stream);


// =====================================================================
// cuBLAS F16 GEMM wrapper. StreamLLM's novelty is in the streaming +
// precision dial, not the core GEMM kernel.
// =====================================================================

// Lazy init on first call.  Safe to call multiple times.
bool batched_gemm_init();

// Release cublasHandle. Called from runtime_hook clear().
void batched_gemm_shutdown();

// Y[M, N] = W[M, K] @ X[K, N].
//   W: row-major  [M, K]   (= col-major [K, M] with lda = K)
//   X: col-major  [K, N]   (= ggml src1 with ne[0]=K, ne[1]=N)
//   Y: col-major  [M, N]   (= ggml dst  with ne[0]=M, ne[1]=N)
// All tensors F16, contiguous, on-device. Stream-ordered. Returns
// false on cuBLAS failure (caller should fall back to per-token GEMV).
bool batched_gemm_f16(
    const void * W_f16, const void * X_f16, void * Y_f16,
    int M, int K, int N,
    cudaStream_t stream);


// =====================================================================
// F32 ↔ F16 cast launchers (used by the hook to bridge ggml f32 activations).
// =====================================================================
void launch_f32_to_f16(const void * src_f32, void * dst_f16,
                       int n, cudaStream_t stream);
void launch_f16_to_f32(const void * src_f16, void * dst_f32,
                       int n, cudaStream_t stream);

} // namespace streamllm_ext
