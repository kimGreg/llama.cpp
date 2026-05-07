// streamllm-ext — cuBLAS F16 GEMM wrapper used for the prefill fast path.
//
// NAVER's LUT-GEMV kernel is strictly one X vector per launch. For a
// 2048-token prefill that's 2048 kernels per tensor × 252 managed
// tensors = 516k launches per forward and 2048× re-reads of the same
// planes from VRAM. For the decode case (n_tokens = 1) that's exactly
// what the kernel was designed for; for prefill we fall back here.
//
// The flow inside the hook for n_tokens > 1 is:
//   1. dequant_planes_to_f16  →  W_f16 [M, K]  (row-major, K inner)
//   2. batched_gemm_f16       →  Y = W @ X
//
// The dequant preserves the runtime precision dial — only the first
// ``precision`` planes contribute — so the quality knob is unchanged.
// cuBLAS is fine for this: it's a well-known primitive and nvidia's
// core BLAS. StreamLLM's novelty is the streaming + precision dial,
// not the core GEMM kernel.

#pragma once

#include <cuda_runtime.h>

namespace streamllm_ext {

// Lazy init on first call. Safe to call multiple times.
bool batched_gemm_init();

// Release cublasHandle. Called from runtime_hook clear().
void batched_gemm_shutdown();

// Y[M, N] = W[M, K] @ X[K, N].
//   W: row-major  [M, K]  (= col-major [K, M] with lda = K)
//   X: col-major  [K, N]  (= ggml src1 with ne[0]=K, ne[1]=N)
//   Y: col-major  [M, N]  (= ggml dst  with ne[0]=M, ne[1]=N)
// All tensors F16, contiguous, on-device. Stream-ordered.
//
// Returns false if cuBLAS fails (caller should fall back to the
// per-token GEMV path).
bool batched_gemm_f16(
    const void * W_f16, const void * X_f16, void * Y_f16,
    int M, int K, int N,
    cudaStream_t stream);

} // namespace streamllm_ext
