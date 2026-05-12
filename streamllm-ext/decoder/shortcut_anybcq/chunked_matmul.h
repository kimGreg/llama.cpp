// streamllm-ext / decoder / shortcut_anybcq — chunked-matmul dispatch.
//
// ────────────────────────────────────────────────────────────────────
// FUTURE: DenseExecutor — no runtime caller in M1.
// ────────────────────────────────────────────────────────────────────
// Mode A M1 retired the dense-managed runtime path entirely. The
// ``streamllm_try_cuda_mul_mat`` hook now aborts loudly (criterion
// 10) on any managed canonical that surfaces as a dense MUL_MAT. The
// kernel + per-encoder pointer staging in this TU is kept on disk
// because future dense streaming will land as a separate
// ``DenseExecutor`` riding the same Runtime / Scheduler /
// ChunkedTensor framework — at that point THIS kernel is the natural
// per-tile compute back-end. Until then, no production code calls
// into it. Do not re-enable a dense fallback by routing
// ``streamllm_try_cuda_mul_mat`` through here — DenseExecutor is the
// approved future path.
// ────────────────────────────────────────────────────────────────────
//
// Shortcut layout: each chunk packs ONE plane's signs + that plane's α
// scalar inline ([signs | α]); β is in a tensor-wide kCidQBias chunk
// pinned at install. Decode is one naver_gemv_launch per token across
// the resident planes; prefill dequants planes into a dense fp16
// scratch and goes through cuBLAS GEMM.
//
// The kernels themselves are shared with the any-prec path
// (``decoder/anybcq/anybcq_gemv.h`` and ``anybcq_gemm.h``); only the
// per-encoder pointer staging — what address α_i resolves to — is
// shortcut-specific and lives here. The any-prec equivalent goes
// through the MoE-fused dispatcher in ``qwen3/`` and the per-plane
// ``update_anyprec_after_load_*`` writers.

#pragma once

#include "anybcq_gemv.h"  // NaverKernelScratch (kernel + scratch decl)
#include "vram_pool.h"    // StreamHandle alias

#include <cstddef>
#include <string>
#include <vector>

namespace streamllm_ext {

// Forward declaration; full definition in core/runtime.h.
struct UpstreamLayoutDevice;
class StreamllmRuntime;

namespace shortcut_anybcq {

// Per-token decode path. Reads ``chunks``, builds per-plane (qw, α)
// pointer arrays from ``L.chunk_ptrs`` + ``L.qw_bytes_per_chunk``, and
// loops naver_gemv_launch over n_tokens. ``d_qw_dev`` / ``d_alpha_dev``
// are the per-tensor device-side pointer arrays — used directly when
// the chunk list is a contiguous prefix [0..P-1] (the fast path);
// pass nullptrs to fall back to a per-call H2D.
// Returns false if q_bias is missing or any chunk pointer is null.
bool chunk_matmul(
    const UpstreamLayoutDevice & L,
    const void * const * d_qw_dev,
    const void * const * d_alpha_dev,
    const std::vector<int> & chunks,
    const void * X_fp16, void * Y_fp16,
    int n_tokens,
    size_t x_stride_bytes, size_t y_stride_bytes,
    NaverKernelScratch * scratch,
    StreamHandle compute_stream);

// Batched / prefill path. Default: dequant planes → fp16 W + cuBLAS GEMM.
// STREAMLLM_BATCHED_BACKEND=fused: native chunked LUT-GEMM (no W
// materialisation). w_scratch_f16 is required for the default path
// (M × K fp16 bytes); ignored under fused.
bool chunk_matmul_batched(
    const UpstreamLayoutDevice & L,
    const std::vector<int> & chunks,
    const void * X_fp16, void * Y_fp16,
    int n_tokens,
    void * w_scratch_f16,
    StreamHandle compute_stream);


// =====================================================================
// Higher-level wrappers — model-layer call site.
//
// Take a runtime + wid, do the framework-level prep (wait_on_stream
// for each chunk, translate cid list → plane indices, look up the
// per-Entry layout + per-plane pointer arrays + scratch), then
// dispatch to the kernel-level helpers above. These exist so the
// model layer (qwen3) calls one function instead of orchestrating
// half of the runtime's internals inline. Core does NOT expose
// chunk_matmul itself — it stays a "dumb" pool.
// =====================================================================

bool chunk_matmul_for_wid(
    StreamllmRuntime &      rt,
    const std::string &     wid,
    const std::vector<int> & chunks,
    const void * X_fp16, void * Y_fp16,
    int n_tokens,
    size_t x_stride_bytes, size_t y_stride_bytes,
    StreamHandle compute_stream);

bool chunk_matmul_batched_for_wid(
    StreamllmRuntime &      rt,
    const std::string &     wid,
    const std::vector<int> & chunks,
    const void * X_fp16, void * Y_fp16,
    int n_tokens,
    void * w_scratch_f16,
    StreamHandle compute_stream);

}  // namespace shortcut_anybcq
}  // namespace streamllm_ext
