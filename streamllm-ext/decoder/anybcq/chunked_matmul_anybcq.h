// streamllm-ext / decoder / anybcq — chunked-matmul dispatch (AnyBCQ).
//
// The runtime's ``chunk_matmul`` and ``chunk_matmul_batched`` are thin
// wrappers around these functions. They sit on top of the NAVER GEMV
// kernel + the dequant-planes path and own the small amount of glue
// that turns a chunk-list into the kernel's per-plane pointer arrays.
//
// They live in the AnyBCQ decoder layer because every signature
// reference (``UpstreamLayoutDevice``, the qw / alpha pointer split,
// the prefix-plane fast path) is AnyBCQ-format-specific. core/runtime
// stays algorithm-blind by delegating here.

#pragma once

#include "naver_gemv.h"        // NaverKernelScratch
#include "vram_pool.h"         // StreamHandle alias

#include <cstddef>
#include <vector>

namespace streamllm_ext {

// Forward declaration. UpstreamLayoutDevice's full definition lives
// in core/runtime.h today (will be moved to upstream_layout.h in a
// follow-up so this dep flows decoder → decoder, not decoder → core).
struct UpstreamLayoutDevice;

namespace anybcq {

// Per-token decode path. Reads the ``chunks`` list, builds
// (qw_ptrs[], alpha_ptrs[]) by indexing ``L.chunk_ptrs`` and adding
// ``L.qw_bytes_per_chunk`` for each plane's alpha base, then calls
// ``naver_gemv_launch`` once per token. ``d_qw_dev`` / ``d_a_dev`` are
// the device-side pointer-array fast paths (used only when ``chunks``
// is a contiguous prefix [PLANE_0..PLANE_{P-1}]; passing nullptrs
// disables the fast path and falls back to a per-call H2D of the
// pointer arrays).
//
// Returns false if no plane chunks were resident, q_bias is missing,
// or any chunk had a null device pointer.
bool chunk_matmul(
    const UpstreamLayoutDevice & L,
    const void * const * d_qw_dev,      // optional fast-path
    const void * const * d_alpha_dev,   // optional fast-path
    const std::vector<int> & chunks,
    const void * X_fp16, void * Y_fp16,
    int n_tokens,
    size_t x_stride_bytes, size_t y_stride_bytes,
    NaverKernelScratch * scratch,
    StreamHandle compute_stream);

// Batched / prefill path. Either dequants planes into a dense fp16
// scratch and calls cuBLAS GEMM (default), or runs the fused chunked
// LUT-GEMM kernel when ``STREAMLLM_BATCHED_BACKEND=fused`` is set.
// ``w_scratch_f16`` is required for the cuBLAS path (size = M × K
// fp16); ignored under fused.
//
// Returns false on a wrong/empty chunk list or missing q_bias.
bool chunk_matmul_batched(
    const UpstreamLayoutDevice & L,
    const std::vector<int> & chunks,
    const void * X_fp16, void * Y_fp16,
    int n_tokens,
    void * w_scratch_f16,
    StreamHandle compute_stream);

}  // namespace anybcq
}  // namespace streamllm_ext
