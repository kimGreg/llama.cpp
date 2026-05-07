// streamllm-ext / decoder / anybcq — chunked-matmul dispatch impl.

#include "chunked_matmul_anybcq.h"

// UpstreamLayoutDevice is declared in core/runtime.h. Including it
// here is a controlled core-from-decoder dependency: the struct's
// shape (qw_bytes_per_chunk, K_groups, group_size) is intrinsically
// AnyBCQ-specific and should eventually move to upstream_layout.h,
// but that's a follow-up refactor.
#include "runtime.h"

#include "naver_gemv.h"
#include "dequant_planes.h"
#include "batched_gemm.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace streamllm_ext { namespace anybcq {

// Build the per-plane (qw, alpha) pointer arrays from the resident
// chunks. Returns ``precision`` (number of planes the kernel will
// consume) and whether the chunks form a contiguous prefix
// [PLANE_0..PLANE_{precision-1}] (= the fast path, where the kernel
// can read the per-tensor device-side pointer arrays directly).
//
// Returns 0 on missing chunk data; the caller treats that as "skip".
static int build_plane_ptrs(
    const UpstreamLayoutDevice & L,
    const std::vector<int> & plane_indices,
    const void * (&qw_ptrs)[kNaverMaxPrecision],
    const void * (&alpha_ptrs)[kNaverMaxPrecision],
    bool & prefix_plan)
{
    int precision = 0;
    prefix_plan = true;
    for (int p : plane_indices) {
        if (p < 0 || p >= kNaverMaxPrecision) continue;
        const void * chunk_ptr = L.chunk_ptrs[p];
        if (chunk_ptr == nullptr) return 0;
        if (p != precision) prefix_plan = false;
        qw_ptrs[precision]    = chunk_ptr;
        alpha_ptrs[precision] = static_cast<const uint8_t *>(chunk_ptr) +
                                L.qw_bytes_per_chunk;
        ++precision;
        if (precision >= kNaverMaxPrecision) break;
    }
    return precision;
}

bool chunk_matmul(
    const UpstreamLayoutDevice & L,
    const void * const * d_qw_dev,
    const void * const * d_alpha_dev,
    const std::vector<int> & plane_indices,
    const void * X_fp16, void * Y_fp16,
    int n_tokens,
    size_t x_stride_bytes, size_t y_stride_bytes,
    NaverKernelScratch * scratch,
    StreamHandle compute_stream)
{
    if (L.q_bias_fp16 == nullptr) return false;

    // Build the per-plane (qw, alpha) pointer arrays. Each data chunk
    // is packed as [signs | alpha]; the kernel wants the two base
    // pointers separately, so we split with the precomputed
    // ``qw_bytes_per_chunk`` offset.
    //
    // Fast path: if ``plane_indices`` is a contiguous prefix
    // [0, 1, ..., P-1] in order (the shape every current scheduler
    // emits), the per-tensor device-side pointer arrays populated by
    // move_chunk are directly the kernel's argument — no per-call
    // H2D memcpy needed. Otherwise we fall back to passing nullptrs
    // for d_qw_dev / d_a_dev so the launcher H2D's the host-built
    // arrays itself.
    const void * qw_ptrs[kNaverMaxPrecision]    = {};
    const void * alpha_ptrs[kNaverMaxPrecision] = {};
    bool prefix_plan = true;
    const int precision = build_plane_ptrs(L, plane_indices,
                                            qw_ptrs, alpha_ptrs,
                                            prefix_plan);
    if (precision == 0) return false;

    const void * const * d_qw_use    = (prefix_plan && d_qw_dev)
                                         ? d_qw_dev    : nullptr;
    const void * const * d_alpha_use = (prefix_plan && d_alpha_dev)
                                         ? d_alpha_dev : nullptr;

    auto stream = (cudaStream_t) compute_stream;
    const uint8_t * x_base = static_cast<const uint8_t *>(X_fp16);
    uint8_t       * y_base = static_cast<uint8_t *>(Y_fp16);
    for (int t = 0; t < n_tokens; ++t) {
        const void * x_t = x_base + (size_t)t * x_stride_bytes;
        void       * y_t = y_base + (size_t)t * y_stride_bytes;
        naver_gemv_launch(
            x_t, y_t,
            qw_ptrs, alpha_ptrs, L.q_bias_fp16,
            L.M, L.K, precision, L.group_size, stream,
            scratch, d_qw_use, d_alpha_use);
    }
    return true;
}

bool chunk_matmul_batched(
    const UpstreamLayoutDevice & L,
    const std::vector<int> & plane_indices,
    const void * X_fp16, void * Y_fp16,
    int n_tokens,
    void * w_scratch_f16,
    StreamHandle compute_stream)
{
    if (L.q_bias_fp16 == nullptr) return false;

    const void * qw_ptrs[kNaverMaxPrecision]    = {};
    const void * alpha_ptrs[kNaverMaxPrecision] = {};
    bool prefix_plan_unused = true;
    const int precision = build_plane_ptrs(L, plane_indices,
                                            qw_ptrs, alpha_ptrs,
                                            prefix_plan_unused);
    (void)prefix_plan_unused;
    if (precision == 0) return false;

    auto stream = (cudaStream_t) compute_stream;

    // Backend selection: default cuBLAS via dequant-planes; fused
    // chunked LUT-GEMM under STREAMLLM_BATCHED_BACKEND=fused. See
    // commit history for the rationale (cuBLAS wins at N=2048 in the
    // current tuning regime).
    const char * backend = std::getenv("STREAMLLM_BATCHED_BACKEND");
    const bool use_fused =
        (backend != nullptr && std::strcmp(backend, "fused") == 0);

    if (use_fused) {
        naver_gemm_launch(
            X_fp16, Y_fp16,
            qw_ptrs, alpha_ptrs, L.q_bias_fp16,
            L.M, L.K, n_tokens, precision, L.group_size, stream);
        return true;
    }

    if (w_scratch_f16 == nullptr) return false;
    launch_dequant_planes_f16(
        qw_ptrs, alpha_ptrs, L.q_bias_fp16,
        w_scratch_f16,
        L.M, L.K, precision, L.group_size, stream);
    return batched_gemm_f16(
        w_scratch_f16, X_fp16, Y_fp16,
        L.M, L.K, n_tokens, stream);
}

}}  // namespace streamllm_ext::anybcq
