// streamllm-ext / decoder / shortcut_anybcq — chunked-matmul impl.
//
// Shortcut layout: each chunk packs [signs | α-scalar]. β is in a
// separate kCidQBias chunk pinned at install. Decode = naver_gemv per
// token; prefill = dequant + cuBLAS GEMM (or fused chunked LUT-GEMM).
// Kernels themselves are shared with the any-prec decoder (under
// ``../anybcq/anybcq_gemv.h`` + ``anybcq_gemm.h``); only the per-encoder
// pointer staging differs.

#include "chunked_matmul.h"

// UpstreamLayoutDevice is declared in core/runtime.h.
#include "runtime.h"

#include "anybcq_gemv.h"
#include "anybcq_gemm.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace streamllm_ext { namespace shortcut_anybcq {

// Reuse the GEMV kernel scratch typedef + entry points from anybcq's
// shared kernel header.
using ::streamllm_ext::naver_gemv_launch;
using ::streamllm_ext::naver_gemm_launch;
using ::streamllm_ext::launch_dequant_planes_f16;
using ::streamllm_ext::batched_gemm_f16;

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
    // Shortcut layout only — any-prec is rejected upstream by
    // StreamllmRuntime::chunk_matmul.
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


// =====================================================================
// Higher-level wrappers — what the model layer (qwen3 dispatch) calls.
// Used to live as ``StreamllmRuntime::chunk_matmul[_batched]`` on core,
// but those were thin pass-throughs that pulled decoder symbols into
// core's translation unit; moved here so core stays decoder-blind.
// =====================================================================

namespace {

// Translate a chunk-cid list to plane indices (shortcut layout: one
// plane per data chunk). Reject any-prec layouts loudly — that path
// goes through the qwen3 MoE-fused kernel, not this dispatcher.
inline std::vector<int> cids_to_planes_or_throw(
    const UpstreamLayoutHost & host,
    const std::string & wid,
    const std::vector<int> & chunks,
    const char * caller)
{
    if (host.any_precision) {
        throw std::runtime_error(
            std::string(caller) +
            ": any-prec layout not supported on this dispatch path; "
            "use the MoE mul_mat_id hook (wid=" + wid + ")");
    }
    std::vector<int> planes;
    planes.reserve(chunks.size());
    for (int cid : chunks) {
        if (cid_is_chunk(cid)) planes.push_back(cid_chunk_index(cid));
    }
    return planes;
}

}  // anonymous

bool chunk_matmul_for_wid(
    StreamllmRuntime &      rt,
    const std::string &     wid,
    const std::vector<int> & chunks,
    const void * X_fp16, void * Y_fp16,
    int n_tokens,
    size_t x_stride_bytes, size_t y_stride_bytes,
    StreamHandle compute_stream)
{
    const UpstreamLayoutDevice * dev = rt.layout(wid);
    const UpstreamLayoutHost   * host = rt.host_layout(wid);
    if (dev == nullptr || host == nullptr) return false;

    for (int cid : chunks) {
        rt.pool().wait_on_stream(wid, cid, compute_stream);
    }

    auto planes = cids_to_planes_or_throw(
        *host, wid, chunks, "shortcut_anybcq::chunk_matmul_for_wid");

    return chunk_matmul(
        *dev,
        (const void * const *) rt.anybcq_d_qw_ptrs(wid),
        (const void * const *) rt.anybcq_d_alpha_ptrs(wid),
        planes, X_fp16, Y_fp16, n_tokens,
        x_stride_bytes, y_stride_bytes,
        rt.gemv_scratch(), compute_stream);
}

bool chunk_matmul_batched_for_wid(
    StreamllmRuntime &      rt,
    const std::string &     wid,
    const std::vector<int> & chunks,
    const void * X_fp16, void * Y_fp16,
    int n_tokens,
    void * w_scratch_f16,
    StreamHandle compute_stream)
{
    const UpstreamLayoutDevice * dev = rt.layout(wid);
    const UpstreamLayoutHost   * host = rt.host_layout(wid);
    if (dev == nullptr || host == nullptr) return false;

    for (int cid : chunks) {
        rt.pool().wait_on_stream(wid, cid, compute_stream);
    }

    auto planes = cids_to_planes_or_throw(
        *host, wid, chunks,
        "shortcut_anybcq::chunk_matmul_batched_for_wid");

    return chunk_matmul_batched(
        *dev, planes, X_fp16, Y_fp16, n_tokens,
        w_scratch_f16, compute_stream);
}

}}  // namespace streamllm_ext::shortcut_anybcq
