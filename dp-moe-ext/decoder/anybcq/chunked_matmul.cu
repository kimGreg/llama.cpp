// DPMoE / decoder / anybcq — MoE chunked-matmul dispatch impl.
//
// See chunked_matmul.h for the contract.  This TU owns the one
// chunks→planes translation in the codebase.  Plane vocabulary may
// not leak out of this file (constraints 1 + 5).

#include "chunked_matmul.h"

#include "launch_diag.h"
#include "runtime.h"           // DPMoERuntime, UpstreamLayoutDevice
#include "fused_kernels.h"   // refresh_q_bias_for_anyprec_launch,
                                // naver_gemv_moe_launch

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace dp_moe_ext {
namespace anybcq {

namespace {

std::atomic<unsigned long long> g_chunk_matmul_calls{0};
std::atomic<unsigned long long> g_kernel_launch_calls{0};
std::atomic<unsigned long long> g_n_chunks_oob{0};

}  // namespace


unsigned long long counter_chunk_matmul_calls() {
    return g_chunk_matmul_calls.load(std::memory_order_relaxed);
}
unsigned long long counter_kernel_launch_calls() {
    return g_kernel_launch_calls.load(std::memory_order_relaxed);
}
unsigned long long counter_n_chunks_oob() {
    return g_n_chunks_oob.load(std::memory_order_relaxed);
}
void reset_counters() {
    g_chunk_matmul_calls.store(0, std::memory_order_relaxed);
    g_kernel_launch_calls.store(0, std::memory_order_relaxed);
    g_n_chunks_oob.store(0, std::memory_order_relaxed);
}


// chunks → planes for this canonical's encoder, per constraint 5:
//   any-prec:  planes = base_p + n_chunks − 1
//   else:      planes =          n_chunks       (identity)
//
// Asserts (constraint 6): 1 ≤ n_chunks ≤ layout.n_chunks
//                          1 ≤ planes  ≤ kMaxChunksPerTensor
//
// Returns -1 on out-of-range input (caller treats as a hard error in
// debug, clamps + increments g_n_chunks_oob in release).
static inline int planes_for_chunks(
    int n_chunks, int n_chunks_max, int base_p, bool any_precision)
{
    if (n_chunks < 1 || n_chunks > n_chunks_max) return -1;
    int planes = any_precision ? (base_p + n_chunks - 1) : n_chunks;
    if (planes < 1)                      planes = 1;
    if (planes > kMaxChunksPerTensor)    planes = kMaxChunksPerTensor;
    return planes;
}


bool moe_chunk_matmul(
    DPMoERuntime &           rt,
    const std::string &          canonical,
    const UpstreamLayoutDevice & layout,
    const MoeExpertTable &       table,
    int *                    prec_per_eid_d,
    const int *              host_n_chunks_per_eid,
    const int32_t *          ids_d,
    const void *             X_fp16,
    void *                   Y_slot_f32,
    int                      n_tokens,
    int                      n_used,
    int                      n_experts,
    int                      M,
    int                      K,
    int                      group_size,
    int                      shared_x,
    int                      routed_chunk_span_hint,
    int                      routed_max_chunk_hint,
    StreamHandle             stream)
{
    g_chunk_matmul_calls.fetch_add(1, std::memory_order_relaxed);

    // host_n_chunks_per_eid is allowed to be nullptr — that signals
    // "prec_per_eid_d is already populated with PLANE counts by the
    // caller" (capture-mode fast path that runs the device plan
    // kernel before us). Otherwise it's the host chunk-count array
    // we must convert + H2D below.
    if (prec_per_eid_d == nullptr ||
        ids_d == nullptr || X_fp16 == nullptr || Y_slot_f32 == nullptr ||
        n_experts <= 0 || n_tokens <= 0 || n_used <= 0)
    {
        std::fprintf(stderr,
            "DPMoE: anybcq::moe_chunk_matmul(%s): bad inputs "
            "(prec_d=%p host_chunks=%p ids_d=%p X=%p Y=%p "
            "n_exp=%d n_t=%d n_u=%d)\n",
            canonical.c_str(),
            (void *) prec_per_eid_d, (void *) host_n_chunks_per_eid,
            (void *) ids_d, X_fp16, Y_slot_f32,
            n_experts, n_tokens, n_used);
        return false;
    }

    (void) rt;  // reserved for future diagnostics
    const int  n_chunks_max  = layout.n_chunks;
    const int  base_p        = layout.base_precision > 0
                                ? layout.base_precision : 1;
    const bool any_precision = layout.any_precision;
    if (n_chunks_max <= 0) {
        std::fprintf(stderr,
            "DPMoE: anybcq::moe_chunk_matmul(%s): empty layout "
            "(n_chunks=%d)\n", canonical.c_str(), n_chunks_max);
        return false;
    }

    // ── Build the kernel-facing per-expert plane array.
    //    n_chunks_per_eid (model layer's unit) → planes (kernel's unit).
    //    This is the ONLY plane-conversion site in the codebase.
    //
    // Capture-mode short-circuit: if the caller passed nullptr it has
    // already populated prec_per_eid_d with PLANE counts (via the on-
    // device plan kernel — qwen3::launch_plan_per_expert_planes). We
    // skip the host conversion + H2D and fall through to the kernel
    // launches with prec_per_eid_d as-is. uniform_precision is unused
    // by the GEMV kernel when prec_per_eid_d is non-null but the
    // wrapper range-checks it, so use n_chunks_max as a safe default.
    int max_planes_seen = 0;
    if (host_n_chunks_per_eid == nullptr) {
        max_planes_seen = any_precision
            ? (base_p + n_chunks_max - 1)
            : n_chunks_max;
        if (max_planes_seen > kMaxChunksPerTensor) {
            max_planes_seen = kMaxChunksPerTensor;
        }
    } else {
        static thread_local std::vector<int> host_planes_per_eid;
        if ((int) host_planes_per_eid.size() < n_experts) {
            host_planes_per_eid.resize((size_t) n_experts);
        }
        for (int e = 0; e < n_experts; ++e) {
            const int nc = host_n_chunks_per_eid[e];
            int planes = planes_for_chunks(nc, n_chunks_max,
                                            base_p, any_precision);
            if (planes < 0) {
                g_n_chunks_oob.fetch_add(1, std::memory_order_relaxed);
#ifndef NDEBUG
                std::fprintf(stderr,
                    "DPMoE: anybcq::moe_chunk_matmul(%s): "
                    "n_chunks_per_eid[%d]=%d out of [1, %d] — clamping\n",
                    canonical.c_str(), e, nc, n_chunks_max);
#endif
                const int safe_nc =
                    std::clamp(nc, 1, n_chunks_max > 0 ? n_chunks_max : 1);
                planes = planes_for_chunks(safe_nc, n_chunks_max,
                                            base_p, any_precision);
                if (planes < 0) planes = 1;
            }
            host_planes_per_eid[e] = planes;
            if (planes > max_planes_seen) max_planes_seen = planes;
        }
        if (max_planes_seen < 1) max_planes_seen = 1;
        if (max_planes_seen > kMaxChunksPerTensor) {
            max_planes_seen = kMaxChunksPerTensor;
        }

        // H2D the per-expert plane counts into prec_per_eid_d.  The
        // kernel reads ``P = prec_per_eid_d[ids[tu]]`` per (t,u) and
        // iterates planes [0, P).  Required plane pointers MUST be
        // non-null at launch (SSOT §6.9 M1).
        const size_t bytes = (size_t) n_experts * sizeof(int);
        ::dp_moe_ext::launch_diag::note_launch(
            ::dp_moe_ext::launch_diag::Kind::MemcpyAsync);
        cudaError_t err = cudaMemcpyAsync(
            prec_per_eid_d, host_planes_per_eid.data(),
            bytes, cudaMemcpyHostToDevice,
            (cudaStream_t) stream);
        if (err != cudaSuccess) {
            std::fprintf(stderr,
                "DPMoE: anybcq::moe_chunk_matmul(%s): H2D "
                "prec_per_eid_d failed: %s\n",
                canonical.c_str(), cudaGetErrorString(err));
            return false;
        }
    }

    // ── Build the kernel-facing pointer table FRESH at launch time.
    //
    // Any-prec α and β are precision-dependent: chunk c packs the
    // optimal precision-(base_p+c) α + β for THIS chunk's precision
    // tier. The ``update_anyprec_after_load`` callback wrote per-plane
    // pointers reflecting the LAST chunk to load — which under
    // varying-precision dial + cap-pressure may be the wrong chunk
    // (eviction race or just stale from a previous dispatch's higher
    // precision).
    //
    // We rebuild d_alpha and d_q_bias for every expert from the top
    // resident chunk's qw section, indexed by THIS dispatch's
    // per-expert precision. After this kernel runs the pointer table
    // is correct as a pure function of (residency, per-expert
    // precision) — no historical state matters.
    if (any_precision) {
        qwen3::refresh_alpha_for_anyprec_launch(
            table, prec_per_eid_d,
            base_p,
            layout.qw_bytes_per_chunk,
            layout.alpha_bytes_per_chunk,
            stream);
    } else {
        // SsAnybcq family: β is per-tensor (single q_bias chunk pinned
        // at install) so the slot-scatter still applies.
        qwen3::refresh_q_bias_for_anyprec_launch(table, stream);
    }

    // ── Fused MoE LUT-GEMV.  uniform_precision is the fallback used
    //    when prec_per_eid_d is null; here we always pass a valid
    //    per-expert table, so uniform_precision just needs to lie in
    //    the kernel's [1, kMaxChunksPerTensor] range check.
    int routed_precision_span_hint = -1;
    if (routed_chunk_span_hint >= 0) {
        // For both direct and any-precision encoders, adding one chunk
        // adds exactly one kernel plane.  The base precision offset only
        // affects the absolute max below.
        routed_precision_span_hint = routed_chunk_span_hint;
    }
    int routed_max_precision_hint = -1;
    if (routed_max_chunk_hint > 0 && routed_max_chunk_hint <= n_chunks_max) {
        routed_max_precision_hint =
            planes_for_chunks(routed_max_chunk_hint, n_chunks_max,
                              base_p, any_precision);
    }

    int launch_max_planes = max_planes_seen;
    if (routed_max_precision_hint > 0 &&
        routed_max_precision_hint <= kMaxChunksPerTensor) {
        launch_max_planes = routed_max_precision_hint;
    }
    if (launch_max_planes < 1) launch_max_planes = 1;
    if (launch_max_planes > max_planes_seen) launch_max_planes = max_planes_seen;

    g_kernel_launch_calls.fetch_add(1, std::memory_order_relaxed);
    ::dp_moe_ext::launch_diag::note_launch(
        ::dp_moe_ext::launch_diag::Kind::MoeMatmul);
    qwen3::naver_gemv_moe_launch(
        X_fp16, Y_slot_f32, ids_d,
        table,
        M, K, n_tokens, n_used,
        /*uniform_precision=*/launch_max_planes,
        /*prec_per_eid_d=*/prec_per_eid_d,
        group_size,
        shared_x,
        routed_precision_span_hint,
        stream);

    return true;
}

}  // namespace anybcq
}  // namespace dp_moe_ext
