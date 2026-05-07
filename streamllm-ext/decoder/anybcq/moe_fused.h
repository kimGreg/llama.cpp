// streamllm-ext / decoder / anybcq — fused MoE LUT-GEMV.
//
// One kernel launch processes every (token, used_idx) pair of a
// single MoE op, instead of looping the per-(token, expert) GEMV.
// Each block decodes its (t, u) from blockIdx.z, looks up
//   eid  = ids[t * n_used + u]
//   gate = probs[t * n_used + u]
//   qw_planes[eid], alpha_planes[eid], q_bias[eid]
// and accumulates ``gate * AnyBCQ(eid, X[t])`` into ``Y[t]``.
//
// Compared to the per-token ``nqmv_bias_planes`` kernel:
//   * Same per-block math (same M_TILE × K_TILE structure, same LUT).
//   * Adds a small per-block expert lookup at the start.
//   * Output goes via atomicAdd into a SHARED Y[t] accumulator (across
//     u values for the same t) instead of an exclusive output slot.
//   * One launch covers all (t, u) pairs → 32× fewer launches per MoE
//     op at decode (n_tokens=1, n_used=8), 96× at decode batch=4.
//
// Caller responsibilities:
//   * Y_acc_f32 must be zero-initialised (size = n_tokens × M floats).
//     Cast Y_acc_f32 → fp16 ``Y_fp16`` after the kernel completes.
//   * ids_d, probs_d are device pointers, lifetime ≥ kernel.
//   * The three device-side tables are built (in MoEScheduler) at
//     install time per canonical MoE tensor — they index 0..n_experts-1
//     into pre-allocated per-expert plane-pointer arrays.

#pragma once

#include "vram_pool.h"  // StreamHandle

#include <cstddef>
#include <cstdint>

namespace streamllm_ext {

// Per-canonical-MoE-tensor expert table. Lives in device memory; stays
// stable across hook calls because each entry is just a pointer to the
// owning expert's pre-allocated d_chunk_qw_ptrs / d_chunk_alpha_ptrs /
// dev.q_bias_fp16 (those mutate in place when a chunk lands; the
// table never has to be rebuilt).
//
// The struct is decoder-neutral (just three device pointer arrays
// indexed by expert id), so it lives in ``streamllm_ext::``. The
// kernel that consumes it (naver_gemv_moe_launch below) is AnyBCQ-
// specific and stays in ``streamllm_ext::anybcq::``.
struct MoeExpertTable {
    // [n_experts] each entry is `(const uint32_t * const *)` pointing
    // at expert e's d_chunk_qw_ptrs (device array of length kMaxPlanes).
    void *** d_qw_planes_per_expert    = nullptr;
    void *** d_alpha_planes_per_expert = nullptr;
    // [n_experts] of __half * — each expert's q_bias on device.
    void **  d_q_bias_per_expert       = nullptr;
    int      n_experts                 = 0;
};

namespace anybcq {

// Allocate + populate the three device-side per-expert pointer tables.
// `host_qw[e]`, `host_alpha[e]`, `host_q_bias[e]` are the per-expert
// pointers gathered host-side at install time. Called ONCE per canonical
// MoE tensor.
void alloc_moe_expert_table(
    MoeExpertTable & out,
    const void * const * host_qw,        // [n_experts] (each is `void **` on device)
    const void * const * host_alpha,
    const void * const * host_q_bias,
    int                  n_experts);

// Free a table built by alloc_moe_expert_table. Safe on zero-initialised.
void free_moe_expert_table(MoeExpertTable & t);


// Fused MoE LUT-GEMV launcher. Single kernel launch covering every
// (t, u) pair of one MoE op.
//
//   X_fp16        :  shared_x=1  → [n_tokens, K]            fp16
//                    shared_x=0  → [n_tokens, n_used, K]    fp16
//   Y_dst_f32     :  [n_tokens, n_used, M]   fp32
//                    must be pre-zeroed (k-tile blocks atomicAdd into it)
//                    written at offset (t*n_used + u)*M + m
//   ids_d         :  [n_tokens, n_used]      int32 (device)
//   table         :  per-canonical-wid expert table built in MoEScheduler
//   M, K, group_size  :  shared across experts (AnyBCQ-tensor scoped)
//   uniform_precision :  fallback plane count consumed per (t, u) when
//                        prec_per_tu_d is nullptr.
//   prec_per_tu_d     :  optional device pointer [n_tokens × n_used] of
//                        per-(t, u) plane counts. When non-null, kernel
//                        reads precision from this array and ignores
//                        uniform_precision. Used by the rank_cum policy
//                        to give top-1 max precision and tail experts
//                        deeper compression on the same dispatch.
//   n_tokens, n_used  :  batch shape
//   shared_x          :  1 if X is laid out per-token only, 0 if per-(t, u)
//
// Caller does NOT pass gate weights — ggml's mul_mat_id semantics put
// gate-weighting + sum-over-u in a downstream graph node. The kernel
// just writes raw decoded outputs into the per-(t, u) dst slots.
void naver_gemv_moe_launch(
    const void *           X_fp16,
    void *                 Y_dst_f32,
    const int32_t *        ids_d,
    const MoeExpertTable & table,
    int                    M,
    int                    K,
    int                    n_tokens,
    int                    n_used,
    int                    uniform_precision,
    const int *            prec_per_tu_d,
    int                    group_size,
    int                    shared_x,
    StreamHandle           stream);

}  // namespace anybcq
}  // namespace streamllm_ext
