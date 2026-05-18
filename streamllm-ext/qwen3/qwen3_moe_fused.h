// streamllm-ext / qwen3 — fused MoE LUT-GEMV (Qwen3-MoE architecture).
//
// Architecture-specific kernel-fusion: instead of looping per-(token,
// expert) GEMV through the decoder's per-token entry point, fold every
// (t, u) pair of one MoE op into a single kernel launch. Each block
// decodes its (t, u) from blockIdx.z, looks up
//   eid  = ids[t * n_used + u]
//   qw_planes[eid], alpha_planes[eid], q_bias[eid]
// and accumulates AnyBCQ(eid, X[t]) into Y[t][u].
//
// Layer placement: this fusion lives at the **model** layer because
// MoE batching across experts is an architecture decision, not a
// decoder primitive. The kernel still uses AnyBCQ math (bitplane × α
// + β) and reads the same per-expert pointer tables the decoder
// populates via ``update_per_plane_after_load*``, but the choice of
// "fuse n_used GEMVs into one launch" belongs to whatever model is
// orchestrating MoE — Qwen3 in this codebase.

#pragma once

#include "vram_pool.h"   // StreamHandle

#include <cstddef>
#include <cstdint>

namespace streamllm_ext {

// Per-canonical-MoE-tensor expert table. Decoder-neutral shape (three
// pointer arrays indexed by expert id) so it lives at the
// ``streamllm_ext::`` namespace level — both core/scheduler.h's
// virtual hook and the qwen3 dispatch/scheduler reference it without
// pulling in any architecture-specific code.
struct MoeExpertTable {
    void *** d_qw_planes_per_expert    = nullptr;
    void *** d_alpha_planes_per_expert = nullptr;
    void **  d_q_bias_per_expert       = nullptr;
    int      n_experts                 = 0;

    // Any-prec only: per-expert addresses of the device-side
    // ``d_qbias_slot`` (each is a ``void **`` on device whose [0]
    // entry holds the highest-active chunk's β buffer pointer).
    // Refreshed into ``d_q_bias_per_expert`` by
    // ``refresh_q_bias_for_anyprec_launch`` before each MoE kernel
    // call. Null for fully-ss_anybcq canonicals.
    void *** d_qbias_slot_per_expert   = nullptr;
    bool     needs_qbias_refresh       = false;
};

namespace qwen3 {

// Allocate + populate the three per-expert pointer tables.
void alloc_moe_expert_table(
    MoeExpertTable & out,
    const void * const * host_qw,
    const void * const * host_alpha,
    const void * const * host_q_bias,
    int                  n_experts);

// Free a table built by alloc_moe_expert_table.
void free_moe_expert_table(MoeExpertTable & t);

// For each expert e, scatter
//   table.d_q_bias_per_expert[e] = *table.d_qbias_slot_per_expert[e]
// in one device pass. Required before any MoE kernel call when
// ``table.needs_qbias_refresh`` is true (any-prec wids).
void refresh_q_bias_for_anyprec_launch(
    const MoeExpertTable & table,
    StreamHandle stream);

// Re-point ``d_alpha_planes_per_expert[eid][0..P-1]`` to the α section
// of the *top resident chunk* for expert ``eid``, where
// ``P = prec_per_eid_d[eid]``.
//
// Any-prec α is precision-dependent: chunk c packs the optimal
// precision-(base_p+c-1) α values for ALL planes [0, base_p+c-1).  At
// load time those α pointers got written into d_alpha by
// ``update_anyprec_after_load``, but the values reflect the precision
// AT LOAD TIME, which may not match what THIS dispatch needs.  Worse,
// if chunk c is evicted but d_alpha still points into the freed slot,
// the kernel reads slot-reuse garbage.
//
// This kernel runs once per MoE dispatch (right before
// ``naver_gemv_moe_launch``) and recomputes d_alpha entries from the
// d_qw pointer of the top resident chunk: alpha_base = d_qw[top_plane]
// + n_planes_signs(top) × qw_bytes_per_chunk, then
// d_alpha[p] = alpha_base + p × alpha_bytes_per_chunk for p in [0, P).
//
//   table             — per-canonical expert table (any-prec only).
//   prec_per_eid_d    — per-expert plane count for this dispatch (PLANES).
//   base_p            — encoder's base_precision (planes in chunk 0).
//   qw_bytes_per_chunk, alpha_bytes_per_chunk
//                     — per-plane strides from UpstreamLayoutDevice.
void refresh_alpha_for_anyprec_launch(
    const MoeExpertTable & table,
    const int *  prec_per_eid_d,
    int          base_p,
    size_t       qw_bytes_per_chunk,
    size_t       alpha_bytes_per_chunk,
    StreamHandle stream);

// Fused MoE LUT-GEMV launcher.
//   X_fp16        : shared_x=1 → [n_tokens, K], else [n_tokens, n_used, K]
//   Y_dst_f32     : [n_tokens, n_used, M], must be pre-zeroed
//   ids_d         : [n_tokens, n_used] int32 device
//   table         : per-canonical expert table from MoEScheduler
//   uniform_precision : plane count used when prec_per_eid_d is nullptr.
//   prec_per_eid_d : optional [n_experts] device array of per-expert
//                    plane counts; kernel reads
//                    ``P = prec_per_eid_d[ids[tu]]``.  Required planes
//                    [0, P) MUST be non-null at launch (SSOT §6.9 M1);
//                    null required pointer → __trap().
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
    const int *            prec_per_eid_d,
    int                    group_size,
    int                    shared_x,
    StreamHandle           stream);
// Mode A milestone 1, S6 — element-wise SwiGLU * up gate.
//
//   slot_gate [N] f32 — gate matmul output (per-slot, flattened)
//   slot_up   [N] f32 — up   matmul output (per-slot, flattened)
//   slot_out  [N] f32 — output: silu(slot_gate) * slot_up
//
// Where silu(x) = x / (1 + exp(-x)). Element-wise, no shape dependence;
// flatten the [n_tokens, n_used, n_ff] tensors to N = n_tokens*n_used*n_ff.
// slot_out may alias slot_gate or slot_up. All device-resident, on
// ``stream``. Used by Qwen3MoEAnyBcqExecutor::forward_moe_layer between
// the gate/up chunked matmuls and the down chunked matmul.
void launch_swiglu_mul(
    const float * slot_gate,
    const float * slot_up,
    float *       slot_out,
    std::size_t   N,
    StreamHandle  stream);

// Mode A milestone 1, S4 — weighted reduce over the per-top-k slot axis.
//
//   slot_out  [n_tokens, n_used, M]  f32  — kernel output from the
//                                           fused MoE op above; per-slot
//                                           (per-(t,u)) M-dim activations
//   weights   [n_tokens, n_used]     f32  — renormalised routing weights
//   layer_out [n_tokens, M]          f32  — output: layer_out[t, m] =
//                                           sum_u weights[t,u] * slot_out[t,u,m]
//
// All buffers device-resident, all operations on ``stream``. Used by
// Qwen3MoEAnyBcqExecutor::forward_moe_layer (S6) to collapse the
// per-slot intermediate into the final block output the post-MoE
// dense subgraph consumes. Standalone — no production path calls
// this until S6 lands the call site.
void launch_weighted_reduce_slots(
    const float *   slot_out,
    const float *   weights,
    float *         layer_out,
    int             n_tokens,
    int             n_used,
    int             M,
    StreamHandle    stream);

// Capture-mode planning kernel — populates ``planes_per_eid_d`` from
// (ids, weights, probs, dial) entirely on the device. Mirrors the
// host-side ``MoEMatMulComp::plan`` logic in qwen3_moe_matmul_comp.cpp
// (per-expert max gate score → chunks_for_gate → planes_for_chunks),
// so the eager and capture paths produce the same prec_per_eid_d for
// the same (routing, dial). Used by Qwen3MoEAnyBcqExecutor's
// capture-mode fast path to eliminate the D2H + cudaStreamSynchronize
// that breaks CUDA-graph capture.
//
//   ids_d         [n_tokens, n_used]      int32  — top-K expert ids
//                                                  per token. Required.
//   weights_d     [n_tokens, n_used]      f32    — renormalised weights;
//                                                  optional (may be null).
//   probs_d       [n_tokens, n_expert]    f32    — raw routing scores;
//                                                  optional (used only
//                                                  if weights_d is null).
//   thresholds_d  [n_tiers]               f32    — score-table device
//                                                  mirror from MoESchd.
//   planes_per_eid_d [n_expert]           int32  — output: number of
//                                                  planes the kernel
//                                                  should sweep for each
//                                                  expert.
//
// Unrouted experts (no (t,u) with ids[t,u] == eid) receive the maximum
// plane count — matching the host pre-fill in MoEMatMulComp::on_install.
void launch_plan_per_expert_planes(
    const int32_t * ids_d,
    const float *   weights_d,
    const float *   probs_d,
    const float *   thresholds_d,
    int             n_tokens,
    int             n_used,
    int             n_expert,
    int             n_tiers,
    int             n_chunks_max,
    int             base_p,
    bool            any_precision,
    int *           planes_per_eid_d,
    StreamHandle    stream);

}  // namespace qwen3
}  // namespace streamllm_ext
