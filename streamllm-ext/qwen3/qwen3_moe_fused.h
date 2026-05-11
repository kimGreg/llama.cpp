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
    // call. Null for fully-shortcut canonicals.
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

}  // namespace qwen3
}  // namespace streamllm_ext
