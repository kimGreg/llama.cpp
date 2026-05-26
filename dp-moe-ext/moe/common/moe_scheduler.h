// DPMoE / qwen3 — model-specific scheduler entry points.
//
// The base class ``Scheduler`` (in core/scheduler.h) is the encoder/
// architecture-blind ABC: install + graph-compute markers + pool-
// pressure callbacks + a plan_for query.  Anything above that — MoE
// expert tables, per-expert planning, the dense per-tensor plan,
// dispatch reservations — is qwen3-specific and lives here.
//
// The Qwen3 MoE scheduler implementation is anonymously-namespaced in
// scheduler.cpp.  These free functions expose the typed
// methods to the dispatch glue (dispatch.cpp) without
// publishing the full class definition.  Each function takes a
// reference to the active ``Scheduler`` and downcasts internally —
// safe because make_scheduler returns exactly one concrete type.

#pragma once

#include "scheduler.h"        // core Scheduler ABC + Plan
#include "fused_kernels.h"    // MoeExpertTable

#include <string>
#include <vector>

struct ggml_tensor;

namespace dp_moe_ext { namespace qwen3 {

// Dense managed mul_mat plan (per-tensor).  Returns nullptr if the
// tensor is unmanaged.  Used by the dense path inside the qwen3
// dispatch glue.
const Plan * scheduler_plan_dense(
    Scheduler &         sched,
    const std::string & wid,
    StreamHandle        compute_stream);

// MoE per-expert plan.  Dispatch has already looked up the per-expert
// max-gate score in the score table and resolved that to a chunk count;
// we just emit a Plan whose chunks list matches.
//
// ``n_chunks_requested`` is in the model layer's unit (CHUNKS).
// Internally the scheduler routes any-prec wids through
// build_plan_for_anyprec(synthetic, n_chunks) and ss_anybcq wids
// through build_plan_for(synthetic, n_chunks) — both interpret the
// argument as a chunk count.  Plane semantics live entirely inside
// decoder/anybcq (SSOT §6.1.5 layering).
const Plan * scheduler_plan_for_expert_with_chunks(
    Scheduler &         sched,
    const std::string & canonical_wid,
    int                 expert_id,
    int                 n_chunks_requested,
    StreamHandle        compute_stream);

// Per-canonical-tensor expert table for the fused MoE LUT-GEMV kernel.
// Returns nullptr if the scheduler has no MoE table for ``canonical_wid``.
const MoeExpertTable * scheduler_moe_expert_table(
    Scheduler &         sched,
    const std::string & canonical_wid);

// Dispatch-window reservations.  Hold a chunk against eviction while
// the in-flight kernel is still reading it; release once the compute
// event has been recorded.  No-op if the scheduler doesn't track
// reservations.
void scheduler_reserve_for_dispatch(
    Scheduler &         sched,
    const std::string & wid,
    int                 cid);
void scheduler_release_from_dispatch(
    Scheduler &         sched,
    const std::string & wid,
    int                 cid);
void scheduler_touch_resident(
    Scheduler &         sched,
    const std::string & wid,
    int                 cid,
    int                 plane);
void scheduler_touch_host_cache(
    Scheduler &         sched,
    const std::string & wid,
    int                 cid);

// Logical cache clock for MoE residency. Runtime mode calls this once
// per MoE layer before planning gate/up/down so all chunks touched by
// the same layer share one age timestamp.
void scheduler_note_moe_layer(
    Scheduler & sched,
    bool        decode_phase,
    int         layer_idx);

// Post-compute hook: called after a managed mul_mat has launched and
// the pool's compute event has been recorded.  Schedulers that want
// "flush after use" semantics evict tail chunks here.
void scheduler_after_compute(
    Scheduler &         sched,
    const std::string & wid,
    StreamHandle        compute_stream);

// ``scheduler_on_managed_node_visit`` and the per-canonical graph
// instrumenter (qwen3_graph_instrumenter.{h,cpp}) were retired in the
// post-M1 cleanup pass. The sentinel name carries the per-layer
// index directly; no per-node visit-marker is needed.

// (P2★) The previous per-op extern-C shim forwarders
// scheduler_handle_mul_mat / _id, scheduler_on_topk_moe_observed,
// scheduler_claims_tensor have been removed.  Each was a downcast
// wrapper for what is now a Scheduler virtual; the per-op extern-C
// entries in runtime_glue.cpp call ``Scheduler::claims_node``
// / ``Scheduler::dispatch_node`` / ``Scheduler::observe_topk_moe`` /
// ``Scheduler::claims_tensor`` directly.

// ─── Live KBar dial accessors ──────────────────────────────────────
// KBar is the single runtime quality dial. ``allocator_mode`` is
// ``profile`` for the residual-profile DP allocator and ``uniform`` for
// round(KBar) chunks on every active expert.
enum class KBarAllocatorMode {
    Profile,
    Uniform,
};

float               scheduler_kbar                  (const Scheduler & sched);
KBarAllocatorMode   scheduler_kbar_allocator_mode    (const Scheduler & sched);
bool                scheduler_set_kbar              (
    Scheduler & sched, float kbar, KBarAllocatorMode mode);
bool                scheduler_set_prefill_decay_end (
    Scheduler & sched, uint64_t n_tokens, bool reset_state);
uint64_t            scheduler_dial_version          (const Scheduler & sched);
bool                scheduler_allow_capture         (const Scheduler & sched);

// ─── Profile-backed per-dispatch K decision ────────────────────────
// Replaces the offline static layout with a per-(layer, expert)
// decision made each dispatch, using BOTH the live gate score and
// the per-(L, e, K) expert-output residual profile baked into the
// GGUF as ``dp_moe.expert_residuals.{R, K_min, K_max}``.
//
// Decision rule (see experiments/3_dynamic_precision/PROBLEM.md):
//   K_chosen[L, e]  =  largest K ∈ [K_min, K_max]
//                       s.t.   g² · (R[L, e, K-1] - R[L, e, K])  >  τ
//                       else   K_min
// where g is the per-expert max gate score over the current batch
// and KBar is the global chunk-budget dial.
//
// R is stored in layer-major order, length = n_layers · n_experts · n_K
// where n_K = K_max − K_min + 1.
bool   scheduler_set_dynamic_residuals(
    Scheduler &              sched,
    const std::vector<float> & R_flat,
    int                      n_layers,
    int                      n_experts,
    int                      K_min,
    int                      K_max);
void   scheduler_clear_dynamic_residuals(Scheduler & sched);
bool   scheduler_has_dynamic_residuals(const Scheduler & sched);
bool   scheduler_allocate_dispatch_budget(
    const Scheduler &          sched,
    int                        layer,
    const std::vector<int>   & experts,
    const std::vector<float> & gates,
    std::vector<int>         & K_out);
// Device-side mirror of dynamic_R_, K_min, K_max — populated when
// residuals are loaded. Read by benchmark capture and eager runtime
// GPU KBar planners.
const float * scheduler_dynamic_R_device(const Scheduler & sched);
int           scheduler_dynamic_K_min   (const Scheduler & sched);
int           scheduler_dynamic_K_max   (const Scheduler & sched);

}}  // namespace dp_moe_ext::qwen3
