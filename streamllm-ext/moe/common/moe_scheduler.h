// streamllm-ext / qwen3 — model-specific scheduler entry points.
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

namespace streamllm_ext { namespace qwen3 {

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

// ─── Live precision dial accessors ─────────────────────────────────
// Score-table snapshot + replace, used by qwen3/runtime_glue.cpp's
// streamllm_set_score_table / streamllm_get_score_table extern-C
// entry points. Concrete impl on MoEScheduler.
//
// Wire format: a single ascending threshold vector of length
// scheduler_score_n_tiers() (= max n_chunks across managed tensors).
// thresholds[k] is the lower-edge gate score for the band that
// loads (k+1) chunks.  Default = all zeros (full precision).
std::vector<float>  scheduler_score_thresholds_snapshot(const Scheduler & sched);
int                 scheduler_score_n_tiers           (const Scheduler & sched);
bool                scheduler_set_score_table(
    Scheduler &                sched,
    const std::vector<float> & thresholds);

// Monotonic counter bumped each time set_score_table() succeeds.  The
// CUDA-graph cache reads this through ggml_cuda_set_streamllm_score_
// version_hook to force re-capture whenever the dial swaps.
uint64_t            scheduler_score_table_version    (const Scheduler & sched);

// Capture-mode accessors. ``score_thresholds_device`` returns the
// device-side mirror of the score-threshold table populated under
// STREAMLLM_ALLOW_CAPTURE=1; nullptr otherwise.  ``allow_capture``
// returns true iff the operator opted into CUDA-graph capture for
// this install.
const float *       scheduler_score_thresholds_device(const Scheduler & sched);
bool                scheduler_allow_capture          (const Scheduler & sched);

// ─── Static per-(layer, expert) chunk-count override ───────────────
// Optional override of the score-threshold-derived chunk count, used
// by the offline budget-solver layout (experiments/4_layout_solver/).
// When a table is loaded, MoEMatMulComp's Pass-2 chunk-count compute
// substitutes ``static_layout_for(layer, expert)`` for any expert
// whose entry is non-zero; falls back to the existing
// chunks_for_gate(g) path otherwise.
//
// Wire format: a flat uint8 array of length n_layers × n_experts in
// row-major (layer-major) order.  Entry [L*E + e] ∈ {1, …,
// max_n_chunks_} means "expert e in layer L always uses that many
// chunks"; 0 means "no override (use threshold path)".
//
// Sets bump score_table_version_ to invalidate captured CUDA graphs,
// reusing the existing dial-swap invalidation mechanism.
bool   scheduler_set_static_layout(
    Scheduler &                  sched,
    const std::vector<uint8_t> & flat_table,
    int                          n_layers,
    int                          n_experts);
void   scheduler_clear_static_layout(Scheduler & sched);
uint8_t scheduler_static_layout_for(const Scheduler & sched, int layer, int expert);
bool   scheduler_has_static_layout(const Scheduler & sched);

// ─── Dynamic per-dispatch K decision (rung 2) ──────────────────────
// Replaces the offline static layout with a per-(layer, expert)
// decision made each dispatch, using BOTH the live gate score and
// the per-(L, e, K) expert-output residual profile baked into the
// GGUF as ``streamllm.expert_residuals.{R, K_min, K_max}``.
//
// Decision rule (see experiments/3_layout_sweep/PROBLEM.md rung 2):
//   K_chosen[L, e]  =  largest K ∈ [K_min, K_max]
//                       s.t.   g² · (R[L, e, K-1] - R[L, e, K])  >  τ
//                       else   K_min
// where g is the per-expert max gate score over the current batch
// and τ is the global error-gain tolerance (STREAMLLM_DYNAMIC_TAU).
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
// Returns the K chosen for (layer, expert) given the live gate score
// and the active tolerance ``tau``.  Returns 0 when residuals aren't
// loaded so the caller knows to fall back to the threshold path.
int    scheduler_dynamic_K_for(const Scheduler & sched,
                                 int layer, int expert,
                                 float gate_score, float tau);
// Active τ tolerance (read from STREAMLLM_DYNAMIC_TAU at construct
// time, mutable later via /streamllm/dynamic_tau HTTP).
float  scheduler_dynamic_tau(const Scheduler & sched);
void   scheduler_set_dynamic_tau(Scheduler & sched, float tau);

// K̄-budget per-dispatch allocator.  When kbar > 0, the runtime picks
//   B_local = round(|experts| × kbar)
// total chunks per dispatch and distributes them across the active
// experts by greedy g²·ΔR.  Returns false when residuals aren't
// loaded or kbar is 0 (caller falls back to the τ / threshold path).
float  scheduler_dynamic_kbar(const Scheduler & sched);
void   scheduler_set_dynamic_kbar(Scheduler & sched, float kbar);
bool   scheduler_allocate_dispatch_budget(
    const Scheduler &          sched,
    int                        layer,
    const std::vector<int>   & experts,
    const std::vector<float> & gates,
    std::vector<int>         & K_out);
// Device-side mirror of dynamic_R_, K_min, K_max — populated by the
// scheduler when STREAMLLM_ALLOW_CAPTURE=1 AND residuals are loaded.
// Read by the capture-mode K̄-knapsack plan kernel
// (launch_plan_per_expert_kbar).  Returns nullptr / 0 / 0 outside
// capture mode.
const float * scheduler_dynamic_R_device(const Scheduler & sched);
int           scheduler_dynamic_K_min   (const Scheduler & sched);
int           scheduler_dynamic_K_max   (const Scheduler & sched);

}}  // namespace streamllm_ext::qwen3
