// streamllm-ext / qwen3 — model-specific scheduler entry points.
//
// The base class ``Scheduler`` (in core/scheduler.h) is the encoder/
// architecture-blind ABC: install + graph-compute markers + pool-
// pressure callbacks + a plan_for query.  Anything above that — MoE
// expert tables, per-expert planning, the dense per-tensor plan,
// dispatch reservations — is qwen3-specific and lives here.
//
// The Qwen3 MoE scheduler implementation is anonymously-namespaced in
// qwen3_moe_scheduler.cpp.  These free functions expose the typed
// methods to the dispatch glue (qwen3_moe_dispatch.cpp) without
// publishing the full class definition.  Each function takes a
// reference to the active ``Scheduler`` and downcasts internally —
// safe because make_scheduler returns exactly one concrete type.

#pragma once

#include "scheduler.h"

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
// build_plan_for_anyprec(synthetic, n_chunks) and shortcut wids
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
// entries in qwen3_runtime_glue.cpp call ``Scheduler::claims_node``
// / ``Scheduler::dispatch_node`` / ``Scheduler::observe_topk_moe`` /
// ``Scheduler::claims_tensor`` directly.

// ─── Live precision dial accessors ─────────────────────────────────
// Score-table snapshot + replace, used by qwen3/qwen3_runtime_glue.cpp's
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

}}  // namespace streamllm_ext::qwen3
