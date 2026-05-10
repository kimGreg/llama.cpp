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

struct ggml_tensor;

namespace streamllm_ext { namespace qwen3 {

// Dense managed mul_mat plan (per-tensor).  Returns nullptr if the
// tensor is unmanaged.  Used by the dense path inside the qwen3
// dispatch glue.
const Plan * scheduler_plan_dense(
    Scheduler &         sched,
    const std::string & wid,
    StreamHandle        compute_stream);

// MoE per-expert plan, gate-threshold form.  ``rank`` is the expert's
// position in the current token's top-k routing (0 = highest gate).
const Plan * scheduler_plan_for_expert(
    Scheduler &         sched,
    const std::string & canonical_wid,
    int                 expert_id,
    float               gate_score,
    int                 rank,
    StreamHandle        compute_stream);

// MoE per-expert plan, direct-precision form.  Bypasses the gate-
// threshold ladder; the dispatch already decided ``desired_precision``
// (e.g. by looking up the expert's max gate score in the score table).
const Plan * scheduler_plan_for_expert_with_precision(
    Scheduler &         sched,
    const std::string & canonical_wid,
    int                 expert_id,
    int                 desired_precision,
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

// Graph-instrumenter pre-dispatch visit. Called from the per-op
// dispatch glue (qwen3_moe_dispatch.cpp) before any compute for a
// managed node. The instrumenter uses the prewalk-built node→layer
// map to detect layer transitions and fires
// Scheduler::on_marker(LayerBegin/LayerEnd). No-op if the
// instrumenter hasn't been seeded (graph_compute_begin not called
// yet, or non-MoE scheduler active).
void scheduler_on_managed_node_visit(
    Scheduler &                sched,
    const struct ggml_tensor * dst,
    StreamHandle               compute_stream);

}}  // namespace streamllm_ext::qwen3
