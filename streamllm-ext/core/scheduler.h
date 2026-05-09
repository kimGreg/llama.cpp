// streamllm-ext / core — Scheduler ABC.
//
// The framework manages chunked tensors via two encoder/architecture-
// blind ABCs (ChunkedTensor, ChunkedComputation) and one global brain:
// the Scheduler.  Concrete schedulers live in model/ subtrees (e.g.
// qwen3::MoEScheduler).
//
// Narrow surface — six virtuals total:
//
//   ─ Lifecycle ───────────────────────────────────────────────────
//     on_install               wid → host bytes / chunk_io / etc.
//     on_graph_compute_begin   ggml about to walk a cgraph
//     on_graph_compute_end     ggml finished walking a cgraph
//
//   ─ Hot path ────────────────────────────────────────────────────
//     on_marker(MarkerEvent)   custom marker node fired during compute
//                              (semantic boundaries: layer entry,
//                              expert dispatch, KV write, ...)
//     plan_for(...)            answer demand for an upcoming
//                              ChunkedComputation (which chunks should
//                              be resident; usually a no-op since
//                              prefetch already loaded them)
//     make_room_for(pool, n)   pool-pressure eviction callback
//
// Anything above that — model-specific dispatch logic, per-op hooks,
// score-policy tables, MoE expert tables — is currently bridged by
// extra virtuals on this base class plus the per-op extern-C shims in
// runtime_hook.{h,cpp}.  Step 4 of the architecture migration replaces
// those shims with custom-op nodes wrapping a ChunkedComputation, at
// which point the bridging virtuals come off the ABC.  See
// /home/jaeyun/.claude/plans/zesty-questing-hippo.md.
//
// Pick at install via STREAMLLM_SCHEDULER.  Currently only "moe" is
// supported; null defaults to it.

#pragma once

#include "stream_reader.h"
#include "upstream_layout.h"
#include "vram_pool.h"

#include <memory>
#include <string>
#include <vector>

struct ggml_tensor;

namespace streamllm_ext {

class StreamllmRuntime;

// Decoder-neutral per-MoE-tensor expert pointer table.  The fused
// kernel that consumes it is architecture-specific (see
// ``qwen3/moe_fused.h``); the *struct* is encoder/architecture-
// agnostic — three device-pointer arrays indexed by expert id —
// so a forward decl here keeps core's contract free of architecture
// types while letting model-side schedulers expose getters via the
// free-function accessors in qwen3_moe_scheduler.h.
struct MoeExpertTable;

// Storage tier for a chunk.  move_chunk(wid, cid, src, dst) currently
// implements the RAM↔VRAM edge; SSD reads happen inside move_chunk
// when host.chunks[p] is empty (pread → ring slot → transform → pool).
enum class Tier : int {
    SSD  = 0,  // on-disk GGUF tensor_offset region
    RAM  = 1,  // host-side layout buffer built from the .gguf bytes
    VRAM = 2,  // upstream layout resident in the VRAM pool
};

const char * tier_name(Tier t);

// A single chunk move the scheduler wants to issue.
struct MoveOp {
    std::string wid;
    int         cid  = 0;
    Tier        src  = Tier::RAM;
    Tier        dst  = Tier::VRAM;
};

// Scheduler's instructions to the dispatch site for one upcoming
// ChunkedComputation.
struct Plan {
    // cids the kernel must consume.  Must be a subset of the chunks
    // guaranteed VRAM-resident after ``moves`` complete (or already
    // resident at entry).
    std::vector<int>    chunks;
    // Moves to issue before compute.  The runtime fires these
    // synchronously; move_chunk records H2D ready events so compute
    // can overlap.
    std::vector<MoveOp> moves;
};

// Marker event payload.  Inserted into the ggml cgraph by the model's
// graph instrumenter (see step 4 of the architecture plan); fires
// inline as ggml walks the graph.  ``kind`` is the only required
// field — concrete schedulers downcast the variant payload to their
// expected type.
enum class MarkerKind : int {
    GraphBegin    = 0,   // before cgraph walk starts
    GraphEnd      = 1,   // after cgraph walk ends
    LayerBegin    = 2,   // entering a transformer layer
    LayerEnd      = 3,
    AttentionIn   = 4,   // about to compute attention
    KvWrite       = 5,   // K/V chunk just written
    MoeDispatch   = 6,   // MoE expert routing decided (gate/topk done)
    Custom        = 7,   // generic sentinel for ad-hoc instrumentation
};

struct MarkerEvent {
    MarkerKind   kind = MarkerKind::Custom;
    int          layer_index = -1;     // valid for Layer*, AttentionIn, MoeDispatch
    StreamHandle compute_stream = nullptr;
    void *       payload = nullptr;    // kind-specific opaque blob
};

class Scheduler {
public:
    virtual ~Scheduler() = default;

    // ─── Lifecycle ─────────────────────────────────────────────────

    // One-time setup called by the runtime after install.  ``rt`` is
    // the owning runtime — the scheduler calls rt.move_chunk() through
    // it to upload chunks.
    virtual void on_install(
        StreamllmRuntime &    rt,
        const StreamReader &  reader,
        const std::string &   gguf_path) = 0;

    // ggml is about to walk a cgraph.  Default no-op.  Step-4 graph
    // instrumenter fires this from the GraphBegin marker; concrete
    // schedulers can use it to prefetch hot tensors / reset per-graph
    // state.
    virtual void on_graph_compute_begin(StreamHandle /*compute_stream*/) {}

    // ggml finished walking a cgraph.  Default no-op.  Symmetric with
    // on_graph_compute_begin.
    virtual void on_graph_compute_end(StreamHandle /*compute_stream*/) {}

    // ─── Hot path ─────────────────────────────────────────────────

    // Marker custom-op node fired during graph walk.  Default no-op.
    // Concrete schedulers dispatch on ``ev.kind`` and use the typed
    // payload.  This is the post-step-4 replacement for the per-op
    // extern-C shims (handle_mul_mat / on_topk_moe_observed / ...).
    virtual void on_marker(const MarkerEvent & /*ev*/) {}

    // Plan a single upcoming ChunkedComputation.  ``tensor_name`` is
    // the canonical wid the runtime passes through; concrete
    // schedulers may inspect ``compute_stream`` to record cross-stream
    // events.  Returns nullptr if the tensor is unmanaged → caller
    // falls through to stock cuBLAS.
    virtual const Plan * plan_for(
        const std::string & tensor_name,
        StreamHandle        compute_stream) = 0;

    // Pool-pressure callback.  Invoked by the runtime when
    // ``pool.load()`` returns a null handle (arena fragmented or
    // full).  The scheduler should evict one or more resident chunks
    // via ``pool.evict(...)`` — picked by whatever residency policy
    // the scheduler maintains — and return true if any progress was
    // made.  Returning false means "no eviction candidates left"; the
    // caller will then surface an out-of-memory error.
    virtual bool make_room_for(VramChunkPool & /*pool*/,
                                size_t /*nbytes_needed*/) {
        return false;
    }

    // Human-readable identifier for log lines.
    virtual const char * name() const = 0;

    // ─── Step-4-doomed bridging virtuals ───────────────────────────
    //
    // The methods below are still vtable-dispatched because the
    // current per-op hook glue (core/runtime_hook.cpp) calls them
    // from extern-C shims.  Step 4 replaces those shims with custom-
    // op nodes wrapping a ChunkedComputation; once that lands the
    // bridging virtuals come off this base class.  See the
    // architecture plan.

    // Score-policy snapshot accessors (live precision dial).  A score
    // policy maps each routed expert's max gate score to a precision
    // tier via a descending-threshold ladder.  Default = "off".
    virtual bool                   is_score_policy() const { return false; }
    virtual std::vector<float>     score_thresholds_snapshot() const { return {}; }
    virtual std::vector<int>       score_chunks_snapshot()    const { return {}; }
    virtual bool set_score_table(const std::vector<float> & /*thresholds*/,
                                  const std::vector<int>   & /*chunks*/) {
        return false;
    }

    // Per-op dispatch hooks — called from extern-C shims registered
    // with ggml-cuda.
    virtual bool handle_mul_mat(StreamHandle /*stream*/,
                                 const struct ggml_tensor * /*src0*/,
                                 const struct ggml_tensor * /*src1*/,
                                 struct ggml_tensor *       /*dst*/) {
        return false;
    }
    virtual bool handle_mul_mat_id(StreamHandle /*stream*/,
                                    const struct ggml_tensor * /*src0*/,
                                    const struct ggml_tensor * /*src1*/,
                                    const struct ggml_tensor * /*ids*/,
                                    struct ggml_tensor *       /*dst*/) {
        return false;
    }
    virtual void on_topk_moe_observed(StreamHandle /*stream*/,
                                       const struct ggml_tensor * /*logits*/,
                                       struct ggml_tensor *       /*weights*/,
                                       struct ggml_tensor *       /*ids*/) {}
    virtual bool claims_tensor(const struct ggml_tensor * /*w*/) {
        return false;
    }
};

// Factory.  ``which`` is currently only "moe" (or nullptr → "moe").
// Anything else logs and returns null so a misspelled flag fails
// loudly instead of silently falling back.
std::unique_ptr<Scheduler> make_scheduler(const char * which);

} // namespace streamllm_ext
