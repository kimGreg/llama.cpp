// DPMoE / core — Scheduler ABC.
//
// The framework manages chunked tensors via two encoder/architecture-
// blind ABCs (ChunkedTensor, ChunkedComputation) and one global brain:
// the Scheduler.  Concrete schedulers live in model subtrees (e.g.
// qwen3::MoEScheduler in qwen3/scheduler.cpp).
//
// Narrow surface — six virtuals total:
//
//   ─ Lifecycle ───────────────────────────────────────────────────
//     on_install               wid → host bytes / chunk_io / etc.
//     on_graph_compute_begin   ggml about to walk a cgraph
//     on_graph_compute_end     ggml finished walking a cgraph
//
//   ─ Hot path ────────────────────────────────────────────────────
//     on_marker(MarkerEvent)   custom marker fires during compute
//                              (semantic boundaries: layer entry,
//                              expert dispatch, KV write, ...)
//     plan_for(...)            answer demand for an upcoming
//                              ChunkedComputation (which chunks should
//                              be resident; usually a no-op since
//                              prefetch already loaded them)
//     make_room_for(pool, n)   pool-pressure eviction callback
//
// Model-specific entry points (per-op dispatch, MoE expert tables,
// score-policy snapshot/replace) are exposed as free-function
// accessors in the model layer's header (qwen3/scheduler.h's
// scheduler_handle_mul_mat / scheduler_set_kbar / etc.). Each
// downcasts internally; core stays free of model vtable contracts.
//
// Pick at install via DP_MOE_SCHEDULER.  Currently only "moe" is
// supported; null defaults to it.

#pragma once

#include "stream_reader.h"
#include "upstream_layout.h"
#include "vram_pool.h"

#include <memory>
#include <string>
#include <vector>

struct ggml_tensor;
struct ggml_cgraph;

namespace dp_moe_ext {

class DPMoERuntime;

// Decoder-neutral per-MoE-tensor expert pointer table.  The fused
// kernel that consumes it is architecture-specific (see
// ``qwen3/fused_kernels.h``); the *struct* is encoder/architecture-
// agnostic — three device-pointer arrays indexed by expert id —
// so a forward decl here keeps core's contract free of architecture
// types while letting model-side schedulers expose getters via the
// free-function accessors in scheduler.h.
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
        DPMoERuntime &    rt,
        const StreamReader &  reader,
        const std::string &   gguf_path) = 0;

    // ggml is about to walk a cgraph.  Default no-op.  Wired through
    // a ggml-cuda hook (ggml_cuda_set_graph_compute_begin_hook) so
    // concrete schedulers can prewalk the cgraph — identify managed
    // mul_mat / mul_mat_id nodes, prefetch their chunks, mark per-
    // graph state.  ``cgraph`` is read-only; structural mutation is
    // unsupported here.
    virtual void on_graph_compute_begin(StreamHandle /*compute_stream*/,
                                         const struct ggml_cgraph * /*cgraph*/) {}

    // ggml finished walking a cgraph.  Default no-op.  Symmetric with
    // on_graph_compute_begin; release per-graph resources here.
    virtual void on_graph_compute_end(StreamHandle /*compute_stream*/,
                                       const struct ggml_cgraph * /*cgraph*/) {}

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

    // ─── Per-node claim + dispatch (P2★) ───────────────────────────
    //
    // The two-virtual contract that ggml-cuda's per-op hooks delegate
    // to. Concrete schedulers override these to identify the cgraph
    // nodes whose execution they intend to claim (Mode A sentinels
    // and, for the S10 clear-fail, managed canonicals surfacing as
    // dense MUL_MAT).
    //
    // claims_node: cheap predicate, called by ggml-cuda's
    //   user_node_claims hook before deciding whether to capture
    //   this cgraph into a cuda-graph (a true return forces eager).
    // claims_tensor: predicate over a weight tensor — used by the
    //   S10 dense-managed clear-fail to detect managed canonicals
    //   surfacing as dense MUL_MAT (which Mode A forbids).
    //
    // The legacy ``dispatch_node`` / ``observe_topk_moe`` virtuals
    // were retired in the post-M1 cleanup pass along with the legacy
    // per-op MUL_MAT_ID and fused-topk_moe hooks; sentinel dispatch
    // runs through the executor (Qwen3MoEAnyBcqExecutor::
    // forward_moe_layer), not through a scheduler virtual.
    virtual bool claims_node(const struct ggml_tensor * /*node*/) const {
        return false;
    }
    virtual bool claims_tensor(const struct ggml_tensor * /*w*/) const {
        return false;
    }

    // ─── Per-replay state (P2★) ────────────────────────────────────
    //
    // Schedulers that need per-replay state (e.g. routing reservations
    // a captured kernel reads through replays) own it themselves
    // rather than the runtime acting as a model-specific staging area.
    // Defaults are no-op so a scheduler with no per-replay concerns
    // pays nothing.
    virtual void add_replay_reservations(
        const std::vector<ChunkKey> & /*v*/) {}
    virtual bool is_replay_reserved(
        const std::string & /*wid*/, int /*cid*/) const { return false; }
    virtual void clear_replay_reservations() {}

    // Human-readable identifier for log lines.
    virtual const char * name() const = 0;
};

// Factory.  ``which`` is currently only "moe" (or nullptr → "moe").
// Anything else logs and returns null so a misspelled flag fails
// loudly instead of silently falling back.
std::unique_ptr<Scheduler> make_scheduler(const char * which);

} // namespace dp_moe_ext
