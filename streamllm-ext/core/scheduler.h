// streamllm-ext — chunk lifecycle scheduler.
//
// The runtime exposes two primitives:
//
//   move_chunk(wid, cid, src_tier, dst_tier) -> event
//   chunk_matmul(wid, chunks, X) -> Y
//
// A Scheduler is a policy that, for each managed mul_mat, decides:
//   (1) which chunks (cids) chunk_matmul should consume — the
//       precision dial.
//   (2) which moves to issue ahead of / alongside compute.
// Both come back in a single Plan; the mul_mat hook fires the moves,
// then calls chunk_matmul with the chunks.
//
// Pick at install via STREAMLLM_SCHEDULER. Unrecognised values fall
// through to "eager" so a misspelled flag doesn't break inference.

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

// Forward-decl (defined in decoder/anybcq/moe_fused.h). Decoder-neutral
// shape: three device pointer arrays indexed by expert id, so it lives
// at the streamllm_ext namespace level and core/scheduler.h doesn't
// have to name a specific decoder family.
struct MoeExpertTable;

// Storage tier for a chunk. move_chunk(wid, cid, src, dst) currently
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

// Scheduler's instructions to the hook for one managed mul_mat.
struct Plan {
    // cids chunk_matmul must consume. Must be a subset of the chunks
    // guaranteed VRAM-resident after ``moves`` complete (or already
    // resident at entry).
    std::vector<int>    chunks;
    // Moves to issue before compute. The hook fires these synchronously;
    // move_chunk records H2D ready events so compute can overlap.
    std::vector<MoveOp> moves;
};

class Scheduler {
public:
    virtual ~Scheduler() = default;

    // One-time setup. ``rt`` is the owning runtime — the scheduler
    // calls rt.move_chunk() through it to upload chunks.
    virtual void on_install(
        StreamllmRuntime & rt,
        const StreamReader & reader,
        const std::string & gguf_path) = 0;

    // Hot-path query from the hook. Returns nullptr if ``tensor_name``
    // isn't managed (caller falls through to stock cuBLAS). Otherwise
    // returns a Plan the hook applies.
    //
    // ``compute_stream`` is ggml-cuda's current stream — schedulers that
    // overlap H2D and compute use it to record cross-stream events.
    virtual const Plan * plan(
        const std::string & tensor_name,
        StreamHandle compute_stream) = 0;

    // Called by the hook after chunk_matmul has launched and the
    // pool's compute event has been recorded. Schedulers that want
    // "flush after use" semantics evict tail chunks here. Default: no-op.
    //
    // Safety: pool.evict() only marks a slot free; a future load() that
    // reuses the slot waits on the recorded compute event before its
    // cudaMemcpyAsync — no overwrite against a still-reading kernel.
    virtual void after_compute(
        const std::string & /*tensor_name*/,
        StreamHandle /*compute_stream*/) {}

    // Reservation hooks for the load→compute window. The hook calls
    // ``reserve_for_dispatch`` for every (wid, cid) the in-flight
    // mul_mat is about to read, ``submit_prefetch``s any non-resident
    // ones, then runs the compute kernels. While reserved, the
    // scheduler must guarantee these chunks are NOT picked as
    // ``make_room_for`` victims — even if a parallel prefetch worker
    // hits a full pool. The hook calls ``release_from_dispatch`` after
    // the compute event has been recorded; subsequent dispatches can
    // then evict these chunks freely.
    //
    // Default no-op for non-streaming schedulers — they never call
    // make_room_for in the first place.
    virtual void reserve_for_dispatch(const std::string & /*wid*/,
                                       int /*cid*/) {}
    virtual void release_from_dispatch(const std::string & /*wid*/,
                                        int /*cid*/) {}

    // Fused MoE kernel: returns a per-canonical-wid expert table built
    // from each routed expert's d_chunk_qw_ptrs / d_chunk_alpha_ptrs /
    // dev.q_bias_fp16. Cached after first call. Default returns
    // nullptr — hook falls back to the per-(t, u) loop.
    virtual const MoeExpertTable * moe_expert_table(
        const std::string & /*canonical_wid*/) {
        return nullptr;
    }

    // Pool-pressure callback. Invoked by the runtime when ``pool.load()``
    // returns a null handle (arena fragmented or full). The scheduler
    // should evict one or more resident chunks via ``pool.evict(...)``
    // — picked by whatever residency policy the scheduler maintains —
    // and return true if any progress was made. Returning false means
    // "no eviction candidates left"; the caller will then surface an
    // out-of-memory error.
    //
    // The runtime calls this in a loop until ``pool.load()`` succeeds
    // or this returns false, so freeing one victim per call is fine.
    virtual bool make_room_for(VramChunkPool & /*pool*/,
                                size_t /*nbytes_needed*/) {
        return false;
    }

    // MoE per-expert plan. Optional — default returns nullptr meaning
    // "this scheduler is dense-only" so the MoE hook falls through to
    // upstream's batched dispatch. MoE-aware schedulers (e.g.
    // MoEScheduler) override this to return per-(canonical_wid,
    // expert_id) chunk sets driven by the gate-score.
    //
    // ``canonical_wid`` is the stacked-tensor name (e.g.
    // "blk.5.ffn_up_exps.weight"). ``expert_id`` is the index within
    // src0_exps->ne[2]. ``gate_score`` is the routing weight (post
    // softmax) for this token's selection of this expert; pass 0.0f if
    // not available — schedulers may treat that as "unknown, use
    // BASE+HOT".
    virtual const Plan * plan_for_expert(
        const std::string & /*canonical_wid*/,
        int /*expert_id*/,
        float /*gate_score*/,
        StreamHandle /*compute_stream*/) {
        return nullptr;
    }

    // Rank-aware variant. ``rank`` is this expert's position in the
    // current token's top-k routing, sorted by gate score descending
    // (0 = highest-gate expert; n_used-1 = lowest-gate among routed).
    // Lets a scheduler express "top-K experts get HIGH planes, the
    // rest get LOW" without trying to back out a rank from continuous
    // gate magnitudes (top-k softmax distributions don't always have
    // a clean magnitude threshold). Default forwards to the rank-less
    // overload so existing schedulers stay unchanged.
    virtual const Plan * plan_for_expert(
        const std::string & canonical_wid,
        int expert_id,
        float gate_score,
        int /*rank*/,
        StreamHandle compute_stream) {
        return plan_for_expert(canonical_wid, expert_id, gate_score,
                                compute_stream);
    }

    // Direct-precision variant. The hook has already decided the
    // desired plane count for this expert (e.g. via the score policy
    // where the per-expert precision is computed from the per-expert
    // max gate score). Skip the gate-threshold ladder; just build a
    // Plan for chunks [0, desired_precision). Default returns nullptr —
    // only `MoEScheduler` overrides.
    virtual const Plan * plan_for_expert_with_precision(
        const std::string & /*canonical_wid*/,
        int /*expert_id*/,
        int /*desired_precision*/,
        StreamHandle /*compute_stream*/) {
        return nullptr;
    }

    // Score-threshold policy accessors. Default = "off" so the hook's
    // per-expert max-score precision branch is skipped on schedulers
    // that don't implement it. MoEScheduler overrides these to expose
    // its env-parsed tables to the hook.
    //
    // ``score_thresholds`` is a descending-order list of cutoffs;
    // ``score_chunks[k]`` is the precision applied when the gate score
    // is at-least ``score_thresholds[k]`` (and below all higher entries).
    // Below the lowest threshold the lookup falls back to ``chunks.back()``.
    virtual bool is_score_policy() const { return false; }
    // Snapshot accessors — return by value because the score table can
    // be swapped at runtime via set_score_table() (live precision dial
    // for the demo path: each chat-completion request can carry its own
    // exact threshold + chunks arrays). Hot-path callers take one
    // snapshot per minibatch and iterate locally without holding a
    // lock during compute.
    virtual std::vector<float> score_thresholds_snapshot() const {
        return {};
    }
    virtual std::vector<int>   score_chunks_snapshot()    const {
        return {};
    }

    // Atomically replace the score-policy table. ``thresholds`` is a
    // descending-order list of gate-score cutoffs; ``chunks[k]`` is the
    // precision applied when the gate score is at-least
    // ``thresholds[k]`` (and below all higher entries). Below the
    // smallest threshold the lookup falls back to ``chunks.back()``.
    // Returns false if the input is malformed (sizes mismatch, empty,
    // non-descending thresholds, etc.). The next dispatch sees the new
    // table; an in-flight dispatch keeps using the snapshot it took at
    // entry, so requests don't tear mid-generation.
    virtual bool set_score_table(
        const std::vector<float> & /*thresholds*/,
        const std::vector<int>   & /*chunks*/) {
        return false;
    }

    // ── ggml-cuda hook entry points ────────────────────────────────
    //
    // The runtime registers thin extern-C shims with ggml-cuda; those
    // shims forward straight here so all model-specific dispatch logic
    // (graph introspection, gate-score reading, per-(t, u) plan,
    // chunk fan-out, kernel launch) lives in the scheduler subclass
    // for the active model architecture, not in core/.
    //
    // Default implementations return false / are no-ops so a model
    // that doesn't override them just falls through to stock
    // ggml-cuda dispatch.

    // Dense managed mul_mat: scheduler may handle the op (return true)
    // or pass (return false). ``stream`` is ggml-cuda's compute stream.
    virtual bool handle_mul_mat(StreamHandle /*stream*/,
                                 const struct ggml_tensor * /*src0*/,
                                 const struct ggml_tensor * /*src1*/,
                                 struct ggml_tensor *       /*dst*/) {
        return false;
    }

    // MoE mul_mat_id: routed expert dispatch.
    virtual bool handle_mul_mat_id(StreamHandle /*stream*/,
                                    const struct ggml_tensor * /*src0*/,
                                    const struct ggml_tensor * /*src1*/,
                                    const struct ggml_tensor * /*ids*/,
                                    struct ggml_tensor *       /*dst*/) {
        return false;
    }

    // Notification before ggml-cuda's fused softmax+argsort+norm
    // kernel. Lets a MoE scheduler stash the (ids, weights) pair so
    // it can later read post-norm routing weights instead of the
    // bypassed selection-probs buffer.
    virtual void on_topk_moe_observed(StreamHandle /*stream*/,
                                       const struct ggml_tensor * /*logits*/,
                                       struct ggml_tensor *       /*weights*/,
                                       struct ggml_tensor *       /*ids*/) {}

    // Fusion-skip query — ggml-cuda calls this on candidate weight
    // tensors for fused subgraphs. Returning true disables fusion so
    // the (managed) tensor reaches the scheduler's mul_mat handler.
    virtual bool claims_tensor(const struct ggml_tensor * /*w*/) {
        return false;
    }

    // Human-readable identifier for log lines.
    virtual const char * name() const = 0;
};

// Factory. ``which`` is one of "eager", "lazy" — or nullptr, which
// maps to "eager". Never returns nullptr.
std::unique_ptr<Scheduler> make_scheduler(const char * which);

} // namespace streamllm_ext
