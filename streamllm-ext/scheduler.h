// streamllm-ext — chunk lifecycle scheduler.
//
// The runtime exposes exactly two primitives to everything above it:
//
//   move_chunk(wid, cid, src_tier, dst_tier) -> event
//   chunk_matmul(wid, chunks, X) -> Y
//
// A Scheduler is a thin policy that, for each managed mul_mat, decides
// two things:
//
//   1. Which chunks (cids) chunk_matmul should consume — i.e. the
//      precision dial.
//   2. Which moves to issue ahead of / alongside compute — prefetch,
//      eviction, tier transitions.
//
// It returns both in a single ``Plan`` struct; the mul_mat hook fires
// the moves, then calls chunk_matmul with the chunks. Concrete policies:
//
//   Eager    — all chunks uploaded at install. plan() returns
//              {chunks = all planes, moves = none}.
//   Lazy     — nothing uploaded at install. plan() emits a one-time
//              SSD→VRAM move on first touch.
//   (future)
//   LayerPrefetch — plan() for layer N also schedules moves for layer N+L.
//   Budgeted       — plan() consults a VRAM-cap + tps-floor controller and
//                    trims chunks to whatever finished loading by the
//                    deadline.
//
// Pick at install time via the env var ``STREAMLLM_SCHEDULER`` —
// values: eager (default), lazy. An unrecognised value falls through
// to eager so a misspelled flag doesn't break inference.

#pragma once

#include "stream_reader.h"
#include "upstream_layout.h"
#include "vram_pool.h"

#include <memory>
#include <string>
#include <vector>

namespace streamllm_ext {

class StreamllmRuntime;

// Storage tier for a chunk. move_chunk(wid, cid, src, dst) is defined
// for any (src, dst) pair but M6's runtime only implements the RAM↔VRAM
// edge — SSD arrives with M10 (pread into a pinned-host buffer pool) and
// is currently folded into RAM at install time.
enum class Tier : int {
    SSD  = 0,  // reserved; on-disk GGUF tensor_offset region
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
    // Moves to issue before compute. The hook fires these synchronously
    // today; M7's async move_chunk will record wait events so compute
    // can overlap H2D.
    std::vector<MoveOp> moves;
};

class Scheduler {
public:
    virtual ~Scheduler() = default;

    // One-time setup: hand the scheduler references it'll use for the
    // process lifetime. ``rt`` is the owning runtime — the scheduler
    // calls rt.move_chunk() through it to upload at install time (Eager)
    // or on first touch (Lazy).
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
    // true "flush after use" semantics (e.g. Scenario C explicit
    // streaming) evict tail chunks here. Default: no-op.
    //
    // Safety: pool.evict() only marks a slot free; a future load()
    // that reuses the slot issues cudaMemcpyAsync on the copy stream,
    // which waits on the recorded compute event — no overwrite
    // against a still-reading kernel.
    virtual void after_compute(
        const std::string & /*tensor_name*/,
        StreamHandle /*compute_stream*/) {}

    // Human-readable identifier for log lines.
    virtual const char * name() const = 0;
};

// Factory. ``which`` is one of "eager", "lazy" — or nullptr, which
// maps to "eager". Never returns nullptr.
std::unique_ptr<Scheduler> make_scheduler(const char * which);

} // namespace streamllm_ext
