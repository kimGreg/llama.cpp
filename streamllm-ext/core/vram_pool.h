// streamllm-ext — VRAM chunk pool (dumb slot allocator).
//
// A single arena of size ``capacity_bytes`` is ``cudaMalloc``-ed once.
// Callers ``load()`` byte ranges from the host — typically one ``.gguf``
// tensor's chunk subset — and receive a VRAM pointer they can pass to
// kernel launches.
//
// **Pool is algorithm-blind and policy-free.** It does NOT maintain
// LRU, pin/unpin state, or auto-evict on full. When ``load()`` cannot
// allocate the request (arena fragmented or bytes simply don't fit),
// it returns ``ChunkHandle{}`` (null device_ptr). The caller — usually
// a model-specific scheduler — decides which resident chunk to evict
// and explicitly calls ``evict()`` before retrying ``load()``.
//
// Async mode: pass ``copy_stream = true`` to allocate a dedicated CUDA
// copy stream for H2D transfers. ``load()`` then returns a handle whose
// ``ready_event`` the caller chains onto its compute stream before
// reading the chunk. No-op in sync mode.

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

struct CUstream_st;
struct CUevent_st;

namespace streamllm_ext {

// Opaque CUDA handles — the header doesn't include <cuda_runtime.h> so
// downstream code outside nvcc-compiled .cu files can still include it.
using StreamHandle = CUstream_st *;
using EventHandle  = CUevent_st *;

// Uniform handle for "chunk-in-VRAM": device pointer + optional event to
// wait on before reading. ``device_ptr == nullptr`` means the load
// failed (arena full / fragmented). ``ready_event`` is non-null only in
// async mode and only when the H2D hasn't been waited on yet.
struct ChunkHandle {
    void *       device_ptr  = nullptr;
    size_t       nbytes      = 0;
    EventHandle  ready_event = nullptr;
};

// Per-chunk residency-progress state (Mode A Milestone 1 Step 6).
//
// A chunk advances monotonically through these states as the loader
// pipeline processes it.  Eviction resets to NOT_RESIDENT.  Backward
// transitions other than eviction are bugs.
//
// Host-side validation (Qwen3MoEAnyBcqExecutor::validate_required_set_)
// checks ``chunk_state(wid, cid) >= POINTER_TABLE_READY`` for every
// chunk the kernel will read, before launching the fused MoE op.
// The device-side M1 required-non-null trap (fused_kernels.cu) is the
// final guard; Step 6 catches misses earlier with a useful message.
enum class ChunkState : uint8_t {
    NOT_RESIDENT        = 0,   // not in the pool
    SLOT_ALLOCATED      = 1,   // slot reserved; H2D not yet enqueued
    H2D_ISSUED          = 2,   // cudaMemcpyAsync enqueued; ready_event recorded
    POINTER_TABLE_READY = 3,   // loader ran ChunkedTensor::after_load —
                               // per-plane device pointer-table store is
                               // visible after a wait on ready_event
    // KERNEL_READY is conceptual rather than tracked per-(stream, chunk).
    // Step 6 derives it from "POINTER_TABLE_READY + caller has called
    // wait_on_stream(this stream) since the last after_load", which the
    // single-stream Mode A pipeline holds by construction.
};

// Chunks are identified by (wid, cid). wid is a string — usually the
// GGUF tensor name; cid is an integer plane / offset index.
struct ChunkKey {
    std::string wid;
    int         cid = 0;

    bool operator==(const ChunkKey & o) const {
        return cid == o.cid && wid == o.wid;
    }
};

struct ChunkKeyHash {
    size_t operator()(const ChunkKey & k) const noexcept {
        size_t h = std::hash<std::string>()(k.wid);
        return h ^ (std::hash<int>()(k.cid) * 0x9e3779b97f4a7c15ULL);
    }
};


class VramChunkPool {
public:
    // capacity_bytes: arena size.
    // device: CUDA device index.
    // copy_stream: if true, allocate a dedicated copy stream for async H2D.
    VramChunkPool(size_t capacity_bytes, int device, bool copy_stream);
    ~VramChunkPool();

    VramChunkPool(const VramChunkPool &) = delete;
    VramChunkPool & operator=(const VramChunkPool &) = delete;

    // Upload ``nbytes`` of host memory into VRAM, associated with
    // (wid, cid). If already resident, no re-upload happens (and host_ptr
    // may be null) — the existing handle is returned.
    //
    // If the arena cannot fit ``nbytes``, returns ``ChunkHandle{}`` (null
    // device_ptr). The caller is responsible for evicting another chunk
    // and retrying — the pool itself does NOT pick a victim.
    //
    // ``compute_stream`` is optional and only consulted under CUDA graph
    // capture: when the provided stream is in an active capture, the
    // pool's copy stream is forked into the same capture session via an
    // event so downstream ``wait_on_stream(compute_stream)`` waits are
    // legal (cross-stream dependencies otherwise trigger the
    // "dependency created on uncaptured work" error).
    ChunkHandle load(const std::string & wid, int cid,
                     const void * host_ptr, size_t nbytes,
                     StreamHandle compute_stream = nullptr);

    // Evict a chunk. No-op if not resident. Always evicts when resident
    // — pool does not track pin state.
    void evict(const std::string & wid, int cid);

    // View an already-resident chunk. Returns nullptr device_ptr if not
    // resident (caller can then load() it).
    ChunkHandle view(const std::string & wid, int cid) const;
    // Back-compat shim: returns true iff chunk_state >= SLOT_ALLOCATED.
    // Note this is too weak to imply kernel-readiness — H2D may still
    // be in flight, and after_load may not have run. Use
    // after_load_done / chunk_state for stronger checks.  (See SSOT
    // §6.4.2.2 / Milestone 1 Step 6.)
    bool is_resident(const std::string & wid, int cid) const;

    // Step 6 (Milestone 1): residency-progress accessors. None of
    // these read device memory — pure host state queries.
    ChunkState chunk_state(const std::string & wid, int cid) const;
    bool       after_load_done(const std::string & wid, int cid) const;
    // True iff chunk_state >= POINTER_TABLE_READY.  ``stream`` is
    // reserved for a future per-(stream, chunk) waited-set; for
    // milestone 1 it is unused — the executor's call sequence
    // (submit_loads → wait_loads_on(stream) → validate) establishes
    // stream-ordering against ready_event by construction (see
    // docs/MODE_A_MILESTONE1.md Step 6 "Host validation API").
    bool       kernel_ready_on(const std::string & wid, int cid,
                                StreamHandle stream) const;

    // Loader transition: advances chunk_state to POINTER_TABLE_READY.
    // Called by ``StreamllmRuntime::move_chunk`` after the encoder's
    // ChunkedTensor::after_load callback has queued the per-plane
    // pointer-table store on copy_stream AND the ready_event has been
    // re-recorded post-after_load.  Idempotent — second call is a no-op.
    void       mark_pointer_table_ready(const std::string & wid, int cid);

    // Make ``stream`` wait on the chunk's pending H2D. No-op in sync mode
    // or if the chunk's copy has already completed.
    void wait_on_stream(const std::string & wid, int cid, StreamHandle stream);

    // Record an event on ``stream`` (default: current compute stream) so
    // subsequent async H2Ds serialize *after* it — prevents the copy
    // stream from overwriting a slot whose bytes an in-flight kernel is
    // about to read.
    void record_compute_event(StreamHandle stream = nullptr);

    // Make ``stream`` (typically copy_stream) wait on the latest
    // compute event before issuing further work.  Used by the
    // eviction path so an ``after_evict`` kernel that nulls per-plane
    // pointers doesn't race a still-running compute kernel reading
    // through the same pointer table.  No-op if no compute event has
    // been recorded yet or if ``stream`` is in active CUDA-graph
    // capture (the launch path handles that separately).
    void wait_compute_on(StreamHandle stream);

    // Total capacity / current usage.
    size_t capacity_bytes() const { return capacity_bytes_; }
    size_t used_bytes()     const { return used_bytes_; }
    size_t peak_used_bytes() const { return peak_used_bytes_; }
    size_t n_resident()     const { return residents_.size(); }
    bool   has_copy_stream() const { return copy_stream_ != nullptr; }
    StreamHandle copy_stream() const { return copy_stream_; }
    int device() const { return device_; }

    // Cumulative H2D transfer statistics. Incremented each time the pool
    // actually copies bytes (``load()`` with ``host_ptr != nullptr`` and
    // the chunk not already resident).
    size_t total_h2d_bytes() const { return total_h2d_bytes_; }
    size_t total_h2d_calls() const { return total_h2d_calls_; }
    // Reset the cumulative counters — useful between benchmark phases
    // (e.g. exclude install warmup from per-step streaming totals).
    void reset_stats();

    // ─── Residency-skip invariant counters (always-on) ────────────────
    //
    // Contract: if ``ChunkKey`` is resident in the pool, the
    // scheduler/load path MUST NOT submit a new H2D for it.  These
    // counters expose the contract to runtime stats and to in-process
    // assertions.  All are simple atomic counters — fine to read
    // without the pool mutex.
    //
    //  required_chunks                 sum of plan.required_set across
    //                                  every chunked dispatch
    //  resident_hits                   required chunks that were already
    //                                  in the pool when plan ran
    //  load_misses                     required chunks that were NOT
    //                                  resident when plan ran (these
    //                                  drive new H2D submissions)
    //  h2d_submitted_chunks            actual cudaMemcpyAsync H2D
    //                                  operations (== total_h2d_calls_,
    //                                  republished under the contract name)
    //  redundant_h2d_skipped           ``move_chunk`` short-circuited a
    //                                  call for a chunk that was already
    //                                  resident (caller bypassed the
    //                                  executor's pre-check; we caught
    //                                  the redundancy at the chokepoint)
    //  unexpected_h2d_for_resident     ``pool.load`` was entered with
    //                                  ``host_ptr != nullptr`` for a key
    //                                  that was already resident.  Must
    //                                  be 0 if ``move_chunk``'s skip is
    //                                  correct.  A non-zero value means
    //                                  a caller bypassed move_chunk OR a
    //                                  race promoted the chunk between
    //                                  the chokepoint check and pool.load.
    size_t required_chunks() const;
    size_t resident_hits() const;
    size_t load_misses() const;
    size_t h2d_submitted_chunks() const { return total_h2d_calls(); }
    size_t redundant_h2d_skipped() const;
    size_t unexpected_h2d_for_resident() const;

    // ``move_chunk``'s chokepoint reports a required-set walk through
    // these.  ``n_required`` is plan.required_set.size(); ``n_resident``
    // is how many were already in the pool.  Both bump in lock-step so
    // ``resident_hits + load_misses == required_chunks`` always holds.
    void note_required_set(size_t n_required, size_t n_resident);
    // Bumped by ``move_chunk`` when the resident-skip fires.
    void note_redundant_h2d_skipped();

private:
    struct Slot {
        size_t offset = 0;   // byte offset inside arena
        size_t nbytes = 0;
    };

    struct Resident {
        Slot        slot;
        EventHandle ready_event;  // nullptr if sync-copied or already waited on
        ChunkState  state = ChunkState::NOT_RESIDENT;  // Step 6 progression
    };

    // First-fit free-list helpers.
    std::optional<size_t> allocate_(size_t nbytes);
    void free_(size_t offset, size_t nbytes);

    void launch_copy_(size_t dst_offset, const void * src, size_t nbytes,
                      EventHandle & out_event,
                      StreamHandle compute_stream = nullptr);

    // --- state -----------------------------------------------------------
    size_t       capacity_bytes_;
    int          device_;
    void *       arena_;
    StreamHandle copy_stream_;
    EventHandle  latest_compute_event_;  // set by record_compute_event
    EventHandle  capture_fork_event_;    // persistent; used to fork copy_stream into a CUDA graph capture
    std::vector<EventHandle> pending_event_destroys_;  // events deferred for destroy until after capture
    std::vector<EventHandle> event_free_list_;  // recycled per-chunk ready events — avoids cudaEventCreate/Destroy on every move

    // Free-list: ordered (offset, length). We keep it sorted and coalesce.
    struct FreeSpan { size_t offset; size_t nbytes; };
    std::vector<FreeSpan> free_list_;

    // Resident map. Lookup-only — no LRU / pin / eviction priority is
    // tracked here. Schedulers maintain their own residency / policy
    // structures and call evict() explicitly.
    std::unordered_map<ChunkKey, Resident, ChunkKeyHash> residents_;

    size_t used_bytes_      = 0;
    size_t peak_used_bytes_  = 0;
    size_t total_h2d_bytes_ = 0;
    size_t total_h2d_calls_ = 0;

    // Residency-invariant atomics — see public accessors above.
    std::atomic<size_t> required_chunks_{0};
    std::atomic<size_t> resident_hits_{0};
    std::atomic<size_t> load_misses_{0};
    std::atomic<size_t> redundant_h2d_skipped_{0};
    std::atomic<size_t> unexpected_h2d_for_resident_{0};

    // Serialises allocator state; H2D/kernel dispatch is lock-free on the
    // CUDA streams themselves.
    mutable std::mutex mu_;
};

} // namespace streamllm_ext
