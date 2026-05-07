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
    bool is_resident(const std::string & wid, int cid) const;

    // Make ``stream`` wait on the chunk's pending H2D. No-op in sync mode
    // or if the chunk's copy has already completed.
    void wait_on_stream(const std::string & wid, int cid, StreamHandle stream);

    // Record an event on ``stream`` (default: current compute stream) so
    // subsequent async H2Ds serialize *after* it — prevents the copy
    // stream from overwriting a slot whose bytes an in-flight kernel is
    // about to read.
    void record_compute_event(StreamHandle stream = nullptr);

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

private:
    struct Slot {
        size_t offset = 0;   // byte offset inside arena
        size_t nbytes = 0;
    };

    struct Resident {
        Slot        slot;
        EventHandle ready_event;  // nullptr if sync-copied or already waited on
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

    // Serialises allocator state; H2D/kernel dispatch is lock-free on the
    // CUDA streams themselves.
    mutable std::mutex mu_;
};

} // namespace streamllm_ext
