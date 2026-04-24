// streamllm-ext — VRAM chunk pool implementation.

#include "vram_pool.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace streamllm_ext {

namespace {

inline void check_cuda(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        throw std::runtime_error(
            std::string("vram_pool: CUDA error in ") + what + ": " +
            cudaGetErrorString(e));
    }
}

} // anon


VramChunkPool::VramChunkPool(size_t capacity_bytes, int device, bool copy_stream)
    : capacity_bytes_(capacity_bytes),
      device_(device),
      arena_(nullptr),
      copy_stream_(nullptr),
      latest_compute_event_(nullptr),
      capture_fork_event_(nullptr)
{
    if (capacity_bytes_ == 0) {
        throw std::runtime_error("vram_pool: capacity_bytes must be > 0");
    }
    int prev = 0;
    check_cuda(cudaGetDevice(&prev), "cudaGetDevice");
    check_cuda(cudaSetDevice(device_), "cudaSetDevice");
    check_cuda(cudaMalloc(&arena_, capacity_bytes_), "cudaMalloc(arena)");
    if (copy_stream) {
        cudaStream_t s;
        check_cuda(
            cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking),
            "cudaStreamCreateWithFlags");
        copy_stream_ = (StreamHandle)s;
    }
    check_cuda(cudaSetDevice(prev), "cudaSetDevice(restore)");

    free_list_.push_back({0, capacity_bytes_});
}


VramChunkPool::~VramChunkPool() {
    // Best-effort cleanup; ignore errors in the destructor.
    for (auto & [k, r] : residents_) {
        if (r.ready_event) {
            cudaEventDestroy((cudaEvent_t)r.ready_event);
        }
    }
    if (latest_compute_event_) {
        cudaEventDestroy((cudaEvent_t)latest_compute_event_);
    }
    if (capture_fork_event_) {
        cudaEventDestroy((cudaEvent_t)capture_fork_event_);
    }
    for (auto h : pending_event_destroys_) {
        cudaEventDestroy((cudaEvent_t)h);
    }
    if (copy_stream_) {
        cudaStreamDestroy((cudaStream_t)copy_stream_);
    }
    if (arena_) {
        cudaFree(arena_);
    }
}


// Try to find a span at least nbytes large. Split if strictly larger.
std::optional<size_t> VramChunkPool::allocate_(size_t nbytes) {
    for (size_t i = 0; i < free_list_.size(); ++i) {
        if (free_list_[i].nbytes >= nbytes) {
            size_t off = free_list_[i].offset;
            if (free_list_[i].nbytes == nbytes) {
                free_list_.erase(free_list_.begin() + (ptrdiff_t)i);
            } else {
                free_list_[i].offset += nbytes;
                free_list_[i].nbytes -= nbytes;
            }
            used_bytes_ += nbytes;
            return off;
        }
    }
    return std::nullopt;
}


void VramChunkPool::free_(size_t offset, size_t nbytes) {
    used_bytes_ -= nbytes;
    FreeSpan span{offset, nbytes};

    // Insert sorted by offset, then coalesce with neighbours.
    auto it = std::lower_bound(
        free_list_.begin(), free_list_.end(), span,
        [](const FreeSpan & a, const FreeSpan & b) { return a.offset < b.offset; });
    it = free_list_.insert(it, span);

    // Coalesce with right neighbour.
    auto next = it + 1;
    if (next != free_list_.end() && it->offset + it->nbytes == next->offset) {
        it->nbytes += next->nbytes;
        free_list_.erase(next);
    }
    // Coalesce with left neighbour.
    if (it != free_list_.begin()) {
        auto prev = it - 1;
        if (prev->offset + prev->nbytes == it->offset) {
            prev->nbytes += it->nbytes;
            free_list_.erase(it);
        }
    }
}


bool VramChunkPool::make_room_(size_t nbytes) {
    while (true) {
        if (allocate_(nbytes).has_value()) {
            // We over-consumed by committing early; undo and re-attempt.
            // Simpler: just do a dry check by iterating free_list_.
            // (allocate_ above already popped a span; nothing to undo —
            // caller should have called make_room_ only to decide eviction.)
            // We don't use this path — see below.
            return true;
        }
        // Find an evictable (unpinned) chunk at the LRU head.
        bool evicted = false;
        for (auto it = lru_.begin(); it != lru_.end(); ++it) {
            auto pin_it = pinned_.find(*it);
            if (pin_it != pinned_.end() && pin_it->second) continue;
            // Evict.
            const ChunkKey key = *it;
            auto res_it = residents_.find(key);
            if (res_it == residents_.end()) {
                lru_.erase(it);
                evicted = true;
                break;
            }
            if (res_it->second.ready_event) {
                cudaEventDestroy((cudaEvent_t)res_it->second.ready_event);
            }
            Slot slot = res_it->second.slot;
            lru_.erase(it);
            residents_.erase(res_it);
            free_(slot.offset, slot.nbytes);
            evicted = true;
            break;
        }
        if (!evicted) return false;
    }
}


void VramChunkPool::launch_copy_(
    size_t dst_offset, const void * src, size_t nbytes,
    EventHandle & out_event,
    StreamHandle compute_stream_opt)
{
    char * dst = (char *)arena_ + dst_offset;
    if (copy_stream_ == nullptr) {
        // Sync path — blocking memcpy on current stream.
        check_cuda(
            cudaMemcpy(dst, src, nbytes, cudaMemcpyHostToDevice),
            "cudaMemcpy(H2D sync)");
        out_event = nullptr;
        return;
    }
    auto stream = (cudaStream_t)copy_stream_;

    // CUDA graph capture fork-join: if the caller's compute stream is in
    // an active capture, record a fresh event on it and have copy_stream
    // wait on that event. This brings copy_stream into the same capture
    // session, so the downstream wait_on_stream(compute_stream, ready_ev)
    // becomes a legal intra-capture dependency. Without this, swap1
    // (per-call H2D during a captured decode step) dies with
    // "dependency created on uncaptured work in another stream".
    //
    // Under capture, we skip the "chain after latest_compute_event_"
    // branch: that event was recorded outside the current capture
    // session and waiting on it from inside capture is exactly what the
    // driver rejects. The fork event is sufficient — the captured graph
    // will serialize copy-stream H2D after whatever the compute stream
    // was doing at the fork point.
    bool in_capture = false;
    if (compute_stream_opt != nullptr) {
        auto cs = (cudaStream_t) compute_stream_opt;
        cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(cs, &status) == cudaSuccess &&
            status == cudaStreamCaptureStatusActive) {
            in_capture = true;
            // Persistent fork event — reused across every H2D in every
            // capture. Avoids per-H2D event create/destroy churn, and
            // keeps the event handle stable for the graph's internal
            // references.
            if (!capture_fork_event_) {
                cudaEvent_t ev;
                check_cuda(
                    cudaEventCreateWithFlags(&ev, cudaEventDisableTiming),
                    "cudaEventCreate(fork)");
                capture_fork_event_ = (EventHandle) ev;
            }
            check_cuda(
                cudaEventRecord((cudaEvent_t)capture_fork_event_, cs),
                "cudaEventRecord(fork)");
            check_cuda(
                cudaStreamWaitEvent(stream, (cudaEvent_t)capture_fork_event_, 0),
                "cudaStreamWaitEvent(fork)");
        }
    }
    if (!in_capture && latest_compute_event_) {
        // Non-capture mode: serialize H2D behind in-flight compute so we
        // don't overwrite a slot a kernel is still reading from.
        check_cuda(
            cudaStreamWaitEvent(stream, (cudaEvent_t)latest_compute_event_, 0),
            "cudaStreamWaitEvent(compute)");
    }
    check_cuda(
        cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyHostToDevice, stream),
        "cudaMemcpyAsync(H2D)");
    cudaEvent_t ev;
    check_cuda(
        cudaEventCreateWithFlags(&ev, cudaEventDisableTiming),
        "cudaEventCreate(ready)");
    check_cuda(cudaEventRecord(ev, stream), "cudaEventRecord(ready)");
    out_event = (EventHandle)ev;
}


ChunkHandle VramChunkPool::load(
    const std::string & wid, int cid,
    const void * host_ptr, size_t nbytes,
    StreamHandle compute_stream)
{
    std::lock_guard<std::mutex> lk(mu_);
    ChunkKey key{wid, cid};

    auto it = residents_.find(key);
    if (it != residents_.end()) {
        // Already resident — touch LRU, return view.
        lru_.erase(it->second.lru_it);
        lru_.push_back(key);
        it->second.lru_it = std::prev(lru_.end());
        return ChunkHandle{
            (char *)arena_ + it->second.slot.offset,
            it->second.slot.nbytes,
            it->second.ready_event,
        };
    }

    if (nbytes > capacity_bytes_) {
        throw std::runtime_error(
            "vram_pool: chunk (" + wid + "," + std::to_string(cid) + ") size " +
            std::to_string(nbytes) + " exceeds capacity " +
            std::to_string(capacity_bytes_));
    }

    // Reserve space (evict LRU if needed).
    std::optional<size_t> off_opt = allocate_(nbytes);
    while (!off_opt.has_value()) {
        // Evict one LRU entry.
        bool any = false;
        for (auto lru_it = lru_.begin(); lru_it != lru_.end(); ++lru_it) {
            auto pin_it = pinned_.find(*lru_it);
            if (pin_it != pinned_.end() && pin_it->second) continue;
            const ChunkKey evict_key = *lru_it;
            auto res_it = residents_.find(evict_key);
            if (res_it != residents_.end()) {
                if (res_it->second.ready_event) {
                    cudaEventDestroy((cudaEvent_t)res_it->second.ready_event);
                }
                Slot slot = res_it->second.slot;
                lru_.erase(lru_it);
                residents_.erase(res_it);
                free_(slot.offset, slot.nbytes);
            } else {
                lru_.erase(lru_it);
            }
            any = true;
            break;
        }
        if (!any) {
            throw std::runtime_error(
                "vram_pool: cannot make room for " + std::to_string(nbytes) +
                " bytes (all residents pinned)");
        }
        off_opt = allocate_(nbytes);
    }

    Slot slot{*off_opt, nbytes};
    EventHandle evt = nullptr;
    if (host_ptr != nullptr) {
        launch_copy_(slot.offset, host_ptr, nbytes, evt, compute_stream);
        total_h2d_bytes_ += nbytes;
        ++total_h2d_calls_;
    }

    lru_.push_back(key);
    Resident r{slot, evt, std::prev(lru_.end())};
    residents_.emplace(key, r);
    if (used_bytes_ > peak_used_bytes_) peak_used_bytes_ = used_bytes_;

    return ChunkHandle{
        (char *)arena_ + slot.offset,
        slot.nbytes,
        evt,
    };
}


void VramChunkPool::reset_stats() {
    std::lock_guard<std::mutex> lk(mu_);
    peak_used_bytes_ = used_bytes_;
    total_h2d_bytes_ = 0;
    total_h2d_calls_ = 0;
}


void VramChunkPool::evict(const std::string & wid, int cid) {
    std::lock_guard<std::mutex> lk(mu_);
    ChunkKey key{wid, cid};
    auto pin_it = pinned_.find(key);
    if (pin_it != pinned_.end() && pin_it->second) return;

    auto it = residents_.find(key);
    if (it == residents_.end()) return;

    if (it->second.ready_event) {
        cudaEventDestroy((cudaEvent_t)it->second.ready_event);
    }
    Slot slot = it->second.slot;
    lru_.erase(it->second.lru_it);
    residents_.erase(it);
    free_(slot.offset, slot.nbytes);
}


void VramChunkPool::pin(const std::string & wid, int cid) {
    std::lock_guard<std::mutex> lk(mu_);
    pinned_[{wid, cid}] = true;
}
void VramChunkPool::unpin(const std::string & wid, int cid) {
    std::lock_guard<std::mutex> lk(mu_);
    pinned_[{wid, cid}] = false;
}


ChunkHandle VramChunkPool::view(const std::string & wid, int cid) const {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = residents_.find({wid, cid});
    if (it == residents_.end()) return ChunkHandle{};
    return ChunkHandle{
        (char *)arena_ + it->second.slot.offset,
        it->second.slot.nbytes,
        it->second.ready_event,
    };
}

bool VramChunkPool::is_resident(const std::string & wid, int cid) const {
    std::lock_guard<std::mutex> lk(mu_);
    return residents_.find({wid, cid}) != residents_.end();
}


void VramChunkPool::wait_on_stream(
    const std::string & wid, int cid, StreamHandle stream)
{
    std::lock_guard<std::mutex> lk(mu_);
    auto it = residents_.find({wid, cid});
    if (it == residents_.end() || it->second.ready_event == nullptr) return;
    auto ev = (cudaEvent_t)it->second.ready_event;
    auto s  = stream ? (cudaStream_t)stream
                     : (cudaStream_t)0;
    check_cuda(cudaStreamWaitEvent(s, ev, 0), "cudaStreamWaitEvent(chunk)");
    // Event consumed — drop it so a second load for the same chunk isn't
    // chained against a stale event. Any future load re-records a new one.
    //
    // Under CUDA graph capture, destroying an event that was just
    // captured in a wait node leaves the graph referencing a destroyed
    // handle — cudaStreamEndCapture or cudaGraphInstantiate then abort.
    // Defer the destroy: push it onto pending_event_destroys_ and drain
    // after capture ends. ready_event pointer is cleared regardless.
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    const bool in_capture =
        s != 0 &&
        cudaStreamIsCapturing(s, &status) == cudaSuccess &&
        status == cudaStreamCaptureStatusActive;
    if (in_capture) {
        pending_event_destroys_.push_back((EventHandle)ev);
    } else {
        // Drain any events we deferred during a previous capture window
        // while we're confirmed out of capture.
        for (auto h : pending_event_destroys_) {
            cudaEventDestroy((cudaEvent_t)h);
        }
        pending_event_destroys_.clear();
        cudaEventDestroy(ev);
    }
    it->second.ready_event = nullptr;
}


void VramChunkPool::record_compute_event(StreamHandle stream) {
    std::lock_guard<std::mutex> lk(mu_);
    if (copy_stream_ == nullptr) return;  // sync mode: nothing to serialize
    auto s = stream ? (cudaStream_t)stream : (cudaStream_t)0;
    // Under CUDA graph capture, the event would get captured into the
    // graph's internal state and the handle becomes meaningless outside
    // that capture session. Subsequent non-captured launch_copy_ calls
    // would then wait on a stale captured event. Skip the update —
    // cross-stream ordering under capture is handled by the fork-join
    // event inside launch_copy_ itself.
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    if (s != 0 && cudaStreamIsCapturing(s, &status) == cudaSuccess &&
        status == cudaStreamCaptureStatusActive) {
        return;
    }
    if (!latest_compute_event_) {
        cudaEvent_t ev;
        check_cuda(
            cudaEventCreateWithFlags(&ev, cudaEventDisableTiming),
            "cudaEventCreate(compute)");
        latest_compute_event_ = (EventHandle)ev;
    }
    check_cuda(
        cudaEventRecord((cudaEvent_t)latest_compute_event_, s),
        "cudaEventRecord(compute)");
}

} // namespace streamllm_ext
