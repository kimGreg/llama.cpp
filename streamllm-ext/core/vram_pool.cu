// streamllm-ext — VRAM chunk pool implementation.
//
// Dumb slot allocator. No LRU, no pin/unpin, no auto-eviction.
// load() returns a null handle on full; the caller drives policy.

#include "vram_pool.h"
#include "launch_diag.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
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
      force_sync_h2d_(false),
      latest_compute_event_(nullptr),
      capture_fork_event_(nullptr)
{
    if (capacity_bytes_ == 0) {
        throw std::runtime_error("vram_pool: capacity_bytes must be > 0");
    }
    int prev = device_;
    const cudaError_t get_prev = cudaGetDevice(&prev);
    const bool have_prev = get_prev == cudaSuccess;
    if (!have_prev) {
        // In some llama.cpp load orders the CUDA driver is visible to
        // ggml but no current cudart device has been established for
        // this extension yet. Clear the stale runtime error and set the
        // requested device explicitly below.
        (void) cudaGetLastError();
        prev = device_;
    }
    check_cuda(cudaSetDevice(device_), "cudaSetDevice");
    check_cuda(cudaMalloc(&arena_, capacity_bytes_), "cudaMalloc(arena)");
    if (copy_stream) {
        cudaStream_t s;
        check_cuda(
            cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking),
            "cudaStreamCreateWithFlags");
        copy_stream_ = (StreamHandle)s;
    }
    if (const char * sync = std::getenv("STREAMLLM_SYNC_H2D")) {
        force_sync_h2d_ = std::atoi(sync) != 0;
    }
    if (have_prev) {
        check_cuda(cudaSetDevice(prev), "cudaSetDevice(restore)");
    }

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
    for (auto h : event_free_list_) {
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


void VramChunkPool::launch_copy_(
    size_t dst_offset, const void * src, size_t nbytes,
    EventHandle & out_event,
    StreamHandle compute_stream_opt)
{
    char * dst = (char *)arena_ + dst_offset;
    if (copy_stream_ == nullptr || force_sync_h2d_) {
        // Sync path — blocking memcpy on current stream.
        check_cuda(
            cudaMemcpy(dst, src, nbytes, cudaMemcpyHostToDevice),
            "cudaMemcpy(H2D sync)");
        if (copy_stream_ != nullptr) {
            cudaEvent_t ev;
            if (!event_free_list_.empty()) {
                ev = (cudaEvent_t)event_free_list_.back();
                event_free_list_.pop_back();
            } else {
                check_cuda(
                    cudaEventCreateWithFlags(&ev, cudaEventDisableTiming),
                    "cudaEventCreate(ready)");
            }
            check_cuda(cudaEventRecord(ev, (cudaStream_t)copy_stream_),
                       "cudaEventRecord(ready sync)");
            out_event = (EventHandle)ev;
            return;
        }
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
    if (!event_free_list_.empty()) {
        ev = (cudaEvent_t)event_free_list_.back();
        event_free_list_.pop_back();
    } else {
        check_cuda(
            cudaEventCreateWithFlags(&ev, cudaEventDisableTiming),
            "cudaEventCreate(ready)");
    }
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
        // Already resident — return view, no LRU touch (pool is policy-free).
        //
        // Residency-invariant trip: if the caller passed bytes
        // intending to copy, that's the chokepoint signal that
        // ``move_chunk``'s resident-skip is being bypassed (or a race
        // promoted the chunk between the chokepoint check and here).
        // The skipped copy is harmless — the slot is already populated
        // — but the count surfaces the broken contract.
        if (host_ptr != nullptr) {
            unexpected_h2d_for_resident_.fetch_add(
                1, std::memory_order_relaxed);
        }
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

    // Try to allocate. On failure, return a null handle — the caller
    // (typically a model-specific scheduler) decides which resident
    // chunk to evict and retries.
    std::optional<size_t> off_opt = allocate_(nbytes);
    if (!off_opt.has_value()) {
        return ChunkHandle{};
    }

    Slot slot{*off_opt, nbytes};
    EventHandle evt = nullptr;
    if (host_ptr != nullptr) {
        launch_copy_(slot.offset, host_ptr, nbytes, evt, compute_stream);
        total_h2d_bytes_ += nbytes;
        ++total_h2d_calls_;
    }

    // Step 6: the chunk has been allocated and (in async mode) had its
    // H2D enqueued on copy_stream. The encoder's after_load callback
    // hasn't run yet — that happens in StreamllmRuntime::move_chunk
    // which calls mark_pointer_table_ready when after_load completes.
    const ChunkState initial_state = (host_ptr != nullptr)
        ? ChunkState::H2D_ISSUED
        : ChunkState::SLOT_ALLOCATED;
    residents_.emplace(key, Resident{slot, evt, initial_state});
    if (used_bytes_ > peak_used_bytes_) peak_used_bytes_ = used_bytes_;

    return ChunkHandle{
        (char *)arena_ + slot.offset,
        slot.nbytes,
        evt,
    };
}

std::vector<ChunkHandle> VramChunkPool::load_batch_sync(
    const std::vector<BatchLoadItem> & items,
    const void * packed_host,
    size_t packed_nbytes,
    StreamHandle compute_stream)
{
    std::lock_guard<std::mutex> lk(mu_);
    std::vector<ChunkHandle> out;
    if (items.empty()) return out;
    if (packed_host == nullptr || packed_nbytes == 0) return out;
    if (packed_nbytes > capacity_bytes_) {
        throw std::runtime_error(
            "vram_pool: batch size " + std::to_string(packed_nbytes) +
            " exceeds capacity " + std::to_string(capacity_bytes_));
    }

    for (const auto & item : items) {
        if (item.host_ptr == nullptr || item.nbytes == 0) {
            return {};
        }
        if (residents_.find(item.key) != residents_.end()) {
            unexpected_h2d_for_resident_.fetch_add(
                1, std::memory_order_relaxed);
            return {};
        }
    }

    std::optional<size_t> off_opt = allocate_(packed_nbytes);
    if (!off_opt.has_value()) {
        return {};
    }

    const size_t base_off = *off_opt;
    char * dst = (char *)arena_ + base_off;
    cudaStream_t stream = copy_stream_ == nullptr || force_sync_h2d_
        ? (cudaStream_t)0
        : (cudaStream_t)copy_stream_;

    try {
        if (copy_stream_ != nullptr && !force_sync_h2d_) {
            bool in_capture = false;
            if (compute_stream != nullptr) {
                auto cs = (cudaStream_t) compute_stream;
                cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
                if (cudaStreamIsCapturing(cs, &status) == cudaSuccess &&
                    status == cudaStreamCaptureStatusActive) {
                    in_capture = true;
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
                check_cuda(
                    cudaStreamWaitEvent(stream, (cudaEvent_t)latest_compute_event_, 0),
                    "cudaStreamWaitEvent(compute)");
            }
            check_cuda(
                cudaMemcpyAsync(dst, packed_host, packed_nbytes,
                                cudaMemcpyHostToDevice, stream),
                "cudaMemcpyAsync(H2D batch)");
            check_cuda(cudaStreamSynchronize(stream),
                       "cudaStreamSynchronize(H2D batch)");
        } else {
            check_cuda(
                cudaMemcpy(dst, packed_host, packed_nbytes, cudaMemcpyHostToDevice),
                "cudaMemcpy(H2D batch sync)");
        }
    } catch (...) {
        free_(base_off, packed_nbytes);
        throw;
    }

    out.reserve(items.size());
    size_t cursor = 0;
    for (const auto & item : items) {
        Slot slot{base_off + cursor, item.nbytes};
        residents_.emplace(item.key,
                           Resident{slot, nullptr, ChunkState::H2D_ISSUED});
        out.push_back(ChunkHandle{
            (char *)arena_ + slot.offset,
            slot.nbytes,
            nullptr,
        });
        cursor += item.nbytes;
    }
    total_h2d_bytes_ += packed_nbytes;
    ++total_h2d_calls_;
    batch_h2d_bytes_ += packed_nbytes;
    ++batch_h2d_calls_;
    if (used_bytes_ > peak_used_bytes_) peak_used_bytes_ = used_bytes_;
    return out;
}


void VramChunkPool::reset_stats() {
    std::lock_guard<std::mutex> lk(mu_);
    peak_used_bytes_ = used_bytes_;
    total_h2d_bytes_ = 0;
    total_h2d_calls_ = 0;
    batch_h2d_bytes_ = 0;
    batch_h2d_calls_ = 0;
    batch_fallbacks_ = 0;
    required_chunks_.store(0, std::memory_order_relaxed);
    resident_hits_.store(0, std::memory_order_relaxed);
    load_misses_.store(0, std::memory_order_relaxed);
    prefill_required_chunks_.store(0, std::memory_order_relaxed);
    prefill_resident_hits_.store(0, std::memory_order_relaxed);
    decode_required_chunks_.store(0, std::memory_order_relaxed);
    decode_resident_hits_.store(0, std::memory_order_relaxed);
    redundant_h2d_skipped_.store(0, std::memory_order_relaxed);
    unexpected_h2d_for_resident_.store(0, std::memory_order_relaxed);
}

void VramChunkPool::note_batch_fallback() {
    std::lock_guard<std::mutex> lk(mu_);
    ++batch_fallbacks_;
}

size_t VramChunkPool::required_chunks() const {
    return required_chunks_.load(std::memory_order_relaxed);
}
size_t VramChunkPool::resident_hits() const {
    return resident_hits_.load(std::memory_order_relaxed);
}
size_t VramChunkPool::load_misses() const {
    return load_misses_.load(std::memory_order_relaxed);
}
size_t VramChunkPool::prefill_required_chunks() const {
    return prefill_required_chunks_.load(std::memory_order_relaxed);
}
size_t VramChunkPool::prefill_resident_hits() const {
    return prefill_resident_hits_.load(std::memory_order_relaxed);
}
size_t VramChunkPool::decode_required_chunks() const {
    return decode_required_chunks_.load(std::memory_order_relaxed);
}
size_t VramChunkPool::decode_resident_hits() const {
    return decode_resident_hits_.load(std::memory_order_relaxed);
}
size_t VramChunkPool::redundant_h2d_skipped() const {
    return redundant_h2d_skipped_.load(std::memory_order_relaxed);
}
size_t VramChunkPool::unexpected_h2d_for_resident() const {
    return unexpected_h2d_for_resident_.load(std::memory_order_relaxed);
}

void VramChunkPool::note_required_set(size_t n_required, size_t n_resident) {
    note_required_set_phase(n_required, n_resident, false);
}

void VramChunkPool::note_required_set_phase(size_t n_required,
                                            size_t n_resident,
                                            bool decode_phase) {
    if (n_resident > n_required) n_resident = n_required;  // defensive
    required_chunks_.fetch_add(n_required, std::memory_order_relaxed);
    resident_hits_.fetch_add(n_resident, std::memory_order_relaxed);
    load_misses_.fetch_add(n_required - n_resident,
                            std::memory_order_relaxed);
    if (decode_phase) {
        decode_required_chunks_.fetch_add(n_required, std::memory_order_relaxed);
        decode_resident_hits_.fetch_add(n_resident, std::memory_order_relaxed);
    } else {
        prefill_required_chunks_.fetch_add(n_required, std::memory_order_relaxed);
        prefill_resident_hits_.fetch_add(n_resident, std::memory_order_relaxed);
    }
}

void VramChunkPool::note_redundant_h2d_skipped() {
    redundant_h2d_skipped_.fetch_add(1, std::memory_order_relaxed);
}


void VramChunkPool::evict(const std::string & wid, int cid) {
    std::lock_guard<std::mutex> lk(mu_);
    ChunkKey key{wid, cid};
    auto it = residents_.find(key);
    if (it == residents_.end()) return;

    if (it->second.ready_event) {
        // Recycle rather than destroy — it'll be re-recorded by the next
        // record_h2d on this pool.
        event_free_list_.push_back(it->second.ready_event);
    }
    Slot slot = it->second.slot;
    residents_.erase(it);
    free_(slot.offset, slot.nbytes);
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

ChunkState VramChunkPool::chunk_state(const std::string & wid, int cid) const {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = residents_.find({wid, cid});
    if (it == residents_.end()) return ChunkState::NOT_RESIDENT;
    return it->second.state;
}

bool VramChunkPool::after_load_done(const std::string & wid, int cid) const {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = residents_.find({wid, cid});
    if (it == residents_.end()) return false;
    return it->second.state >= ChunkState::POINTER_TABLE_READY;
}

bool VramChunkPool::kernel_ready_on(const std::string & wid, int cid,
                                     StreamHandle /*stream*/) const {
    // Stream-ordering against ready_event is established by the
    // executor's submit_loads → wait_loads_on(stream) sequence (for
    // load_set) plus the single-compute-stream invariant in Mode A
    // milestone 1 (for keep_set). Host-side check is therefore reduced
    // to "after_load has run" — anything weaker is a bug worth
    // catching here rather than in the M1 kernel trap.  See
    // docs/MODE_A_MILESTONE1.md Step 6 "Host validation API".
    return after_load_done(wid, cid);
}

void VramChunkPool::mark_pointer_table_ready(const std::string & wid, int cid) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = residents_.find({wid, cid});
    if (it == residents_.end()) return;        // raced with eviction
    if (it->second.state < ChunkState::POINTER_TABLE_READY) {
        it->second.state = ChunkState::POINTER_TABLE_READY;
    }
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
    launch_diag::note_stream_wait_event();
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
        // back into the recycle pool while we're confirmed out of capture.
        for (auto h : pending_event_destroys_) {
            event_free_list_.push_back(h);
        }
        pending_event_destroys_.clear();
        event_free_list_.push_back((EventHandle)ev);
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


void VramChunkPool::wait_compute_on(StreamHandle stream) {
    std::lock_guard<std::mutex> lk(mu_);
    if (latest_compute_event_ == nullptr) return;
    if (stream == nullptr) return;
    auto s = (cudaStream_t) stream;
    // Skip under capture for the same reason record_compute_event does
    // (the event is captured-graph-scoped from outside the capture and
    // waiting on it here would tie an uncaptured launch to a captured
    // event handle).
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(s, &status) == cudaSuccess &&
        status == cudaStreamCaptureStatusActive) {
        return;
    }
    check_cuda(
        cudaStreamWaitEvent(s, (cudaEvent_t) latest_compute_event_, 0),
        "cudaStreamWaitEvent(compute, evict)");
}

} // namespace streamllm_ext
