// streamllm-ext — runtime singleton. Owns a VRAM pool + a Scheduler.
// Core primitive:
//
//   move_chunk(wid, cid, src, dst) -> event       // async tier move
//
// Chunk vocabulary (a "chunk" is the smallest unit the pool moves):
//
//   Q_BIAS  = 0                   — shared fixed-meta chunk (per tensor)
//   PLANE_i = kCidChunkBase + i   — plane i's payload
//
// Chunk computation (chunk_matmul) lives in the encoder layer, not
// here — see ``decoder/ss_anybcq/chunked_matmul.h`` for the
// per-tensor path and ``qwen3/fused_kernels.h`` for the fused MoE GEMV.
// Core only exposes the layout + pool primitives the encoder layer
// reads.

#pragma once

#include "computation.h"
#include "scheduler.h"
#include "stream_reader.h"
#include "upstream_layout.h"
#include "vram_pool.h"

#include <cuda_runtime.h>

// Decode hot-path scratch — fwd-declared so this header doesn't pull in
// the AnyBCQ kernel header. Defined in decoder/anybcq/naver_gemv.h.
namespace streamllm_ext { struct NaverKernelScratch; }
// AnyBCQFamilyTensor — concrete ChunkedTensor that owns each managed
// tensor's host layout + per-plane device pointer tables. Forward-
// declared so the runtime header doesn't drag the decoder include in;
// runtime.cpp pulls in decoder/anybcq/tensor.h directly.
namespace streamllm_ext {
class ChunkedTensor;
namespace anybcq { class AnyBCQFamilyTensor; }
}  // namespace streamllm_ext

#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Framework-level cap on how many data chunks a managed tensor can have.
// Tied to the NAVER kernel's per-call precision limit via static_assert
// so the two layers stay in sync without either one taking a dependency
// on the other's public surface.

namespace streamllm_ext {

// Cid encoding. See header comment.
constexpr int kCidQBias      = 0;
constexpr int kCidChunkBase  = 100;
constexpr int kMaxChunksPerTensor = 16;
// The decoder TU (core/runtime.cpp) static_asserts that this matches
// the AnyBCQ kernel's per-launch precision cap.

inline int  cid_chunk(int i)        { return kCidChunkBase + i; }
inline bool cid_is_chunk(int cid)   {
    return cid >= kCidChunkBase && cid < kCidChunkBase + kMaxChunksPerTensor;
}
inline int  cid_chunk_index(int cid) {
    return cid_is_chunk(cid) ? cid - kCidChunkBase : -1;
}

// Device-side view of one managed tensor. ``chunk_ptrs[i]`` points at
// chunk i's packed blob on device when resident (signs first, then α
// at offset qw_bytes_per_chunk). ``q_bias_fp16`` points at the shared
// q_bias chunk (cid = kCidQBias).
struct UpstreamLayoutDevice {
    const void * chunk_ptrs[kMaxChunksPerTensor] = {};
    const void * q_bias_fp16 = nullptr;
    int          M           = 0;  // n_out
    int          K           = 0;  // padded_m (in)
    int          n_chunks    = 0;  // number of data chunks per tensor
    int          K_groups    = 0;
    int          group_size  = 0;
    size_t       qw_bytes_per_chunk = 0;  // offset to α inside a chunk
    size_t       alpha_bytes_per_chunk = 0;  // per-plane α stride (any-prec
                                              // refresh_alpha kernel uses
                                              // this to walk α inside the
                                              // top resident chunk)
    // Any-prec: tells the dispatch layer to skip the host-side
    // ``chunk_ptrs[p]`` walk in ``build_plane_ptrs`` and go directly
    // through the device-side per-plane pointer table (which carries
    // the correct plane→signs/α mapping populated by
    // update_anyprec_after_load_async). For ss_anybcq wids both layouts
    // coincide so the flag is false.
    bool         any_precision = false;
    int          base_precision = 0;  // chunk-0 owns this many planes
};


class StreamllmRuntime {
public:
    StreamllmRuntime(size_t capacity_bytes, int device, bool copy_stream,
                     const char * scheduler_name);
    ~StreamllmRuntime();

    StreamllmRuntime(const StreamllmRuntime &) = delete;
    StreamllmRuntime & operator=(const StreamllmRuntime &) = delete;

    void install(const StreamReader & reader,
                 const std::string & gguf_path);

    // (P2★) ``is_managed_name`` removed — scheduler-policy concern,
    // moved to ``Scheduler::claims_tensor`` (consult that directly).

    // --- primitives -----------------------------------------------------

    // ``compute_stream`` is only consulted when CUDA graphs are active:
    // if it's in an active capture, the pool's copy stream is forked
    // into the same capture session so the kernel's downstream wait on
    // the H2D ready event is a legal intra-capture dependency. Safe to
    // pass nullptr (install-time bulk uploads, outside any capture).
    EventHandle move_chunk(const std::string & wid, int cid,
                           Tier src, Tier dst,
                           StreamHandle compute_stream = nullptr);

    // (chunk_matmul / chunk_matmul_batched moved out of core. Callers
    // now invoke the encoder-side helpers in
    // ``decoder/ss_anybcq/chunked_matmul.h`` directly. Core only
    // exposes the layout + pool primitives those helpers need.)

    // --- accessors ------------------------------------------------------

    const UpstreamLayoutDevice * layout(const std::string & wid) const;

    const VramChunkPool & pool() const { return *pool_; }
    VramChunkPool & pool() { return *pool_; }
    const Scheduler & scheduler() const { return *scheduler_; }
    Scheduler & scheduler() { return *scheduler_; }

    // Decode-path scratch the encoder-side ``chunk_matmul`` reuses to
    // avoid per-call cudaMallocAsync. Lifetime tied to the runtime.
    NaverKernelScratch * gemv_scratch() const { return gemv_scratch_.get(); }

    // Lifecycle helpers for schedulers.
    // Drop the host-side chunk byte buffers for a registered tensor.
    // Safe iff the scheduler has uploaded the chunks to VRAM and will
    // never need to re-upload (i.e. it pins them, or never evicts).
    // Reduces install-time host RAM from O(total managed bytes) to
    // O(one tensor at a time) — critical at MoE scale where tracking
    // 18k experts' chunk bytes consumes ~46 GB of heap on Qwen3-30B-A3B.
    void release_host_bytes(const std::string & wid);

    void register_layout(const std::string & wid,
                         UpstreamLayoutHost host,
                         UpstreamLayoutDevice dev);

    // Per-tensor managed-tensor accessor.  Returns the abstract
    // ChunkedTensor base — encoder/architecture-agnostic.  Decoder-
    // private state (per-plane device pointer tables, host layout,
    // β buffer, etc.) is reachable only via downcast in the decoder
    // layer (see decoder/anybcq/tensor.h). Returns null if the wid
    // is unknown.  Replaces the previous encoder-named accessors
    // (anybcq_d_qw_ptrs / _alpha_ptrs / _qbias_slot) that mixed
    // AnyBCQ specifics into core's surface.
    ChunkedTensor * tensor(const std::string & wid) const;
    // Typed convenience accessor for the AnyBCQ family. Returns the
    // concrete subclass pointer when the managed tensor was parsed
    // by the AnyBCQ / SsAnybcq-AnyBCQ encoder; null otherwise.
    // Decoder-side callers (qwen3 dispatch glue, ss_anybcq chunked-
    // matmul wrappers) use this to reach d_qw_ptrs() / d_alpha_ptrs()
    // / d_qbias_slot() without dragging encoder names into core.
    anybcq::AnyBCQFamilyTensor * tensor_anybcq(const std::string & wid) const;

    // Clear the entry's device-side per-plane pointer slot for cid.
    // Must be called whenever a chunk is evicted from the VRAM pool —
    // pool::evict() frees the slot but does not touch the per-Entry
    // pointer arrays the fused MoE kernel reads, so without this call
    // the kernel keeps using the freed pointer (the slot may have been
    // re-allocated to a different chunk's load), silently corrupting
    // outputs.  ``stream`` is the pool's copy stream (mid-run) so the
    // clear runs async and overlaps with subsequent loads, or
    // ``nullptr`` (teardown).  q_bias and unknown cids are silently
    // ignored.
    void clear_chunk_device_ptr(const std::string & wid, int cid,
                                StreamHandle stream = nullptr);

    // Drop a single (wid, cid) chunk's host bytes synchronously. Called
    // by the scheduler under its DRAM-cache LRU mutex when evicting to
    // stay under STREAMLLM_HOST_CAP_MB. Acquires the per-Entry host
    // mutex so it serialises against any in-flight move_chunk reads/
    // writes for host.chunks[p]. q_bias is never evictable; cid is
    // silently ignored if it doesn't map to a chunk plane.
    void release_chunk_host(const std::string & wid, int cid);

    // Is the (wid, cid) chunk currently resident in host DRAM (i.e.
    // host.chunks[p] populated)? Used by the hook to attribute cache
    // misses to DRAM-hit vs SSD-miss before issuing the move. Lock-free
    // probe — caller tolerates a benign race against a concurrent
    // eviction (the bucket count is approximate, not load-bearing for
    // correctness).
    bool host_resident(const std::string & wid, int cid) const;

    // Bump-allocate ``bytes`` from the install-time small-slab. Used
    // by callers that need many tiny device allocations (per-expert
    // pointer arrays, per-canonical MoeExpertTable internals); a
    // single big cudaMalloc on the slab + slicing here avoids
    // 18k+ separate cudaMallocs at install. Returns nullptr if the
    // slab is exhausted. 8-byte alignment.
    void * small_alloc(size_t bytes);

    // Stash per-chunk file offsets so move_chunk can pread chunk bytes
    // directly when host.chunks[p] is empty (SSD-streaming path).
    // Pass alongside register_layout — each entry in chunk_file_offsets
    // matches host.chunks[i]'s position; chunk_file_sizes[i] is the
    // number of bytes for that plane in the GGUF.
    void register_chunk_io(const std::string & wid,
                           std::vector<int64_t> chunk_file_offsets,
                           std::vector<int64_t> chunk_file_sizes);

private:
    struct Entry {
        // Concrete ChunkedTensor (AnyBCQTensor / SsAnybcqTensor) owning
        // this tensor's host layout + per-plane device pointer tables.
        // Allocated by register_layout from the parsed UpstreamLayoutHost
        // (slab-allocated pointer arrays bound via set_device_state).
        std::unique_ptr<anybcq::AnyBCQFamilyTensor> tensor;
        UpstreamLayoutDevice dev;
        // Per-Entry mutex guarding tensor->host().chunks reads/writes.
        // Owned via unique_ptr because std::mutex isn't move-constructible
        // but unordered_map::emplace requires Entry to be movable. Hot
        // path takes only this mutex — concurrent move_chunks on
        // different wids never contend (vs. a global host_data_mu_
        // that bottlenecks at the memcpy bandwidth ceiling at high
        // worker counts).
        std::unique_ptr<std::mutex> host_mu = std::make_unique<std::mutex>();
        // Per-chunk file offsets / sizes in the GGUF. Populated by
        // register_chunk_io. When host.chunks[p] is empty, move_chunk
        // preads chunk_file_sizes[p] bytes from chunk_file_offsets[p]
        // in gguf_path_, uploads to VRAM, and frees the temp buffer.
        // Empty for entries where the scheduler kept all host bytes
        // and no streaming is required.
        std::vector<int64_t> chunk_file_offsets;
        std::vector<int64_t> chunk_file_sizes;
    };

    std::unique_ptr<VramChunkPool> pool_;
    std::unique_ptr<Scheduler>     scheduler_;
    std::unordered_map<std::string, Entry> entries_;
    std::unique_ptr<NaverKernelScratch> gemv_scratch_;  // decode hot-path scratch
    std::string                    gguf_path_;     // stashed at install for SSD-streaming preads

    // Cached SSD-streaming infrastructure. Open the GGUF fd once at
    // install (avoid open/close-per-chunk syscall overhead), keep a
    // small pinned-host ring as the pread destination so the
    // subsequent cudaMemcpyAsync runs as a true async DMA without an
    // internal pageable→pinned staging copy.
    int                            gguf_fd_ = -1;       // cached fd, owned

    // Single-allocation slab for per-tensor pointer arrays.
    // ``register_layout`` slices ``2 × kMaxChunksPerTensor × void*`` out
    // of this for each managed tensor's d_chunk_qw_ptrs /
    // d_chunk_alpha_ptrs. Replaces 2 × n_tensors cudaMallocs (which
    // cost ~3 ms each at scale and added 4–5 min cold-launch latency
    // on Qwen3-30B-A3B's 18,432 experts).
    void *                         small_slab_     = nullptr;  // device
    size_t                         small_slab_bytes_ = 0;
    size_t                         small_slab_used_  = 0;
    std::mutex                     small_slab_mu_;

    void *                         host_ring_ = nullptr;
    std::vector<void *>            host_ring_slot_ptrs_; // one ptr per slot
    std::vector<EventHandle>       host_ring_events_;    // event from last use
    size_t                         host_ring_slot_bytes_ = 0;

    // Atomic so multiple async pread worker threads can fetch_add their
    // own slot indices without locks. Slots are large enough (32) that
    // wraparound-induced contention is rare under typical hook traffic.
    std::atomic<uint32_t>          host_ring_next_{0};

    // Async chunk-load worker pool. The MoE dispatch path can fan
    // out hundreds of pread+H2D calls per hook invocation;
    // serialising them on the hook thread caps wall-clock at QD1
    // NVMe latency (~200 us/call). Workers consume from a bounded
    // queue and call move_chunk in parallel; the hook fans out then
    // joins via wait_async_load_idle() before returning.
    //
    // (Despite occasional "prefetch" naming in older code, this is
    // demand-loading — chunks load when needed, not ahead of need.
    // Real graph-prewalk-driven prefetch is a separate future task.)
    struct AsyncLoadRequest {
        std::string  wid;
        int          cid;
        // Optional per-batch counter for fine-grained (per-expert)
        // host-side waits. nullptr means the request is only counted
        // by the global ``io_in_flight_`` counter that
        // ``wait_async_load_idle`` polls.
        std::shared_ptr<std::atomic<uint32_t>> batch_remaining;
    };
    std::vector<std::thread>           io_workers_;
    std::mutex                         io_mu_;
    std::condition_variable            io_cv_work_;
    std::condition_variable            io_cv_done_;
    std::deque<AsyncLoadRequest>       io_queue_;
    std::atomic<uint32_t>              io_in_flight_{0};
    std::atomic<bool>                  io_stop_{false};
    // Serializes copy_stream emissions across workers. ``pool->load``
    // already holds ``pool::mu_`` around its ``cudaMemcpyAsync`` +
    // per-chunk ``cudaEventRecord``, but ``move_chunk``'s post-load
    // ``after_load`` kernel + the per-chunk re-record happen OUTSIDE
    // that lock. With concurrent workers a racing emission can let the
    // trailing batch ready_event land on copy_stream BEFORE another
    // worker's ``after_load`` — so the captured kernel reads still-
    // stale per-plane pointers. ``io_stream_mu_`` covers the entire
    // worker-thread emission window so copy_stream's queued order is
    // strictly batch-FIFO and ``batch_ready_event`` is the last node.
    std::mutex                         io_stream_mu_;

    void io_worker_loop_();

public:
    // Submit a chunk for async load. Fire-and-forget within a hook
    // call; pair with wait_async_load_idle() (or wait_async_load_batch
    // on the per-expert handle) before reading the chunk's slot.
    // Returns false if no worker pool is active.
    bool submit_async_load(const std::string & wid, int cid);
    // Variant that increments a caller-owned counter; the worker
    // decrements it after move_chunk returns. Pair with
    // wait_async_load_batch to wait on just this batch (= one
    // expert's chunks) instead of the whole hook fire.
    bool submit_async_load(const std::string & wid, int cid,
                           std::shared_ptr<std::atomic<uint32_t>> batch_remaining);
    // Block until every submitted async-load has finished.
    void wait_async_load_idle();
    // Block until ``remaining`` reaches zero — the caller increments
    // it once per submit_async_load and the worker decrements after
    // each move_chunk completes.
    void wait_async_load_batch(
        const std::shared_ptr<std::atomic<uint32_t>> & remaining);
    // Number of async-load worker threads (0 = disabled).
    int  io_worker_count() const { return (int)io_workers_.size(); }

    // Score-table snapshot taken once per cgraph_compute by the
    // scheduler's ``on_graph_compute_begin``. Computations that
    // depend on the live dial read this snapshot during their plan()
    // so the dial value is consistent across all managed dispatches
    // in the same compute pass.
    void                       set_replay_score_table(std::vector<float> snap);
    const std::vector<float> & current_replay_score_table() const;

    // Score-table version key, refreshed alongside the snapshot.
    // Read by ggml-cuda's CUDA-graph cache predicate to force re-
    // capture when the dial changes (e.g. HTTP /streamllm/score_table
    // swap or a phase-aware reasoning→generation transition). Atomic
    // so the cache hook can read it lock-free per cgraph_compute.
    void     set_replay_score_table_version(uint64_t v);
    uint64_t replay_score_table_version() const;

private:
    bool installed_ = false;

    // Score-table snapshot, refreshed by the scheduler in
    // on_graph_compute_begin.
    std::vector<float>                       replay_score_table_;
    mutable std::mutex                       replay_score_table_mu_;
    std::atomic<uint64_t>                    replay_score_table_version_{0};
};

} // namespace streamllm_ext
