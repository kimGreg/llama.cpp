// streamllm-ext — runtime implementation. Thin facade over
// VramChunkPool + Scheduler, exposing the two primitives.

#include "runtime.h"
#include "decoder_registry.h"
#include "pointer_patch.h"
#include "runtime_diag.h"
#include "streamllm_nvtx.h"

// Encoder-agnostic UpstreamLayoutHost struct + dispatcher entry point.
// Includes the function-pointer typedefs the encoder registers callbacks
// against (disk_to_kernel_fn / after_load_fn / after_evict_fn). Core's
// move_chunk invokes them blindly — no decoder includes needed.
#include "upstream_layout.h"
#include "anybcq_gemv.h"         // NaverKernelScratch (decode-path scratch
                                  // owned by the runtime; encoder-side
                                  // helpers reuse it via gemv_scratch()).
#include "decoder/anybcq/tensor.h"  // complete type for tensor_anybcq()
namespace streamllm_ext {
// Keep the framework's chunks-per-tensor cap aligned with the AnyBCQ
// kernel's per-launch precision cap. The two layers must move together;
// asserted in the decoder TU so core/runtime.h stays decoder-agnostic.
static_assert(kMaxChunksPerTensor == kNaverMaxPrecision,
              "framework / kernel chunk cap mismatch");
} // namespace streamllm_ext

#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

namespace streamllm_ext {
// Provided by runtime_glue.cpp; called when STREAMLLM_PROFILE=1 is set
// to attribute SSD-stream H2D wall time between pread and pool->load.
void profile_record_move_chunk_ssd(uint64_t pread_ns, uint64_t pool_load_ns);
// Bumped each time pool.load returned null and we're about to ask the
// scheduler to make room — measures pool-pressure under STREAMLLM_PROFILE=1.
void profile_record_pool_make_room();

// Process-wide host-DRAM cache size (sum of host.chunks[p].size() over
// every Entry). Updated atomically on populate / release; exposed via
// streamllm_stat_host_dram_bytes() for /streamllm/stats.
std::atomic<uint64_t> g_host_dram_bytes{0};
}  // namespace streamllm_ext

extern "C" unsigned long long streamllm_stat_host_dram_bytes(void) {
    return (unsigned long long)
        streamllm_ext::g_host_dram_bytes.load(std::memory_order_relaxed);
}

namespace streamllm_ext {

namespace {
// Scoped move_chunk timer. Records (now - t0) into the named diag
// bucket on destruction. Compiles to a no-op when STREAMLLM_DIAG is
// off (no chrono calls, no atomic adds).
struct DiagSpan {
#ifdef STREAMLLM_DIAG
    diag::MoveSite site;
    std::chrono::steady_clock::time_point t0;
    explicit DiagSpan(diag::MoveSite s)
        : site(s), t0(std::chrono::steady_clock::now()) {}
    ~DiagSpan() {
        const uint64_t ns = (uint64_t)std::chrono::duration_cast<
            std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - t0).count();
        diag::record_move_ns(site, ns);
    }
#else
    explicit DiagSpan(diag::MoveSite) {}
#endif
};

bool host_chunk_cache_enabled() {
    if (const char * s = std::getenv("STREAMLLM_HOST_CACHE")) {
        return !(s[0] == '0');
    }
    if (const char * s = std::getenv("STREAMLLM_DISABLE_HOST_CACHE")) {
        return !(s[0] && s[0] != '0');
    }
    return true;
}
} // anon

const char * tier_name(Tier t) {
    switch (t) {
        case Tier::SSD:  return "ssd";
        case Tier::RAM:  return "ram";
        case Tier::VRAM: return "vram";
    }
    return "?";
}

StreamllmRuntime::StreamllmRuntime(size_t capacity_bytes, int device,
                                   bool copy_stream,
                                   const char * scheduler_name)
    : pool_(std::make_unique<VramChunkPool>(capacity_bytes, device, copy_stream))
    , scheduler_(make_scheduler(scheduler_name))
    , gemv_scratch_(std::make_unique<NaverKernelScratch>()) {
    // Pre-allocate decode kernel scratch. Initial acc_f32 capacity sized
    // to 1 M elems (4 MiB) — covers every Qwen3 M; grows on demand for
    // larger lm_head / vocab matmuls.
    gemv_scratch_->init(device, /*acc_capacity_elems=*/1 << 20);
}

StreamllmRuntime::~StreamllmRuntime() {
    // Stop the prefetch worker pool first — workers may still be
    // dereferencing entries_ or pool_ when this dtor runs.
    if (!io_workers_.empty()) {
        {
            std::lock_guard<std::mutex> lk(io_mu_);
            io_stop_.store(true, std::memory_order_relaxed);
        }
        io_cv_work_.notify_all();
        for (auto & t : io_workers_) {
            if (t.joinable()) t.join();
        }
        io_workers_.clear();
    }
    // Per-tensor pointer arrays were carved out of small_slab_; one
    // cudaFree handles them all. Per-entry pointers become invalid.
    if (small_slab_) {
        cudaFree(small_slab_);
        small_slab_ = nullptr;
        small_slab_bytes_ = 0;
        small_slab_used_  = 0;
    }
    if (gemv_scratch_) gemv_scratch_->destroy();
    // Tear down SSD-streaming infrastructure.
    for (auto ev : host_ring_events_) {
        if (ev) cudaEventDestroy((cudaEvent_t)ev);
    }
    host_ring_events_.clear();
    host_ring_slot_ptrs_.clear();
    if (host_ring_) {
        cudaFreeHost(host_ring_);
        host_ring_ = nullptr;
    }
    if (batch_staging_) {
        cudaFreeHost(batch_staging_);
        batch_staging_ = nullptr;
        batch_staging_bytes_ = 0;
    }
    if (gguf_fd_ >= 0) {
        ::close(gguf_fd_);
        gguf_fd_ = -1;
    }
}

void StreamllmRuntime::install(const StreamReader & reader,
                               const std::string & gguf_path) {
    if (installed_) {
        throw std::runtime_error(
            "StreamllmRuntime: already installed; each instance handles one GGUF");
    }
    // (P2★) managed_names_ snapshot moved to the scheduler.
    gguf_path_ = gguf_path;

    // Open the GGUF once for SSD-streaming preads. fd lives until the
    // runtime is destroyed — avoids an open/close pair per chunk
    // (~4000 promotions per token at 18k experts is significant).
    gguf_fd_ = ::open(gguf_path.c_str(), O_RDONLY | O_CLOEXEC);
    if (gguf_fd_ < 0) {
        throw std::runtime_error(
            "StreamllmRuntime::install: cannot open " + gguf_path +
            ": " + std::strerror(errno));
    }

    // Pinned-host ring for SSD-stream pread destinations. Sized so
    // the cudaMemcpyAsync from pinned host memory is a true async DMA
    // (pageable would force an internal staging copy that effectively
    // synchronises wrt the host). 32 slots × largest chunk ≈ ~10 MB
    // pinned — tractable on any consumer GPU. The slot count is more
    // than n_used_per_tok so the ring rarely waits on a previous H2D.
    {
        size_t slot_bytes = 0;
        for (const auto & name : reader.managed_tensor_names()) {
            const TensorLayout * L = reader.layout(name);
            if (L == nullptr) continue;  // placeholder-only canonical
            for (uint32_t cb : L->chunk_bytes) {
                if ((size_t)cb > slot_bytes) slot_bytes = (size_t)cb;
            }
        }
        if (slot_bytes > 0) {
            // Pinned host-buffer ring. 32 slots × largest chunk ≈ 10 MB
            // pinned — comfortably above the 8-worker peak concurrency
            // (one slot reservation per worker at a time under the
            // single-call move_chunk path).
            const int n_slots = 32;
            host_ring_slot_bytes_ = slot_bytes;
            cudaError_t cerr = cudaHostAlloc(
                &host_ring_, slot_bytes * (size_t)n_slots,
                cudaHostAllocDefault);
            if (cerr != cudaSuccess) {
                throw std::runtime_error(
                    std::string("StreamllmRuntime::install: cudaHostAlloc(host_ring) failed: ") +
                    cudaGetErrorString(cerr));
            }
            host_ring_slot_ptrs_.resize(n_slots);
            host_ring_events_.assign(n_slots, nullptr);
            for (int i = 0; i < n_slots; ++i) {
                host_ring_slot_ptrs_[i] =
                    (char *)host_ring_ + (size_t)i * slot_bytes;
            }
            host_ring_next_.store(0, std::memory_order_relaxed);
        }
    }

    // Pre-allocate the small-slab for many tiny device buffers
    // populated at install time:
    //   - per-tensor d_chunk_qw_ptrs / d_chunk_alpha_ptrs arrays
    //     (2 × kMaxChunksPerTensor × void* per managed tensor)
    //   - per-canonical MoeExpertTable internals
    //     (3 arrays × n_experts × void* per canonical MoE tensor)
    //   - small headroom for ad-hoc small_alloc callers
    // One cudaMalloc + bump allocator slicing here avoids 36k+ tiny
    // cudaMallocs at install; cold launch goes from 4-5 min to ~30 s.
    {
        const size_t per_tensor_bytes =
            (size_t)kMaxChunksPerTensor * sizeof(void *) * 2 +
            sizeof(void *);  // +1 slot for any-prec d_qbias_slot
        const size_t n_tensors = reader.managed_tensor_names().size();
        // Headroom for MoeExpertTable: assume ≤ 256 canonicals × 3
        // arrays × ≤ 1024 experts × 8 bytes = ~6 MB. Round to 16 MB
        // for safety + future small_alloc consumers.
        const size_t slab_bytes = n_tensors * per_tensor_bytes
                                  + 16 * 1024 * 1024;
        if (slab_bytes > 0) {
            cudaError_t cerr = cudaMalloc(&small_slab_, slab_bytes);
            if (cerr != cudaSuccess) {
                throw std::runtime_error(
                    std::string("StreamllmRuntime::install: cudaMalloc(small_slab) failed: ") +
                    cudaGetErrorString(cerr));
            }
            cudaMemset(small_slab_, 0, slab_bytes);  // zero-init
            small_slab_bytes_ = slab_bytes;
            small_slab_used_  = 0;
        }
    }

    scheduler_->on_install(*this, reader, gguf_path);

    // Spin up the async-prefetch worker pool. Default 8 workers — at
    // QD1 SSD pread latency of ~200 us/call, parallel preads scale
    // sub-linearly until they hit core-count or NVMe queue-depth
    // limits. On the dev box (8-core Ryzen, single Gen4 NVMe) N=8
    // gave +47% tg32 vs sync; N=16 only +53%, with double the CPU
    // pressure. Pick N=8 as the default sweet spot. Override via
    // STREAMLLM_IO_WORKERS=N (0 disables and falls back to
    // inline pread on the hook thread).
    {
        int n_workers = 8;
        if (const char * w = getenv("STREAMLLM_IO_WORKERS")) {
            n_workers = std::atoi(w);
            if (n_workers < 0) n_workers = 0;
            if (n_workers > 32) n_workers = 32;
        }
        io_stop_.store(false, std::memory_order_relaxed);
        io_workers_.reserve(n_workers);
        for (int i = 0; i < n_workers; ++i) {
            io_workers_.emplace_back(
                &StreamllmRuntime::io_worker_loop_, this);
        }
    }

    installed_ = true;
}

void StreamllmRuntime::register_layout(const std::string & wid,
                                        UpstreamLayoutHost host,
                                        UpstreamLayoutDevice dev) {
    auto [it, _] = entries_.emplace(wid, Entry{});
    Entry & e = it->second;

    // Account install-time host bytes so the matching release_host_bytes
    // doesn't underflow g_host_dram_bytes.
    {
        size_t install_bytes = 0;
        for (const auto & c : host.chunks) install_bytes += c.size();
        if (install_bytes) g_host_dram_bytes.fetch_add(
            install_bytes, std::memory_order_relaxed);
    }

    e.tensor = make_chunked_tensor(wid, std::move(host));
    e.dev    = dev;

    // Populate the dispatch-side layout fields up-front from the host
    // layout.  These are static metadata (M, K, group_size, etc.) known
    // at install time — they don't depend on any chunk being resident.
    // ``move_chunk`` later re-writes them on every call, but the FIRST
    // hook firing happens BEFORE any move_chunk has run for this wid in
    // dynamic-streaming mode.  Without this priming the hook reads
    // group_size=0, bails to false, and upstream's mul_mat_id reads the
    // empty placeholder → CUDA illegal-access on the next op.  Pointers
    // (chunk_ptrs[], q_bias_fp16) stay null; the hook paths that need
    // them check separately.
    {
        UpstreamLayoutDevice & d = e.dev;
        const UpstreamLayoutHost & h = e.tensor->host();
        d.M                  = h.n;
        d.K                  = h.padded_m;
        d.n_chunks           = h.n_chunks;
        d.K_groups           = h.K_groups;
        d.group_size         = h.group_size;
        d.qw_bytes_per_chunk    = h.qw_bytes_per_chunk;
        d.alpha_bytes_per_chunk = h.alpha_bytes_per_chunk;
        d.any_precision         = h.any_precision;
        d.base_precision        = h.base_precision;
    }

    // Slice the per-tensor pointer arrays out of the pre-allocated
    // small_slab_ (one cudaMalloc for the whole runtime, instead of
    // 36k cudaMallocs for 18k experts). The slab was zero-init'd at
    // install — the kernel's prefix-plan fast path distinguishes
    // null vs non-null per slot, so initial zero state is correct.
    void ** d_qw     = nullptr;
    void ** d_alpha  = nullptr;
    void ** d_qbias  = nullptr;
    {
        std::lock_guard<std::mutex> lk(small_slab_mu_);
        const size_t per_array = (size_t)kMaxChunksPerTensor * sizeof(void *);
        // 2 × per_array (qw + α tables) + 1 × sizeof(void*) (qbias slot)
        const size_t slot_bytes = sizeof(void *);
        if (small_slab_ == nullptr ||
            small_slab_used_ + 2 * per_array + slot_bytes > small_slab_bytes_) {
            throw std::runtime_error(
                "StreamllmRuntime::register_layout: small_slab exhausted");
        }
        d_qw    = (void **)((char *)small_slab_ + small_slab_used_);
        small_slab_used_ += per_array;
        d_alpha = (void **)((char *)small_slab_ + small_slab_used_);
        small_slab_used_ += per_array;
        d_qbias = (void **)((char *)small_slab_ + small_slab_used_);
        small_slab_used_ += slot_bytes;
    }
    e.tensor->set_device_state(d_qw, d_alpha, d_qbias);
}

void StreamllmRuntime::release_host_bytes(const std::string & wid) {
    auto it = entries_.find(wid);
    if (it == entries_.end()) return;
    UpstreamLayoutHost & h = it->second.tensor->host();
    // Free the per-chunk byte buffers — these are large (~2 MB/expert)
    // and re-readable from disk via register_chunk_io's offset map.
    // KEEP q_bias: it's small (~96 KB/expert), derived from the FIXED_META
    // section, and needed at every chunk_matmul call. Re-computing it
    // would require parsing FIXED_META on every move_chunk, which is
    // wasteful for a tensor that fits comfortably in heap.
    //
    // Hold the per-Entry host mutex: a concurrent move_chunk on
    // another thread may be reading host.chunks[p].
    std::lock_guard<std::mutex> lk(*it->second.host_mu);
    size_t freed = 0;
    for (auto & c : h.chunks) {
        freed += c.size();
        std::vector<uint8_t>().swap(c);
    }
    std::vector<std::vector<uint8_t>>().swap(h.chunks);
    if (freed) g_host_dram_bytes.fetch_sub(freed, std::memory_order_relaxed);
    // h.q_bias intentionally retained.
}

void StreamllmRuntime::release_chunk_host(const std::string & wid, int cid) {
    auto it = entries_.find(wid);
    if (it == entries_.end()) return;
    if (!cid_is_chunk(cid)) return;          // q_bias / unknown
    int p = cid_chunk_index(cid);
    if (p < 0) return;
    std::lock_guard<std::mutex> lk(*it->second.host_mu);
    auto & chunks = it->second.tensor->host().chunks;
    if (p < (int)chunks.size() && !chunks[p].empty()) {
        const size_t freed = chunks[p].size();
        std::vector<uint8_t>().swap(chunks[p]);
        g_host_dram_bytes.fetch_sub(freed, std::memory_order_relaxed);
    }
}

bool StreamllmRuntime::host_resident(const std::string & wid, int cid) const {
    auto it = entries_.find(wid);
    if (it == entries_.end()) return false;
    const UpstreamLayoutHost & host = it->second.tensor->host();
    if (cid == kCidQBias) return !host.q_bias.empty();
    if (!cid_is_chunk(cid)) return false;
    int p = cid_chunk_index(cid);
    if (p < 0) return false;
    const auto & chunks = host.chunks;
    return p < (int)chunks.size() && !chunks[p].empty();
}

void StreamllmRuntime::register_chunk_io(
        const std::string & wid,
        std::vector<int64_t> chunk_file_offsets,
        std::vector<int64_t> chunk_file_sizes) {
    std::vector<uint8_t> kernel_ready(chunk_file_offsets.size(), 0);
    std::vector<std::string> bundle(chunk_file_offsets.size());
    register_chunk_io(wid, std::move(chunk_file_offsets),
                      std::move(chunk_file_sizes),
                      std::move(kernel_ready), std::move(bundle));
}

void StreamllmRuntime::register_chunk_io(
        const std::string & wid,
        std::vector<int64_t> chunk_file_offsets,
        std::vector<int64_t> chunk_file_sizes,
        std::vector<uint8_t> chunk_file_kernel_ready,
        std::vector<std::string> chunk_file_bundle) {
    auto it = entries_.find(wid);
    if (it == entries_.end()) {
        throw std::runtime_error(
            "StreamllmRuntime::register_chunk_io: unknown wid " + wid);
    }
    if (chunk_file_offsets.size() != chunk_file_sizes.size() ||
        chunk_file_offsets.size() != chunk_file_kernel_ready.size() ||
        chunk_file_offsets.size() != chunk_file_bundle.size()) {
        throw std::runtime_error(
            "StreamllmRuntime::register_chunk_io: source vector size mismatch for " +
            wid);
    }
    it->second.chunk_file_offsets = std::move(chunk_file_offsets);
    it->second.chunk_file_sizes   = std::move(chunk_file_sizes);
    it->second.chunk_file_kernel_ready =
        std::move(chunk_file_kernel_ready);
    it->second.chunk_file_bundle = std::move(chunk_file_bundle);
}

const UpstreamLayoutDevice * StreamllmRuntime::layout(const std::string & wid) const {
    auto it = entries_.find(wid);
    if (it == entries_.end()) return nullptr;
    return &it->second.dev;
}

ChunkedTensor * StreamllmRuntime::tensor(const std::string & wid) const {
    auto it = entries_.find(wid);
    if (it == entries_.end()) return nullptr;
    return it->second.tensor.get();
}

anybcq::AnyBCQFamilyTensor *
StreamllmRuntime::tensor_anybcq(const std::string & wid) const {
    auto it = entries_.find(wid);
    if (it == entries_.end()) return nullptr;
    return dynamic_cast<anybcq::AnyBCQFamilyTensor *>(it->second.tensor.get());
}

void StreamllmRuntime::clear_chunk_device_ptr(const std::string & wid, int cid,
                                              StreamHandle stream) {
    if (!cid_is_chunk(cid)) return;
    auto it = entries_.find(wid);
    if (it == entries_.end()) return;
    int p = cid_chunk_index(cid);

    if (defer_pointer_clears_) {
        if (deferred_clear_stream_ == nullptr) deferred_clear_stream_ = stream;
        if (stream == deferred_clear_stream_) {
            if (it->second.tensor->append_after_evict_patch(
                    p, deferred_clear_patches_)) {
                return;
            }
        }
    }

    // CROSS-STREAM EVICTION SAFETY.
    //
    // ``after_evict`` writes nulls into the per-plane d_qw / d_alpha
    // pointer table (device memory).  The fused MoE kernel reads the
    // same memory on the compute stream.  Without a wait, the
    // null-clear on copy_stream can land mid-kernel and turn a
    // resident-but-no-longer-reserved chunk's plane pointer into
    // null — silent corruption if the kernel had already read the
    // pre-null value (slot still valid) OR a kernel trap if it
    // reads post-null (we'd see violations, but at varying-
    // precision dial points the kernel may have moved past that
    // plane index already so neither outcome is detected).
    //
    // ``launch_copy_`` already serialises new H2Ds behind
    // ``latest_compute_event_`` for the same reason — see
    // VramChunkPool::launch_copy_.  Mirror the same guard here so
    // eviction-side writes also wait out the in-flight kernel
    // before mutating the pointer table.  Cost: eviction blocks
    // until the current compute kernel completes, but eviction
    // already serialises with H2Ds via io_stream_mu_, so this is
    // a wash on critical-path latency.
    if (pool_) {
        pool_->wait_compute_on(stream);
    }

    // ChunkedTensor::after_evict knows the encoder's chunk_idx →
    // plane_idx mapping (ss_anybcq: identity; any-prec: plane_idx_first
    // off chunk_planes). Routes to the registered encoder callback
    // internally.  Caller passes the pool's copy_stream so the clear
    // is async and ordered against subsequent loads (SSOT §6.1.6 step 6).
    it->second.tensor->after_evict(p, stream);
}

void StreamllmRuntime::begin_deferred_pointer_clears_(StreamHandle stream) {
    defer_pointer_clears_ = true;
    deferred_clear_stream_ = stream;
    deferred_clear_patches_.clear();
}

void StreamllmRuntime::flush_deferred_pointer_clears_() {
    if (!defer_pointer_clears_) return;
    StreamHandle stream = deferred_clear_stream_;
    defer_pointer_clears_ = false;
    deferred_clear_stream_ = nullptr;

    if (!deferred_clear_patches_.empty()) {
        if (pool_) {
            pool_->wait_compute_on(stream);
        }
        apply_pointer_patches_async(deferred_clear_patches_, stream);
        deferred_clear_patches_.clear();
    }
}

void StreamllmRuntime::cancel_deferred_pointer_clears_() {
    defer_pointer_clears_ = false;
    deferred_clear_stream_ = nullptr;
    deferred_clear_patches_.clear();
}

void * StreamllmRuntime::small_alloc(size_t bytes) {
    if (bytes == 0) return nullptr;
    // 8-byte align the bump pointer.
    bytes = (bytes + 7) & ~size_t(7);
    std::lock_guard<std::mutex> lk(small_slab_mu_);
    if (small_slab_ == nullptr ||
        small_slab_used_ + bytes > small_slab_bytes_) {
        return nullptr;
    }
    void * out = (char *)small_slab_ + small_slab_used_;
    small_slab_used_ += bytes;
    return out;
}

bool StreamllmRuntime::prepare_chunk_payload_(
    const std::string & wid, int cid, PreparedChunkPayload & out)
{
    auto it = entries_.find(wid);
    if (it == entries_.end()) {
        throw std::runtime_error(
            "StreamllmRuntime::prepare_chunk_payload_: unknown wid " + wid);
    }
    Entry & e = it->second;
    UpstreamLayoutHost & host = e.tensor->host();

    out.key = ChunkKey{wid, cid};
    out.bytes.clear();

    if (cid == kCidQBias) {
        const auto * p = reinterpret_cast<const uint8_t *>(host.q_bias.data());
        const size_t n = host.q_bias.size() * sizeof(uint16_t);
        out.bytes.assign(p, p + n);
        return n > 0;
    }
    if (!cid_is_chunk(cid)) {
        throw std::runtime_error(
            "StreamllmRuntime::prepare_chunk_payload_: unknown cid " +
            std::to_string(cid));
    }

    const int p = cid_chunk_index(cid);
    const bool have_io =
        p >= 0 &&
        p < (int)e.chunk_file_offsets.size() &&
        p < (int)e.chunk_file_sizes.size();

    if (host_chunk_cache_enabled()) {
        std::lock_guard<std::mutex> lk(*e.host_mu);
        if (p >= 0 && p < (int)host.chunks.size() &&
            !host.chunks[p].empty()) {
            out.bytes = host.chunks[p];
            diag::record_move_event(diag::MoveEvent::DramHit);
            diag::record_move_event(diag::MoveEvent::CacheSnapshot);
            return true;
        }
    }

    if (!have_io || host_ring_slot_bytes_ == 0) {
        return false;
    }

    const int64_t off = e.chunk_file_offsets[p];
    const int64_t size = e.chunk_file_sizes[p];
    const bool kernel_ready_source =
        p < (int)e.chunk_file_kernel_ready.size() &&
        e.chunk_file_kernel_ready[p] != 0;
    std::vector<uint8_t> disk_bytes((size_t)size);
    int64_t total = 0;
    while (total < size) {
        ssize_t got = ::pread(gguf_fd_, disk_bytes.data() + total,
                              (size_t)(size - total),
                              (off_t)(off + total));
        if (got < 0) {
            int err = errno;
            throw std::runtime_error(
                "StreamllmRuntime::prepare_chunk_payload_: pread failed: " +
                std::string(std::strerror(err)));
        }
        if (got == 0) {
            throw std::runtime_error(
                "StreamllmRuntime::prepare_chunk_payload_: short pread for " + wid);
        }
        total += got;
    }
    (void)::posix_fadvise(gguf_fd_, (off_t)off, (off_t)size,
                          POSIX_FADV_DONTNEED);
    diag::record_move_event(diag::MoveEvent::SsdMiss);

    size_t expected_disk_bytes = 0;
    size_t kernel_chunk_bytes = 0;
    if (host.any_precision) {
        if (p < 0 || p >= (int)host.chunk_planes.size()) {
            throw std::runtime_error(
                "StreamllmRuntime::prepare_chunk_payload_: any-prec chunk index "
                + std::to_string(p) + " out of range for " + wid);
        }
        expected_disk_bytes = host.chunk_planes[p].disk_chunk_bytes;
        kernel_chunk_bytes  = host.chunk_planes[p].kernel_chunk_bytes;
    } else {
        expected_disk_bytes = host.disk_bytes_per_chunk;
        kernel_chunk_bytes  = host.bytes_per_chunk;
    }
    const size_t expected_source_bytes =
        kernel_ready_source ? kernel_chunk_bytes : expected_disk_bytes;
    if ((size_t)size != expected_source_bytes) {
        throw std::runtime_error(
            "StreamllmRuntime::prepare_chunk_payload_: SSD-stream plane size "
            "doesn't match expected disk layout for " + wid);
    }

    out.bytes.assign(kernel_chunk_bytes, 0);
    if (kernel_ready_source) {
        std::memcpy(out.bytes.data(), disk_bytes.data(), kernel_chunk_bytes);
    } else if (host.direct_matrix || host.direct_expert_block) {
        std::memcpy(out.bytes.data(), disk_bytes.data(), kernel_chunk_bytes);
    } else if (host.disk_to_kernel_fn == nullptr) {
        throw std::runtime_error(
            "StreamllmRuntime::prepare_chunk_payload_: missing disk_to_kernel_fn for " +
            wid);
    } else {
        host.disk_to_kernel_fn(host, p, disk_bytes.data(), out.bytes.data());
    }

    if (host_chunk_cache_enabled()) {
        std::lock_guard<std::mutex> lk(*e.host_mu);
        if (p >= (int)host.chunks.size()) {
            host.chunks.resize(p + 1);
        }
        const size_t prev_bytes = host.chunks[p].size();
        host.chunks[p] = out.bytes;
        const size_t new_bytes = host.chunks[p].size();
        if (new_bytes > prev_bytes) {
            g_host_dram_bytes.fetch_add(new_bytes - prev_bytes,
                                         std::memory_order_relaxed);
        } else if (prev_bytes > new_bytes) {
            g_host_dram_bytes.fetch_sub(prev_bytes - new_bytes,
                                         std::memory_order_relaxed);
        }
        diag::record_move_event(diag::MoveEvent::CacheInsert);
    }
    return true;
}

void StreamllmRuntime::prepare_chunk_payloads_batch_(
    const std::vector<ChunkKey> & keys,
    std::vector<PreparedChunkPayload> & payloads,
    size_t & packed_bytes)
{
    struct PendingBundle {
        ChunkKey key;
        Entry * entry = nullptr;
        int p = -1;
        int64_t off = 0;
        int64_t size = 0;
        std::string bundle;
    };

    std::vector<PendingBundle> pending;
    pending.reserve(keys.size());

    for (const auto & key : keys) {
        if (pool_->is_resident(key.wid, key.cid)) {
            pool_->note_redundant_h2d_skipped();
            continue;
        }
        auto it = entries_.find(key.wid);
        if (it == entries_.end() || !cid_is_chunk(key.cid)) {
            PreparedChunkPayload payload;
            if (prepare_chunk_payload_(key.wid, key.cid, payload)) {
                packed_bytes += payload.bytes.size();
                payloads.push_back(std::move(payload));
            }
            continue;
        }
        Entry & e = it->second;
        const int p = cid_chunk_index(key.cid);

        if (host_chunk_cache_enabled()) {
            std::lock_guard<std::mutex> lk(*e.host_mu);
            auto & chunks = e.tensor->host().chunks;
            if (p >= 0 && p < (int)chunks.size() && !chunks[p].empty()) {
                PreparedChunkPayload payload;
                payload.key = key;
                payload.bytes = chunks[p];
                packed_bytes += payload.bytes.size();
                payloads.push_back(std::move(payload));
                diag::record_move_event(diag::MoveEvent::DramHit);
                diag::record_move_event(diag::MoveEvent::CacheSnapshot);
                continue;
            }
        }

        const bool have_bundle =
            p >= 0 &&
            p < (int)e.chunk_file_offsets.size() &&
            p < (int)e.chunk_file_sizes.size() &&
            p < (int)e.chunk_file_kernel_ready.size() &&
            p < (int)e.chunk_file_bundle.size() &&
            e.chunk_file_kernel_ready[p] != 0 &&
            !e.chunk_file_bundle[p].empty();
        if (!have_bundle) {
            PreparedChunkPayload payload;
            if (prepare_chunk_payload_(key.wid, key.cid, payload)) {
                packed_bytes += payload.bytes.size();
                payloads.push_back(std::move(payload));
            }
            continue;
        }
        pending.push_back(PendingBundle{
            key, &e, p, e.chunk_file_offsets[p], e.chunk_file_sizes[p],
            e.chunk_file_bundle[p],
        });
    }

    std::stable_sort(
        pending.begin(), pending.end(),
        [](const PendingBundle & a, const PendingBundle & b) {
            if (a.bundle != b.bundle) return a.bundle < b.bundle;
            return a.off < b.off;
        });

    size_t i = 0;
    while (i < pending.size()) {
        size_t last = i + 1;
        int64_t span_off = pending[i].off;
        int64_t span_end = pending[i].off + pending[i].size;
        while (last < pending.size() &&
               pending[last].bundle == pending[i].bundle &&
               pending[last].off == span_end) {
            span_end += pending[last].size;
            ++last;
        }
        const int64_t span_size = span_end - span_off;
        std::vector<uint8_t> span((size_t)span_size);
        {
            DiagSpan _t(diag::MoveSite::Pread);
            int64_t total = 0;
            while (total < span_size) {
                ssize_t got = ::pread(gguf_fd_, span.data() + total,
                                      (size_t)(span_size - total),
                                      (off_t)(span_off + total));
                if (got < 0) {
                    int err = errno;
                    throw std::runtime_error(
                        "StreamllmRuntime::prepare_chunk_payloads_batch_: pread failed: " +
                        std::string(std::strerror(err)));
                }
                if (got == 0) {
                    throw std::runtime_error(
                        "StreamllmRuntime::prepare_chunk_payloads_batch_: short pread");
                }
                total += got;
            }
            (void)::posix_fadvise(gguf_fd_, (off_t)span_off,
                                  (off_t)span_size, POSIX_FADV_DONTNEED);
        }
        diag::record_move_event(diag::MoveEvent::SsdMiss);

        for (size_t j = i; j < last; ++j) {
            const PendingBundle & pb = pending[j];
            UpstreamLayoutHost & host = pb.entry->tensor->host();
            size_t kernel_chunk_bytes = host.bytes_per_chunk;
            if (host.any_precision) {
                if (pb.p < 0 || pb.p >= (int)host.chunk_planes.size()) {
                    throw std::runtime_error(
                        "StreamllmRuntime::prepare_chunk_payloads_batch_: "
                        "any-prec chunk index out of range for " + pb.key.wid);
                }
                kernel_chunk_bytes = host.chunk_planes[pb.p].kernel_chunk_bytes;
            }
            if ((size_t)pb.size != kernel_chunk_bytes) {
                throw std::runtime_error(
                    "StreamllmRuntime::prepare_chunk_payloads_batch_: "
                    "bundle kernel-ready slice size mismatch for " + pb.key.wid);
            }
            const size_t rel = (size_t)(pb.off - span_off);
            PreparedChunkPayload payload;
            payload.key = pb.key;
            payload.bytes.assign(span.data() + rel,
                                 span.data() + rel + kernel_chunk_bytes);

            if (host_chunk_cache_enabled()) {
                std::lock_guard<std::mutex> lk(*pb.entry->host_mu);
                if (pb.p >= (int)host.chunks.size()) host.chunks.resize(pb.p + 1);
                const size_t prev_bytes = host.chunks[pb.p].size();
                host.chunks[pb.p] = payload.bytes;
                const size_t new_bytes = host.chunks[pb.p].size();
                if (new_bytes > prev_bytes) {
                    g_host_dram_bytes.fetch_add(new_bytes - prev_bytes,
                                                std::memory_order_relaxed);
                } else if (prev_bytes > new_bytes) {
                    g_host_dram_bytes.fetch_sub(prev_bytes - new_bytes,
                                                std::memory_order_relaxed);
                }
                diag::record_move_event(diag::MoveEvent::CacheInsert);
            }

            packed_bytes += payload.bytes.size();
            payloads.push_back(std::move(payload));
        }
        i = last;
    }
}

void StreamllmRuntime::finish_loaded_chunk_(
    const std::string & wid, int cid, void * device_ptr)
{
    auto it = entries_.find(wid);
    if (it == entries_.end()) return;
    Entry & e = it->second;
    UpstreamLayoutHost & host = e.tensor->host();
    UpstreamLayoutDevice & dev = e.dev;

    if (cid == kCidQBias) {
        dev.q_bias_fp16 = device_ptr;
    } else if (cid_is_chunk(cid)) {
        const int p = cid_chunk_index(cid);
        if (p < kMaxChunksPerTensor) {
            dev.chunk_ptrs[p] = device_ptr;
            if (device_ptr != nullptr) {
                e.tensor->after_load(p, device_ptr, pool_->copy_stream());
                if (host.any_precision &&
                    p < (int)host.chunk_planes.size()) {
                    const auto & cp = host.chunk_planes[p];
                    dev.q_bias_fp16 =
                        (const uint8_t *) device_ptr + cp.ker_off_qbias;
                }
            }
        }
    }

    dev.M                  = host.n;
    dev.K                  = host.padded_m;
    dev.n_chunks           = host.n_chunks;
    dev.K_groups           = host.K_groups;
    dev.group_size         = host.group_size;
    dev.qw_bytes_per_chunk = host.qw_bytes_per_chunk;
    dev.any_precision      = host.any_precision;
    dev.base_precision     = host.base_precision;
    pool_->mark_pointer_table_ready(wid, cid);
}

void StreamllmRuntime::finish_loaded_chunks_batch_(
    const std::vector<std::pair<ChunkKey, void *>> & loaded)
{
    std::vector<PointerPatch> patches;
    patches.reserve(loaded.size());
    std::vector<ChunkKey> ready_keys;
    ready_keys.reserve(loaded.size());

    for (const auto & item : loaded) {
        const ChunkKey & key = item.first;
        void * device_ptr = item.second;
        auto it = entries_.find(key.wid);
        if (it == entries_.end()) continue;
        Entry & e = it->second;
        UpstreamLayoutHost & host = e.tensor->host();
        UpstreamLayoutDevice & dev = e.dev;

        if (key.cid == kCidQBias) {
            dev.q_bias_fp16 = device_ptr;
        } else if (cid_is_chunk(key.cid)) {
            const int p = cid_chunk_index(key.cid);
            if (p < kMaxChunksPerTensor) {
                dev.chunk_ptrs[p] = device_ptr;
                if (device_ptr != nullptr) {
                    if (!e.tensor->append_after_load_patch(
                            p, device_ptr, patches)) {
                        e.tensor->after_load(p, device_ptr,
                                             pool_->copy_stream());
                    }
                    if (host.any_precision &&
                        p < (int)host.chunk_planes.size()) {
                        const auto & cp = host.chunk_planes[p];
                        dev.q_bias_fp16 =
                            (const uint8_t *) device_ptr + cp.ker_off_qbias;
                    }
                }
            }
        }

        dev.M                  = host.n;
        dev.K                  = host.padded_m;
        dev.n_chunks           = host.n_chunks;
        dev.K_groups           = host.K_groups;
        dev.group_size         = host.group_size;
        dev.qw_bytes_per_chunk = host.qw_bytes_per_chunk;
        dev.any_precision      = host.any_precision;
        dev.base_precision     = host.base_precision;
        ready_keys.push_back(key);
    }

    apply_pointer_patches_async(patches, pool_->copy_stream());
    for (const auto & key : ready_keys) {
        pool_->mark_pointer_table_ready(key.wid, key.cid);
    }
}

EventHandle StreamllmRuntime::move_chunks_batch(
    const std::vector<ChunkKey> & keys,
    StreamHandle compute_stream)
{
    if (keys.empty()) return nullptr;
    std::vector<PreparedChunkPayload> payloads;
    payloads.reserve(keys.size());
    size_t packed_bytes = 0;
    prepare_chunk_payloads_batch_(keys, payloads, packed_bytes);
    if (payloads.empty() || packed_bytes == 0) return nullptr;

    size_t max_batch_bytes = 4ull * 1024ull * 1024ull;
    if (const char * env = std::getenv("STREAMLLM_BATCH_CHUNK_MAX_MB")) {
        const long mb = std::strtol(env, nullptr, 10);
        if (mb > 0) {
            max_batch_bytes = (size_t)mb * 1024ull * 1024ull;
        }
    }

    auto run_payload_batch = [&](size_t first, size_t last, size_t nbytes) {
        if (nbytes > batch_staging_bytes_) {
            if (batch_staging_) {
                cudaFreeHost(batch_staging_);
                batch_staging_ = nullptr;
                batch_staging_bytes_ = 0;
            }
            cudaError_t cerr = cudaHostAlloc(&batch_staging_, nbytes,
                                             cudaHostAllocDefault);
            if (cerr != cudaSuccess) {
                throw std::runtime_error(
                    std::string("StreamllmRuntime::move_chunks_batch: "
                                "cudaHostAlloc(batch staging) failed: ") +
                    cudaGetErrorString(cerr));
            }
            batch_staging_bytes_ = nbytes;
        }
        uint8_t * packed = reinterpret_cast<uint8_t *>(batch_staging_);
        std::vector<BatchLoadItem> items;
        items.reserve(last - first);
        size_t cursor = 0;
        for (size_t i = first; i < last; ++i) {
            const auto & payload = payloads[i];
            std::memcpy(packed + cursor,
                        payload.bytes.data(),
                        payload.bytes.size());
            items.push_back(BatchLoadItem{
                payload.key,
                packed + cursor,
                payload.bytes.size(),
            });
            cursor += payload.bytes.size();
        }

        std::vector<ChunkHandle> handles;
        begin_deferred_pointer_clears_(pool_->copy_stream());
        try {
            for (;;) {
                handles = pool_->load_batch_sync(items, packed,
                                                 nbytes, compute_stream);
                if (!handles.empty()) break;
                if (!scheduler_->make_room_for(*pool_, nbytes)) {
                    flush_deferred_pointer_clears_();
                    pool_->note_batch_fallback();
                    for (size_t i = first; i < last; ++i) {
                        const auto & payload = payloads[i];
                        (void)move_chunk(payload.key.wid, payload.key.cid,
                                         Tier::RAM, Tier::VRAM, compute_stream);
                        pool_->wait_on_stream(payload.key.wid, payload.key.cid,
                                              compute_stream);
                    }
                    return;
                }
            }
            flush_deferred_pointer_clears_();
        } catch (...) {
            cancel_deferred_pointer_clears_();
            throw;
        }

        std::vector<std::pair<ChunkKey, void *>> loaded;
        loaded.reserve(handles.size());
        for (size_t j = 0; j < handles.size(); ++j) {
            const size_t i = first + j;
            loaded.push_back({payloads[i].key, handles[j].device_ptr});
        }
        finish_loaded_chunks_batch_(loaded);
        if (pool_->copy_stream() != nullptr) {
            cudaStreamSynchronize((cudaStream_t)pool_->copy_stream());
        }
    };

    size_t first = 0;
    size_t cur_bytes = 0;
    for (size_t i = 0; i < payloads.size(); ++i) {
        const size_t n = payloads[i].bytes.size();
        if (i > first && cur_bytes + n > max_batch_bytes) {
            run_payload_batch(first, i, cur_bytes);
            first = i;
            cur_bytes = 0;
        }
        cur_bytes += n;
    }
    if (first < payloads.size()) {
        run_payload_batch(first, payloads.size(), cur_bytes);
    }
    return nullptr;
}


EventHandle StreamllmRuntime::move_chunk(const std::string & wid, int cid,
                                          Tier src, Tier dst,
                                          StreamHandle compute_stream) {
    STLM_NVTX_RANGE("move_chunk");
    if (src != Tier::RAM || dst != Tier::VRAM) {
        // Only RAM→VRAM is implemented; SSD reads happen inline below
        // when host.chunks[p] is empty.
        return nullptr;
    }

    // ── Residency-skip invariant (canonical chokepoint).
    //
    //   If the chunk is already in the pool, the entry's dev pointers
    //   and the per-plane pointer table were populated by the prior
    //   ``move_chunk`` that originally loaded it (and ``after_load`` ran
    //   then; eviction nulls dev.chunk_ptrs and pool.evict erases the
    //   resident entry atomically).  Skipping here saves the entire
    //   pread / disk-to-kernel xform / DRAM-cache snapshot / pool.load
    //   chain and — crucially — guarantees the invariant for every
    //   caller (executor, async io worker, scheduler install) without
    //   relying on each call site to remember the pre-check.
    if (pool_->is_resident(wid, cid)) {
        pool_->note_redundant_h2d_skipped();
        ChunkHandle existing = pool_->view(wid, cid);
        return existing.ready_event;
    }

    auto it = entries_.find(wid);
    if (it == entries_.end()) {
        throw std::runtime_error(
            "StreamllmRuntime::move_chunk: unknown wid " + wid);
    }
    Entry & e = it->second;
    UpstreamLayoutHost & host = e.tensor->host();
    UpstreamLayoutDevice & dev = e.dev;

    const void * src_ptr = nullptr;
    size_t       nbytes  = 0;
    // Track which ring slot (if any) the bytes came from so we can
    // tag it with the H2D's ready-event and let the ring reclaim it
    // when the event fires (next time we wrap to that slot).
    int          ring_slot_idx = -1;
    // Profile state: set by the SSD-stream branch and consumed after
    // pool_->load to attribute pread vs pool-load wall time.
    bool     mc_profile_active = false;
    uint64_t mc_profile_pread_ns = 0;
    // Scratch for the disk-format → kernel-format plane transform used
    // by the SSD-stream branch. Function-scoped so its lifetime extends
    // through pool_->load below (which copies from src_ptr); per-call
    // so concurrent async pread workers don't share state.
    std::vector<uint8_t> xform_scratch;
    bool src_ptr_is_ephemeral = false;

    if (cid == kCidQBias) {
        src_ptr = host.q_bias.data();
        nbytes  = host.q_bias.size() * sizeof(uint16_t);
    } else if (cid_is_chunk(cid)) {
        int p = cid_chunk_index(cid);
        const bool have_io =
            p >= 0 &&
            p < (int)e.chunk_file_offsets.size() &&
            p < (int)e.chunk_file_sizes.size();

        // Try the DRAM cache first. Snapshot bytes into the local
        // xform_scratch under the per-Entry mutex, then drop the lock
        // — pool_->load reads from xform_scratch (function-local,
        // stable for the rest of this call) so concurrent
        // release_chunk_host eviction can't yank the source out from
        // under it. Costs one ~240 KB memcpy/hit but eliminates the
        // cross-thread contention we'd hit holding the lock through
        // pool_->load.
        if (host_chunk_cache_enabled()) {
            std::lock_guard<std::mutex> lk(*e.host_mu);
            if (p >= 0 && p < (int)host.chunks.size() &&
                !host.chunks[p].empty()) {
                DiagSpan _t(diag::MoveSite::HostCacheSnapshot);
                xform_scratch.assign(host.chunks[p].begin(),
                                     host.chunks[p].end());
                src_ptr = xform_scratch.data();
                nbytes  = xform_scratch.size();
                src_ptr_is_ephemeral = true;
                diag::record_move_event(diag::MoveEvent::DramHit);
                diag::record_move_event(diag::MoveEvent::CacheSnapshot);
            }
        }
        if (src_ptr == nullptr && have_io && host_ring_ != nullptr) {
            // SSD-stream: pread chunk bytes into a pinned ring slot.
            const int64_t off  = e.chunk_file_offsets[p];
            const int64_t size = e.chunk_file_sizes[p];
            const bool kernel_ready_source =
                p < (int)e.chunk_file_kernel_ready.size() &&
                e.chunk_file_kernel_ready[p] != 0;
            if ((size_t)size > host_ring_slot_bytes_) {
                throw std::runtime_error(
                    "StreamllmRuntime::move_chunk: chunk size (" +
                    std::to_string(size) + ") exceeds ring slot size (" +
                    std::to_string(host_ring_slot_bytes_) + ")");
            }
            // Reserve next slot. fetch_add lets multiple async
            // workers reserve distinct slots without locks. The ring
            // size (32) >> typical concurrency (4 workers) so wrap
            // contention on the same slot is rare.
            const int n_slots = (int)host_ring_slot_ptrs_.size();
            const uint32_t raw =
                host_ring_next_.fetch_add(1, std::memory_order_relaxed);
            ring_slot_idx = (int)(raw % (uint32_t)n_slots);
            // Slot reclamation: the pool's copy stream is FIFO-ordered,
            // so by the time we wrap back to a given slot we've issued
            // n_slots = 32 subsequent H2Ds. With ~290 KB/chunk and PCIe
            // DMA at 5-25 GB/s, the in-flight queue drains in ~0.4-2 ms
            // — well below typical hook re-entry intervals. We don't
            // track per-slot events; the event itself is owned by the
            // pool and tied to its slot life-cycle.
            void * dst = host_ring_slot_ptrs_[ring_slot_idx];

            const bool prof = (getenv("STREAMLLM_PROFILE") != nullptr);
            const auto t_pread_start = prof
                ? std::chrono::steady_clock::now()
                : std::chrono::steady_clock::time_point{};
            STLM_NVTX_RANGE("pread");
            uint64_t pread_ns = 0;
            {
                DiagSpan _t(diag::MoveSite::Pread);
                int64_t total = 0;
                while (total < size) {
                    ssize_t got = ::pread(gguf_fd_, (char *)dst + total,
                                          (size_t)(size - total),
                                          (off_t)(off + total));
                    if (got < 0) {
                        int err = errno;
                        throw std::runtime_error(
                            "StreamllmRuntime::move_chunk: pread failed: " +
                            std::string(std::strerror(err)));
                    }
                    if (got == 0) {
                        throw std::runtime_error(
                            "StreamllmRuntime::move_chunk: short pread for " + wid);
                    }
                    total += got;
                }
                (void)::posix_fadvise(gguf_fd_, (off_t)off, (off_t)size,
                                      POSIX_FADV_DONTNEED);
                if (prof) {
                    auto dt = std::chrono::steady_clock::now() - t_pread_start;
                    pread_ns = (uint64_t)std::chrono::duration_cast<
                        std::chrono::nanoseconds>(dt).count();
                }
            }
            diag::record_move_event(diag::MoveEvent::SsdMiss);
            // CRITICAL: transform on-disk chunk bytes → kernel-format.
            //
            // BASE-pinned chunks are pre-transformed at install time by
            // plane_disk_to_kernel: signs go from [n, padded_m/8] MSB-
            // packed bytes to [K/32, n] LSB-packed uint32, and alpha goes
            // from [row, kg] fp32 to [kg, row] fp16. HOT-promoted chunks
            // streamed from disk skipped that step before this fix —
            // device saw disk-format bytes that the kernel reinterpreted
            // as kernel-format → garbage scales and bit-flipped signs →
            // NaN / wrong output. Visible only when HOT > 0.
            // Per-chunk size + transform dispatch.  SsAnybcq layout has
            // a uniform per-plane disk size; any-prec has variable
            // per-chunk sizes (chunk 0 carries base_p planes; later
            // chunks carry 1).  Both go through codec-aware
            // plane_disk_to_kernel / any_prec_chunk_disk_to_kernel.
            size_t expected_disk_bytes;
            size_t kernel_chunk_bytes;
            if (host.any_precision) {
                if (p < 0 || p >= (int)host.chunk_planes.size()) {
                    throw std::runtime_error(
                        "StreamllmRuntime::move_chunk: any-prec chunk index "
                        + std::to_string(p) + " out of range for " + wid);
                }
                expected_disk_bytes = host.chunk_planes[p].disk_chunk_bytes;
                kernel_chunk_bytes  = host.chunk_planes[p].kernel_chunk_bytes;
            } else {
                expected_disk_bytes = host.disk_bytes_per_chunk;
                kernel_chunk_bytes  = host.bytes_per_chunk;
            }
            const size_t expected_source_bytes =
                kernel_ready_source ? kernel_chunk_bytes : expected_disk_bytes;
            if ((size_t)size != expected_source_bytes) {
                throw std::runtime_error(
                    "StreamllmRuntime::move_chunk: SSD-stream plane size "
                    "doesn't match expected disk layout for " + wid +
                    " (cid=" + std::to_string(cid) + ", got " +
                    std::to_string(size) + " expected " +
                    std::to_string(expected_disk_bytes) + ")");
            }
            {
                DiagSpan _a(diag::MoveSite::Alloc);
                xform_scratch.assign(kernel_chunk_bytes, 0);
            }
            {
                DiagSpan _x(diag::MoveSite::Xform);
                // Direct/native chunks are stored in the same byte layout
                // consumed by GGML, so SSD streaming is a raw copy. Encoded
                // chunk families install a codec-aware disk->kernel
                // transform below.
                if (kernel_ready_source ||
                    host.direct_matrix || host.direct_expert_block) {
                    std::memcpy(xform_scratch.data(), dst, kernel_chunk_bytes);
                } else if (host.disk_to_kernel_fn == nullptr) {
                    throw std::runtime_error(
                        "StreamllmRuntime::move_chunk: missing "
                        "disk_to_kernel_fn for " + wid);
                } else {
                    host.disk_to_kernel_fn(
                        host, p,
                        (const uint8_t *) dst,
                        xform_scratch.data());
                }
            }
            // Populate the DRAM cache: copy the kernel-format bytes
            // into host.chunks[p] under the per-Entry mutex so a
            // future move_chunk for the same (wid, cid) can hit the
            // cache instead of re-preading. Eviction is the
            // scheduler's job (release_chunk_host under cap pressure);
            // here we just keep the bytes alive as a side effect.
            if (host_chunk_cache_enabled()) {
                DiagSpan _i(diag::MoveSite::HostCacheInsert);
                std::lock_guard<std::mutex> lk(*e.host_mu);
                if (p >= (int)host.chunks.size()) {
                    host.chunks.resize(p + 1);
                }
                const size_t prev_bytes = host.chunks[p].size();
                host.chunks[p] = xform_scratch;
                const size_t new_bytes  = host.chunks[p].size();
                if (new_bytes > prev_bytes) {
                    g_host_dram_bytes.fetch_add(new_bytes - prev_bytes,
                                                 std::memory_order_relaxed);
                } else if (prev_bytes > new_bytes) {
                    g_host_dram_bytes.fetch_sub(prev_bytes - new_bytes,
                                                 std::memory_order_relaxed);
                }
                diag::record_move_event(diag::MoveEvent::CacheInsert);
            }
            src_ptr = xform_scratch.data();
            // For any-prec the kernel-chunk byte size is per-chunk, not
            // a single host-wide constant.  bytes_per_chunk on the host
            // layout is the MAX across chunks (used by the pool for
            // scratch sizing); use the actual per-chunk size for the
            // pool load.
            nbytes  = kernel_chunk_bytes;
            src_ptr_is_ephemeral = true;
            // Stash for profile attribution after pool_->load below.
            mc_profile_pread_ns = pread_ns;
            mc_profile_active   = prof;
        }
        if (src_ptr == nullptr) {
            // No host bytes AND no SSD chunk_io for this plane —
            // most commonly because the encoder/reader registered
            // fewer planes than the scheduler is asking for. Silently
            // no-op (returning a null event); the kernel can still
            // run with the planes that ARE resident, and stderr-
            // spamming inside prefetch workers ruins TPS in tight
            // cap regimes.
            return nullptr;
        }
    } else {
        throw std::runtime_error(
            "StreamllmRuntime::move_chunk: unknown cid " +
            std::to_string(cid));
    }

    const auto t_pool_start = mc_profile_active
        ? std::chrono::steady_clock::now()
        : std::chrono::steady_clock::time_point{};
    STLM_NVTX_MARK("pool.load");
    ChunkHandle h;
    {
        DiagSpan _l(diag::MoveSite::PoolLoadEnqueue);
        h = pool_->load(wid, cid, src_ptr, nbytes, compute_stream);
    }
    // Pool returns a null handle when the arena can't fit ``nbytes``
    // (full or too fragmented). Pool itself is policy-free; ask the
    // scheduler to evict a victim, then retry. Schedulers that don't
    // implement make_room_for leave its default false → throw on pool
    // exhaustion.
    while (h.device_ptr == nullptr) {
        profile_record_pool_make_room();
        {
            DiagSpan _r(diag::MoveSite::MakeRoom);
            if (!scheduler_->make_room_for(*pool_, nbytes)) {
                throw std::runtime_error(
                    "StreamllmRuntime::move_chunk: pool full and scheduler "
                    "(" + std::string(scheduler_->name()) + ") could not free "
                    "space for " + std::to_string(nbytes) + " bytes (" +
                    wid + ", cid=" + std::to_string(cid) + ")");
            }
        }
        DiagSpan _l(diag::MoveSite::PoolLoadEnqueue);
        h = pool_->load(wid, cid, src_ptr, nbytes, compute_stream);
    }
    if (src_ptr_is_ephemeral && h.ready_event != nullptr) {
        // ``src_ptr`` points into xform_scratch, a function-local buffer.
        // The pool copy stream may still be DMA-reading it after
        // pool_->load returns.  Keep the buffer alive until the H2D event
        // fires; otherwise the next allocator reuse can corrupt the
        // transfer and surface as an illegal memory access in a later
        // CUDA call.
        cudaError_t e_sync = cudaEventSynchronize((cudaEvent_t) h.ready_event);
        if (e_sync != cudaSuccess) {
            throw std::runtime_error(
                "StreamllmRuntime::move_chunk: H2D wait failed for " + wid +
                " (cid=" + std::to_string(cid) + "): " +
                cudaGetErrorString(e_sync));
        }
    }
    if (mc_profile_active) {
        auto dt = std::chrono::steady_clock::now() - t_pool_start;
        uint64_t pool_load_ns = (uint64_t)std::chrono::duration_cast<
            std::chrono::nanoseconds>(dt).count();
        profile_record_move_chunk_ssd(mc_profile_pread_ns, pool_load_ns);
    }
    (void)ring_slot_idx;  // FIFO-trusted; no per-slot event tracking.

    if (cid == kCidQBias) {
        dev.q_bias_fp16 = h.device_ptr;
    } else if (cid_is_chunk(cid)) {
        int p = cid_chunk_index(cid);
        if (p < kMaxChunksPerTensor) {
            dev.chunk_ptrs[p] = h.device_ptr;
            DiagSpan _u(diag::MoveSite::PtrUpdate);
            if (h.device_ptr != nullptr) {
                // ChunkedTensor::after_load (encoder-registered)
                // writes per-plane device pointer-table entries the
                // kernel reads. Async path uses the pool's copy_stream;
                // sync fallback uses the default stream.
                e.tensor->after_load(p, h.device_ptr, pool_->copy_stream());
                if (pool_->copy_stream() != nullptr && h.ready_event != nullptr) {
                    cudaEventRecord((cudaEvent_t)h.ready_event,
                                    (cudaStream_t)pool_->copy_stream());
                }
                // Step 6 (Milestone 1): advance the chunk's residency
                // state to POINTER_TABLE_READY now that after_load has
                // been enqueued on copy_stream and ready_event has
                // been re-recorded post-after_load. The host-side
                // pre-launch validation in
                // Qwen3MoEAnyBcqExecutor::validate_required_set_
                // refuses to launch until every required chunk reaches
                // this state.
                pool_->mark_pointer_table_ready(wid, cid);
                // Any-prec only: stash β pointer host-side so the kernel
                // launcher can dereference dev.q_bias_fp16 directly.
                // Last-landed chunk wins; chunk_matmul re-asserts this
                // from the highest-cid in the plan before launch.
                if (host.any_precision &&
                    p < (int)host.chunk_planes.size()) {
                    const auto & cp = host.chunk_planes[p];
                    dev.q_bias_fp16 =
                        (const uint8_t *) h.device_ptr + cp.ker_off_qbias;
                }
            }
        }
    }

    dev.M                  = host.n;
    dev.K                  = host.padded_m;
    dev.n_chunks                  = host.n_chunks;
    dev.K_groups           = host.K_groups;
    dev.group_size         = host.group_size;
    dev.qw_bytes_per_chunk = host.qw_bytes_per_chunk;
    dev.any_precision      = host.any_precision;
    dev.base_precision     = host.base_precision;

    // The pool itself doesn't track pin state. Schedulers that want
    // "keep resident forever" semantics simply never call pool.evict()
    // on those chunks. Schedulers that stream (e.g. MoE HOT promotion)
    // implement make_room_for() so eviction picks a model-appropriate
    // victim when the arena is full.
    return h.ready_event;
}

// (chunk_matmul / chunk_matmul_batched moved to
// decoder/ss_anybcq/chunked_matmul.{h,cu} as
// ``ss_anybcq::chunk_matmul_for_wid`` / ``..._batched_for_wid``.
// Core's runtime now exposes only the layout + pool primitives those
// helpers need — no encoder dispatch lives here.)

// --- async prefetch worker pool -----------------------------------

bool StreamllmRuntime::submit_async_load(const std::string & wid, int cid) {
    if (io_workers_.empty()) return false;
    {
        std::lock_guard<std::mutex> lk(io_mu_);
        io_queue_.push_back(AsyncLoadRequest{wid, cid, nullptr, nullptr});
        io_in_flight_.fetch_add(1, std::memory_order_relaxed);
    }
    io_cv_work_.notify_one();
    return true;
}

bool StreamllmRuntime::submit_async_load(
    const std::string & wid, int cid,
    std::shared_ptr<std::atomic<uint32_t>> batch_remaining,
    StreamHandle compute_stream)
{
    if (io_workers_.empty()) return false;
    if (batch_remaining) {
        batch_remaining->fetch_add(1, std::memory_order_relaxed);
    }
    {
        std::lock_guard<std::mutex> lk(io_mu_);
        io_queue_.push_back(
            AsyncLoadRequest{wid, cid, compute_stream,
                             std::move(batch_remaining)});
        io_in_flight_.fetch_add(1, std::memory_order_relaxed);
    }
    io_cv_work_.notify_one();
    return true;
}

void StreamllmRuntime::wait_async_load_idle() {
    if (io_workers_.empty()) return;
    std::unique_lock<std::mutex> lk(io_mu_);
    io_cv_done_.wait(lk, [this] {
        return io_queue_.empty() &&
               io_in_flight_.load(std::memory_order_relaxed) == 0;
    });
}

void StreamllmRuntime::wait_async_load_batch(
    const std::shared_ptr<std::atomic<uint32_t>> & remaining)
{
    if (!remaining) return;
    if (io_workers_.empty()) return;
    // Spin briefly, then fall back to cv-wake. The 100 ns spin avoids
    // a syscall when chunks are already loaded by the time we wait.
    for (int i = 0; i < 64; ++i) {
        if (remaining->load(std::memory_order_acquire) == 0) return;
    }
    std::unique_lock<std::mutex> lk(io_mu_);
    io_cv_done_.wait(lk, [&] {
        return remaining->load(std::memory_order_acquire) == 0;
    });
}

void StreamllmRuntime::io_worker_loop_() {
    for (;;) {
        AsyncLoadRequest req;
        {
            std::unique_lock<std::mutex> lk(io_mu_);
            io_cv_work_.wait(lk, [this] {
                return io_stop_.load(std::memory_order_relaxed) ||
                       !io_queue_.empty();
            });
            if (io_queue_.empty()) {
                if (io_stop_.load(std::memory_order_relaxed)) return;
                continue;
            }
            req = std::move(io_queue_.front());
            io_queue_.pop_front();
        }

        // Serialize copy_stream emissions across workers — see
        // io_stream_mu_'s field doc on runtime.h. The lock covers
        // move_chunk's cudaMemcpyAsync + after_load kernel + per-
        // chunk event-record, so concurrent workers' emissions queue
        // in lock-acquire order rather than interleaving on copy_stream
        // (where they'd race against each other's plane-pointer table
        // updates).
        bool last_batch = false;
        {
            std::lock_guard<std::mutex> stream_lk(io_stream_mu_);
            try {
                (void) move_chunk(req.wid, req.cid,
                                   Tier::RAM, Tier::VRAM,
                                   req.compute_stream);
            } catch (const std::exception & e) {
                static std::atomic<uint64_t> err_count{0};
                uint64_t v = err_count.fetch_add(
                    1, std::memory_order_relaxed);
                if (v < 8 || (v % 100000 == 0)) {
                    std::fprintf(stderr,
                        "streamllm-ext: load worker failed for "
                        "(%s, %d): %s [err#%lu]\n",
                        req.wid.c_str(), req.cid, e.what(),
                        (unsigned long)(v + 1));
                }
            }
            last_batch = req.batch_remaining
                ? req.batch_remaining->fetch_sub(
                      1, std::memory_order_acq_rel) == 1
                : false;
        }

        const bool last_global =
            io_in_flight_.fetch_sub(1, std::memory_order_acq_rel) == 1;
        if (last_global || last_batch) {
            std::lock_guard<std::mutex> lk(io_mu_);
            io_cv_done_.notify_all();
        }
    }
}

// ---------------------------------------------------------------------------
// Replay-scoped state — KBar snapshot + chunk reservations cleared
// at the end of each graph_compute pass.
// ---------------------------------------------------------------------------

void StreamllmRuntime::set_replay_kbar(float kbar) {
    std::lock_guard<std::mutex> lk(replay_dial_mu_);
    replay_kbar_ = kbar;
}

float StreamllmRuntime::current_replay_kbar() const {
    std::lock_guard<std::mutex> lk(replay_dial_mu_);
    return replay_kbar_;
}

void StreamllmRuntime::set_replay_dial_version(uint64_t v) {
    replay_dial_version_.store(v, std::memory_order_release);
}

uint64_t StreamllmRuntime::replay_dial_version() const {
    return replay_dial_version_.load(std::memory_order_acquire);
}

} // namespace streamllm_ext
