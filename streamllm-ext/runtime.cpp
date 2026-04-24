// streamllm-ext — runtime implementation. Thin facade over
// VramChunkPool + Scheduler, exposing the two primitives.

#include "runtime.h"

#include "cast_f16.h"
#include "naver_gemv.h"
#include "dequant_planes.h"
#include "batched_gemm.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace streamllm_ext {

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
    , scheduler_(make_scheduler(scheduler_name)) {
    // Pre-allocate decode kernel scratch. Initial acc_f32 capacity sized
    // to 1 M elems (4 MiB) — covers every Qwen3 M; grows on demand for
    // larger lm_head / vocab matmuls.
    gemv_scratch_.init(device, /*acc_capacity_elems=*/1 << 20);
}

StreamllmRuntime::~StreamllmRuntime() {
    for (auto & kv : entries_) {
        Entry & e = kv.second;
        if (e.d_chunk_qw_ptrs)    { cudaFree(e.d_chunk_qw_ptrs);    e.d_chunk_qw_ptrs    = nullptr; }
        if (e.d_chunk_alpha_ptrs) { cudaFree(e.d_chunk_alpha_ptrs); e.d_chunk_alpha_ptrs = nullptr; }
    }
    gemv_scratch_.destroy();
}

void StreamllmRuntime::install(const StreamReader & reader,
                               const std::string & gguf_path) {
    if (installed_) {
        throw std::runtime_error(
            "StreamllmRuntime: already installed; each instance handles one GGUF");
    }
    scheduler_->on_install(*this, reader, gguf_path);
    installed_ = true;
}

void StreamllmRuntime::register_layout(const std::string & wid,
                                        UpstreamLayoutHost host,
                                        UpstreamLayoutDevice dev) {
    auto [it, _] = entries_.emplace(wid, Entry{});
    it->second.host = std::move(host);
    it->second.dev  = dev;
}

UpstreamLayoutDevice * StreamllmRuntime::mutable_layout(const std::string & wid) {
    auto it = entries_.find(wid);
    if (it == entries_.end()) return nullptr;
    return &it->second.dev;
}

const UpstreamLayoutDevice * StreamllmRuntime::layout(const std::string & wid) const {
    auto it = entries_.find(wid);
    if (it == entries_.end()) return nullptr;
    return &it->second.dev;
}

const UpstreamLayoutHost * StreamllmRuntime::host_layout(const std::string & wid) const {
    auto it = entries_.find(wid);
    if (it == entries_.end()) return nullptr;
    return &it->second.host;
}

EventHandle StreamllmRuntime::move_chunk(const std::string & wid, int cid,
                                          Tier src, Tier dst,
                                          StreamHandle compute_stream) {
    if (src != Tier::RAM || dst != Tier::VRAM) {
        // Other tier transitions are placeholders for M10's pinned-host
        // SSD tier and for M9's eviction path.
        return nullptr;
    }
    auto it = entries_.find(wid);
    if (it == entries_.end()) {
        throw std::runtime_error(
            "StreamllmRuntime::move_chunk: unknown wid " + wid);
    }
    Entry & e = it->second;
    const UpstreamLayoutHost & host = e.host;
    UpstreamLayoutDevice & dev = e.dev;

    const void * src_ptr = nullptr;
    size_t       nbytes  = 0;

    if (cid == kCidQBias) {
        src_ptr = host.q_bias.data();
        nbytes  = host.q_bias.size() * sizeof(uint16_t);
    } else if (cid_is_chunk(cid)) {
        int p = cid_chunk_index(cid);
        if (p < 0 || p >= (int)host.chunks.size()) {
            throw std::runtime_error(
                "StreamllmRuntime::move_chunk: plane " +
                std::to_string(p) + " out of range for " + wid);
        }
        src_ptr = host.chunks[p].data();
        nbytes  = host.chunks[p].size();
    } else {
        throw std::runtime_error(
            "StreamllmRuntime::move_chunk: unknown cid " +
            std::to_string(cid));
    }

    ChunkHandle h = pool_->load(wid, cid, src_ptr, nbytes, compute_stream);

    if (cid == kCidQBias) {
        dev.q_bias_fp16 = h.device_ptr;
    } else if (cid_is_chunk(cid)) {
        int p = cid_chunk_index(cid);
        if (p < kMaxChunksPerTensor) {
            dev.chunk_ptrs[p] = h.device_ptr;
            // Lazy-allocate the per-tensor device-side ptr arrays on
            // the first plane of a tensor. Both arrays together cost
            // 256 B per managed tensor (~64 KB total for Qwen3-4B), so
            // we don't bother with per-tensor sizing.
            if (e.d_chunk_qw_ptrs == nullptr) {
                cudaMalloc((void **)&e.d_chunk_qw_ptrs,
                           kMaxChunksPerTensor * sizeof(void *));
                cudaMalloc((void **)&e.d_chunk_alpha_ptrs,
                           kMaxChunksPerTensor * sizeof(void *));
            }
            if (e.d_chunk_qw_ptrs && e.d_chunk_alpha_ptrs) {
                // Derive qw / alpha base from the chunk (signs then α).
                const void * qw_p    = h.device_ptr;
                const void * alpha_p = static_cast<const uint8_t *>(h.device_ptr) +
                                       host.qw_bytes_per_chunk;
                // One-off tiny H2D on the pool's copy stream so it orders
                // after the plane load and before any downstream kernel
                // that reads d_plane_*_ptrs. Use the default stream as a
                // synchronous barrier since this is infrequent and a
                // single 8-byte copy.
                cudaMemcpy(
                    (uint8_t *)e.d_chunk_qw_ptrs + (size_t)p * sizeof(void *),
                    &qw_p, sizeof(void *), cudaMemcpyHostToDevice);
                cudaMemcpy(
                    (uint8_t *)e.d_chunk_alpha_ptrs + (size_t)p * sizeof(void *),
                    &alpha_p, sizeof(void *), cudaMemcpyHostToDevice);
            }
        }
    }

    dev.M                  = host.n;
    dev.K                  = host.padded_m;
    dev.n_chunks                  = host.n_chunks;
    dev.K_groups           = host.K_groups;
    dev.group_size         = host.group_size;
    dev.qw_bytes_per_chunk = host.qw_bytes_per_chunk;

    // move_chunk does NOT auto-pin. Schedulers that want "keep
    // resident forever" (Eager / Lazy / Prefetch) call pool.pin()
    // explicitly after the move. Budgeted leaves chunks unpinned so
    // the VRAM pool's LRU can evict under a tight cap.
    return h.ready_event;
}

bool StreamllmRuntime::chunk_matmul(const std::string & wid,
                                     const std::vector<int> & chunks,
                                     const void * X_fp16, void * Y_fp16,
                                     int n_tokens,
                                     size_t x_stride_bytes,
                                     size_t y_stride_bytes,
                                     StreamHandle compute_stream) {
    auto it = entries_.find(wid);
    if (it == entries_.end()) return false;
    const Entry &                entry = it->second;
    const UpstreamLayoutDevice & L     = entry.dev;
    if (L.q_bias_fp16 == nullptr) return false;

    // Wait on pending H2D for every chunk we're about to read (incl.
    // q_bias if present in the list; no-op if already synced).
    for (int cid : chunks) {
        pool_->wait_on_stream(wid, cid, compute_stream);
    }

    // Build q_weight / alpha pointer arrays from the data chunks in
    // ``chunks``. Each data chunk is packed as [signs | alpha]; the
    // kernel wants separate base pointers, so we split with the
    // precomputed byte offset.
    //
    // Fast path: chunks is a plane-prefix [PLANE_0, PLANE_1, ..., PLANE_{P-1}]
    // in order (the shape every current scheduler emits). In that case
    // the per-tensor device-side ptr arrays populated by move_chunk are
    // directly the kernel's argument — no per-call H2D memcpy needed.
    const void * qw_ptrs[kMaxChunksPerTensor]    = {};
    const void * alpha_ptrs[kMaxChunksPerTensor] = {};
    int  precision   = 0;
    bool prefix_plan = true;  // chunks consume PLANE_0..PLANE_{precision-1} in order
    for (int cid : chunks) {
        if (!cid_is_chunk(cid)) continue;
        int p = cid_chunk_index(cid);
        if (p < 0 || p >= kMaxChunksPerTensor) continue;
        const void * chunk_ptr = L.chunk_ptrs[p];
        if (chunk_ptr == nullptr) return false;
        if (p != precision) prefix_plan = false;
        qw_ptrs[precision]    = chunk_ptr;
        alpha_ptrs[precision] = static_cast<const uint8_t *>(chunk_ptr) +
                                L.qw_bytes_per_chunk;
        ++precision;
        if (precision >= kMaxChunksPerTensor) break;
    }
    if (precision == 0) return false;

    const void * const * d_qw_dev =
        (prefix_plan && entry.d_chunk_qw_ptrs)    ? (const void * const *) entry.d_chunk_qw_ptrs    : nullptr;
    const void * const * d_a_dev  =
        (prefix_plan && entry.d_chunk_alpha_ptrs) ? (const void * const *) entry.d_chunk_alpha_ptrs : nullptr;

    const uint8_t * x_base = static_cast<const uint8_t *>(X_fp16);
    uint8_t       * y_base = static_cast<uint8_t *>(Y_fp16);
    for (int t = 0; t < n_tokens; ++t) {
        const void * x_t = x_base + (size_t)t * x_stride_bytes;
        void       * y_t = y_base + (size_t)t * y_stride_bytes;
        naver_gemv_launch(
            x_t, y_t,
            qw_ptrs, alpha_ptrs, L.q_bias_fp16,
            L.M, L.K, precision, L.group_size, compute_stream,
            &gemv_scratch_, d_qw_dev, d_a_dev);
    }
    return true;
}

bool StreamllmRuntime::chunk_matmul_batched(const std::string & wid,
                                             const std::vector<int> & chunks,
                                             const void * X_fp16, void * Y_fp16,
                                             int n_tokens,
                                             void * w_scratch_f16,
                                             StreamHandle compute_stream) {
    auto it = entries_.find(wid);
    if (it == entries_.end()) return false;
    const UpstreamLayoutDevice & L = it->second.dev;
    if (L.q_bias_fp16 == nullptr) return false;

    for (int cid : chunks) {
        pool_->wait_on_stream(wid, cid, compute_stream);
    }

    const void * qw_ptrs[kMaxChunksPerTensor]    = {};
    const void * alpha_ptrs[kMaxChunksPerTensor] = {};
    int precision = 0;
    for (int cid : chunks) {
        if (!cid_is_chunk(cid)) continue;
        int p = cid_chunk_index(cid);
        if (p < 0 || p >= kMaxChunksPerTensor) continue;
        const void * chunk_ptr = L.chunk_ptrs[p];
        if (chunk_ptr == nullptr) return false;
        qw_ptrs[precision]    = chunk_ptr;
        alpha_ptrs[precision] = static_cast<const uint8_t *>(chunk_ptr) +
                                L.qw_bytes_per_chunk;
        ++precision;
        if (precision >= kMaxChunksPerTensor) break;
    }
    if (precision == 0) return false;

    // Backend selection:
    //   default                       → dequant_planes + cublasGemmEx (M12).
    //   STREAMLLM_BATCHED_BACKEND=fused → fused chunked LUT-GEMM (M13).
    //
    // The fused path preserves the chunked-matmul abstraction (no dense
    // W materialisation), but at N ≈ 2048 it measured ~5× slower than
    // cuBLAS because:
    //   * cuBLAS uses tensor cores (330 TF fp16) vs our CUDA cores (50 TF).
    //   * Cross-K-tile atomicAdd contention on the output grid.
    //   * Shared-memory footprint caps N_TILE and hurts occupancy.
    // Default stays cuBLAS until the fused kernel is tuned for small-N
    // regimes where its DRAM-BW win pays off.
    const char * backend = std::getenv("STREAMLLM_BATCHED_BACKEND");
    const bool use_fused = (backend != nullptr && std::strcmp(backend, "fused") == 0);

    if (use_fused) {
        naver_gemm_launch(
            X_fp16, Y_fp16,
            qw_ptrs, alpha_ptrs, L.q_bias_fp16,
            L.M, L.K, n_tokens, precision, L.group_size, compute_stream);
        return true;
    }

    if (w_scratch_f16 == nullptr) return false;
    launch_dequant_planes_f16(
        qw_ptrs, alpha_ptrs, L.q_bias_fp16,
        w_scratch_f16,
        L.M, L.K, precision, L.group_size, compute_stream);
    return batched_gemm_f16(
        w_scratch_f16, X_fp16, Y_fp16,
        L.M, L.K, n_tokens, compute_stream);
}

} // namespace streamllm_ext
