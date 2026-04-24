// streamllm-ext — runtime singleton. Owns a VRAM pool + a Scheduler
// and exposes two primitives to the mul_mat hook:
//
//   move_chunk(wid, cid, src, dst) -> event       // async tier move
//   chunk_matmul(wid, chunks, X, Y, ...)          // launch nqmv_bias
//
// Chunk vocabulary (a "chunk" is the smallest unit the pool moves):
//
//   Q_BIAS  = 0           — shared fixed-meta chunk (per tensor)
//   PLANE_i = kCidChunkBase + i   — plane i's (signs + α) blob
//
// ``chunks`` passed to chunk_matmul is a list of these cids. Precision
// = number of PLANE cids in the list. Q_BIAS is always consumed when
// present (the NAVER kernel always adds q_bias × sum_lut_255).

#pragma once

#include "naver_gemv.h"
#include "scheduler.h"
#include "stream_reader.h"
#include "upstream_layout.h"
#include "vram_pool.h"

#include <memory>
#include <string>
#include <unordered_map>
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

// Keep the framework's chunks-per-tensor cap aligned with the NAVER
// kernel's per-launch precision cap. If one moves, the other must.
static_assert(kMaxChunksPerTensor == kNaverMaxPrecision,
              "framework / kernel chunk cap mismatch");

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

    // --- primitives -----------------------------------------------------

    // ``compute_stream`` is only consulted when CUDA graphs are active:
    // if it's in an active capture, the pool's copy stream is forked
    // into the same capture session so the kernel's downstream wait on
    // the H2D ready event is a legal intra-capture dependency. Safe to
    // pass nullptr (install-time bulk uploads, outside any capture).
    EventHandle move_chunk(const std::string & wid, int cid,
                           Tier src, Tier dst,
                           StreamHandle compute_stream = nullptr);

    // Run nqmv_bias over the chunks in ``chunks``. Precision =
    // number of PLANE cids present. Q_BIAS must be resident.
    bool chunk_matmul(const std::string & wid,
                      const std::vector<int> & chunks,
                      const void * X_fp16, void * Y_fp16,
                      int n_tokens,
                      size_t x_stride_bytes, size_t y_stride_bytes,
                      StreamHandle compute_stream);

    // Batched prefill fast path. Reconstructs W_f16[M, K] into the
    // caller-provided scratch via launch_dequant_planes_f16 using the
    // planes named in ``chunks`` (precision = count of PLANE cids),
    // then runs cublas F16 GEMM: Y[M, N] = W @ X[K, N].
    //
    // X and Y are tightly packed:
    //   X: col-major [K, n_tokens], contiguous, fp16
    //   Y: col-major [M, n_tokens], contiguous, fp16
    // i.e. the same layout ggml uses when src1/dst have ne = {K, N} / {M, N}.
    //
    // ``w_scratch_f16`` must be at least M*K*2 bytes. Caller owns it
    // (typically a per-stream scratch in runtime_hook.cpp).
    bool chunk_matmul_batched(const std::string & wid,
                              const std::vector<int> & chunks,
                              const void * X_fp16, void * Y_fp16,
                              int n_tokens,
                              void * w_scratch_f16,
                              StreamHandle compute_stream);

    // --- accessors ------------------------------------------------------

    const UpstreamLayoutDevice * layout(const std::string & wid) const;

    const VramChunkPool & pool() const { return *pool_; }
    VramChunkPool & pool() { return *pool_; }
    const Scheduler & scheduler() const { return *scheduler_; }
    Scheduler & scheduler() { return *scheduler_; }

    // Lifecycle helpers for schedulers.
    void register_layout(const std::string & wid,
                         UpstreamLayoutHost host,
                         UpstreamLayoutDevice dev);
    UpstreamLayoutDevice * mutable_layout(const std::string & wid);
    const UpstreamLayoutHost * host_layout(const std::string & wid) const;

private:
    struct Entry {
        UpstreamLayoutHost   host;
        UpstreamLayoutDevice dev;
        // Device-side cache of the per-plane pointer arrays, in plane-
        // index order: d_plane_{qw,alpha}_ptrs[p] matches plane p's qw
        // / alpha base. Kept in sync by move_chunk. When chunk_matmul
        // receives a prefix-style chunks list (common case across all
        // current schedulers), it uses these directly and skips the
        // per-call H2D memcpy of the pointer arrays.
        void ** d_chunk_qw_ptrs    = nullptr;  // device [kMaxChunksPerTensor × void*]
        void ** d_chunk_alpha_ptrs = nullptr;
    };

    std::unique_ptr<VramChunkPool> pool_;
    std::unique_ptr<Scheduler>     scheduler_;
    std::unordered_map<std::string, Entry> entries_;
    NaverKernelScratch             gemv_scratch_;  // decode hot-path scratch
    bool installed_ = false;
};

} // namespace streamllm_ext
