// streamllm-ext / core — encoder-agnostic chunk interface.
//
// **The framework's view of one managed tensor's chunks.** Everything that
// is encoder-specific (AnyBCQ vs scaled-LR vs JPEG2000 etc.) lives behind
// the ``ChunkCodec`` virtual surface. Framework code (runtime, scheduler)
// must talk to chunks ONLY through this header.
//
// Three boundaries crossed by every chunk in flight:
//
//   1. Disk      — bytes on the GGUF file at a known offset, owned by the
//                  encoder's format spec.
//   2. Kernel    — bytes in a VRAM blob the encoder's kernel reads.
//                  Layout is encoder-defined; the framework only knows
//                  the size and where it lives in the pool.
//   3. Dispatch  — kernel call args (pointer arrays etc.) the encoder
//                  builds from the currently-resident chunks.
//
// ChunkCodec is the per-managed-tensor object that owns the encoder's
// view across all three boundaries.

#pragma once

#include "vram_pool.h"   // StreamHandle

#include <cstddef>
#include <cstdint>
#include <vector>

namespace streamllm_ext {

// ─── Encoder-agnostic per-chunk metadata ─────────────────────────────
//
// Computed once at install time. The framework reads these to decide
// pool sizing, file pread offsets, and chunk-level eviction policy.
// Encoder-specific fields are NOT here — they live as private state
// inside the codec implementation.

struct ChunkInfo {
    int     cid;             // unique within tensor (0..n_chunks-1)
    int64_t disk_offset;     // absolute byte offset in the GGUF file
    size_t  disk_bytes;      // bytes on disk for this chunk
    size_t  kernel_bytes;    // bytes in the kernel-format VRAM blob
};

// ─── Encoder behavior surface ────────────────────────────────────────
//
// One ChunkCodec instance per managed tensor. The runtime stores it in
// its per-tensor Entry and calls into it whenever a chunk is read,
// landed, evicted, or used by the kernel.
//
// Lifetime: created at install time by the encoder-specific factory
// (e.g. ``make_anybcq_codec``); owned by the runtime; destroyed at
// runtime teardown. All hot-path methods are thread-safe with the
// runtime's per-Entry mutex held by the caller.

class ChunkCodec {
public:
    virtual ~ChunkCodec() = default;

    // ─── Static surface for the framework ──────────────────────────
    // (called only at install / pool-size accounting time)

    virtual int                  n_chunks() const                      = 0;
    virtual const ChunkInfo &    chunk_info(int chunk_idx) const       = 0;
    virtual size_t               max_kernel_chunk_bytes() const        = 0;

    // ─── Disk → kernel transform ──────────────────────────────────
    //
    // Read at install time AND on the SSD-stream HOT path. ``disk_in``
    // size = ``chunk_info(chunk_idx).disk_bytes``. ``kernel_out`` is
    // a writable buffer of ``chunk_info(chunk_idx).kernel_bytes``.

    virtual void disk_to_kernel(int chunk_idx,
                                 const uint8_t * disk_in,
                                 uint8_t       * kernel_out) const     = 0;

    // ─── Activation hooks ──────────────────────────────────────────
    //
    // Called by ``move_chunk`` once a chunk is resident in VRAM at
    // ``dev_ptr``. The codec writes any encoder-specific pointer-table
    // updates (per-plane sign pointers for AnyBCQ shortcut, full α/β
    // override for any-prec, etc.) to its private device-side state.
    //
    // ``stream`` is the pool's copy stream — the codec's own kernels
    // launched here are ordered before any compute kernel that calls
    // ``make_dispatch_args`` and reads the codec's pointer tables.

    virtual void on_chunk_loaded (int chunk_idx,
                                   const void * dev_ptr,
                                   StreamHandle stream)                = 0;
    virtual void on_chunk_evicted(int chunk_idx)                       = 0;

    // ─── Dispatch arg construction ────────────────────────────────
    //
    // Called from ``chunk_matmul``. ``active`` is the cid list the
    // scheduler chose for this dispatch (a subset of resident chunks).
    // The codec returns an opaque-to-framework pointer to a struct the
    // encoder's kernel launcher knows how to consume. The lifetime of
    // the returned pointer extends until the next ``make_dispatch_args``
    // call on this codec — callers must launch the kernel before
    // calling again.

    virtual void * make_dispatch_args(const std::vector<int> & active) = 0;

    // ─── MoE: per-canonical-tensor expert table ───────────────────
    //
    // For fused MoE kernels: gather the per-expert dispatch state into
    // a single device-side table. Default returns nullptr meaning
    // "this codec has no fused MoE path; caller falls back to per-
    // expert ``make_dispatch_args`` invocations".
    //
    // ``per_expert`` is the list of codecs (one per routed expert) for
    // the same canonical name; the table aggregates their state. The
    // returned pointer's lifetime is the same as the calling canonical
    // (built once at install, mutated in-place when chunks land).

    virtual void * make_moe_dispatch_args(
        const std::vector<ChunkCodec *> & /*per_expert*/) {
        return nullptr;
    }

    // ─── Diagnostics ──────────────────────────────────────────────

    virtual const char * name() const = 0;  // e.g. "anybcq"
};

}  // namespace streamllm_ext
