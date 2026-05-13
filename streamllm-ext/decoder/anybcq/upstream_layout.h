// streamllm-ext / decoder / anybcq — host layout builder (dispatcher +
// any-prec specifics).
//
// The struct ``UpstreamLayoutHost`` is encoder-agnostic — both the
// shortcut and any-prec parsers fill it. The byte layout each parser
// reads from disk is encoder-specific:
//
//   SHORTCUT (any_prec = 0, default):  see decoder/shortcut_anybcq/
//   ANY-PREC (any_prec = 1):
//     [32 B header] (no fixed_meta β)
//     [(target − base + 1) chunks, each carrying P=base+i:
//        [plane signs: base_p planes for chunk 0, 1 plane for chunks 1..N]
//        [α^(P=p): d1 × p × α_dtype, row-major (d1 outer, plane inner)]
//        [β^(P=p): d1 × β_dtype]]
//
// Each any-prec chunk is self-contained at its precision level: the
// runtime uses the highest-loaded chunk's α + β; sign buffers from
// every loaded chunk are concatenated into the kernel's per-plane sign
// pointer table. Cannot prefix-truncate this layout — choose precision
// via the scheduler's "highest-active chunk" hook.
//
// Per managed tensor, the framework sees two chunk kinds:
//   - cid = kCidQBias       the install-pinned meta blob (shortcut: β).
//   - cid = kCidChunkBase+i a data chunk i.

#pragma once

#include "stream_reader.h"
#include "vram_pool.h"   // StreamHandle (used by encoder callbacks)

#include <cstdint>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace streamllm_ext {
class ChunkedTensor;
namespace anybcq { class AnyBCQFamilyTensor; }
}  // namespace streamllm_ext

namespace streamllm_ext {

// Forward decl so the callback typedefs can reference the struct.
struct UpstreamLayoutHost;

// Encoder-registered callbacks that core's move_chunk path invokes
// to do the encoder-specific transform + per-plane pointer-table
// staging. Set by ``build_upstream_layout_*`` at install time, so
// core never has to ``#include`` decoder headers or branch on
// ``host.any_precision``.
//
// ``ChunkDiskToKernelFn``: convert one chunk's disk bytes into the
// kernel-format VRAM payload (signs+α+β packed per the encoder).
using ChunkDiskToKernelFn = void (*)(
    const UpstreamLayoutHost & host,
    int          chunk_idx,
    const uint8_t * disk_in,
    uint8_t      * kernel_out);

// ``ChunkAfterLoadFn``: after the chunk lands in VRAM, write the
// per-plane device-pointer-table entries the kernel reads.
//   d_qbias_slot is nullable — used only by encoders that store β
//   per chunk (any-prec). Shortcut passes nullptr.
using ChunkAfterLoadFn = void (*)(
    const UpstreamLayoutHost & host,
    int           chunk_idx,
    const void *  chunk_device_ptr,
    void **       d_qw_ptrs,
    void **       d_alpha_ptrs,
    void **       d_qbias_slot,
    StreamHandle  stream);

// ``ChunkAfterEvictFn``: after the chunk is evicted from VRAM, clear
// the per-plane pointer-table entries so the kernel doesn't read
// stale pointers if the slot is reused.
//
// MUST be the exact inverse of ``ChunkAfterLoadFn`` at the plane
// level — every plane the load wrote, the evict clears.  In
// particular, any-prec chunk 0 owns ``base_p`` planes, so its evict
// MUST clear all of ``[plane_first, plane_first + n_planes)`` — not
// just ``plane_first``.  Shortcut: 1 plane per chunk (chunk_idx ==
// plane_idx, n_planes always 1).
//
//   ``plane_first`` — first plane index this chunk owns.
//                     shortcut: chunk_idx
//                     any-prec: ``host.chunk_planes[chunk_idx].plane_idx_first``
//   ``n_planes``    — number of planes this chunk owns.
//                     shortcut: 1
//                     any-prec: ``host.chunk_planes[chunk_idx].n_planes_this_chunk``
//   ``stream``      — pool's copy stream; clear is enqueued async
//                     so it overlaps with the worker's other
//                     emissions (SSOT §6.1.6 step 6).  May be null
//                     only on teardown.
using ChunkAfterEvictFn = void (*)(
    void **       d_qw_ptrs,
    void **       d_alpha_ptrs,
    int           plane_first,
    int           n_planes,
    StreamHandle  stream);


// Host-side layout for one managed tensor. Fields used by both
// encoders are unmarked; encoder-specific fields are noted.
struct UpstreamLayoutHost {
    std::vector<std::vector<uint8_t>> chunks;    // length = n_chunks
    std::vector<uint16_t>              q_bias;   // [K_groups * n] fp16
                                                  // shortcut: filled from
                                                  // FIXED_META β; any-prec:
                                                  // empty (β lives per chunk)

    size_t qw_bytes_per_chunk    = 0;
    size_t alpha_bytes_per_chunk = 0;
    size_t bytes_per_chunk       = 0;  // shortcut: full chunk size; any-
                                        // prec: max over chunks (pool sizing)
    size_t q_bias_n_elem         = 0;
    size_t q_bias_bytes_per_chunk = 0; // any-prec: K_groups * n * 2

    int32_t n           = 0;
    int32_t padded_m    = 0;
    int32_t n_chunks    = 0;
    int32_t K_groups    = 0;
    int32_t group_size  = 0;

    int32_t disk_alpha_size = 2;
    int32_t disk_beta_size  = 2;
    size_t  disk_bytes_per_chunk = 0;  // shortcut SSD-stream HOT path; any-
                                       // prec stores per-chunk sizes in
                                       // chunk_planes[i].disk_chunk_bytes

    bool any_precision = false;
    int32_t base_precision   = 0;  // any-prec: chunk 0's precision tier
    int32_t target_precision = 0;  // any-prec: = n_chunks-1 + base_precision

    // Any-prec only.
    struct ChunkPlanes {
        int32_t precision_at_chunk;
        int32_t n_planes_this_chunk;
        int32_t plane_idx_first;
        size_t  kernel_chunk_bytes;
        size_t  disk_chunk_bytes;
        size_t  ker_off_signs;
        size_t  ker_off_alpha;
        size_t  ker_off_qbias;
    };
    std::vector<ChunkPlanes> chunk_planes;

    // Encoder-registered callbacks (see typedefs above). ``build_*``
    // sets these so core's move_chunk can invoke them blindly.
    ChunkDiskToKernelFn disk_to_kernel_fn = nullptr;
    ChunkAfterLoadFn    after_load_fn     = nullptr;
    ChunkAfterEvictFn   after_evict_fn    = nullptr;
};

// Top-level dispatcher: reads the v2 stream header (magic / version /
// flag bits) and forwards to either decoder/shortcut_anybcq's parser
// or the any-prec parser below. Throws on bad magic / version /
// per-encoder validation failure.
UpstreamLayoutHost build_upstream_layout_host(
    const TensorLayout & layout,
    const uint8_t * tensor_data,
    uint32_t group_size);

// Same dispatch but wraps the parsed host layout in the appropriate
// concrete ChunkedTensor (anybcq::AnyBCQTensor for any-prec,
// shortcut_anybcq::ShortcutTensor for shortcut). The framework hands
// the runtime ChunkedTensor pointers; this is the entry point that
// produces them.
std::unique_ptr<ChunkedTensor> build_chunked_tensor(
    std::string          wid,
    const TensorLayout & layout,
    const uint8_t * tensor_data,
    uint32_t group_size);

// Wrap an already-parsed UpstreamLayoutHost in the appropriate
// ChunkedTensor subclass (any-prec → anybcq::AnyBCQTensor, shortcut →
// shortcut_anybcq::ShortcutTensor). Used by call sites that obtained
// the parsed layout via build_upstream_layout_host directly (e.g.
// the runtime's register_layout backward-compat path). Returns the
// concrete AnyBCQ-family pointer so the runtime can read the per-
// tensor pointer-table accessors (host(), d_qw_ptrs(), etc.) without
// downcasting.
std::unique_ptr<anybcq::AnyBCQFamilyTensor> wrap_host_in_tensor(
    std::string         wid,
    UpstreamLayoutHost  host);

// Any-prec specific: transform one chunk's disk-format bytes into
// kernel-format. ``disk_in`` size = (n_planes × plane_sign_bytes) +
// (precision × d1 × α_size) + (d1 × β_size). ``kernel_out`` size =
// (n_planes × qw_bytes) + (precision × alpha_bytes) + (q_bias_bytes).
void any_prec_chunk_disk_to_kernel(
    const uint8_t * disk_in,
    uint8_t       * kernel_out,
    int32_t         n,
    int32_t         padded_m,
    int32_t         ng,
    int32_t         n_planes_in_chunk,
    int32_t         precision_at_chunk,
    int32_t         disk_alpha_size,
    int32_t         disk_beta_size);

}  // namespace streamllm_ext
