// streamllm-ext / core — ChunkedTensor abstract base class.
//
// One of the framework's two encoder/architecture-blind ABCs (the
// other is ChunkedComputation in computation.h). A ChunkedTensor
// represents one tensor whose payload is split into independently
// loadable chunks. The pool (move_chunk + residency tracking) and
// the scheduler operate on ChunkedTensor pointers; concrete
// subclasses live in ``decoder/``.
//
// The framework knows about a chunked tensor:
//   - how many chunks it has,
//   - per-chunk byte sizes (disk + kernel-format),
//   - how to transform one chunk's disk bytes into kernel-format
//     bytes (encoder-private),
//   - what to do after a chunk lands in / is evicted from VRAM
//     (encoder-private — usually pointer-table updates the kernel
//     reads).
//
// The framework does NOT know:
//   - the byte format of any chunk,
//   - which encoder produced the data,
//   - what the kernel looks like.
//
// Adding a new encoding (different bit-pack, different α/β layout,
// quantized KV, etc.) means subclassing ChunkedTensor; no framework
// edits needed.

#pragma once

#include "vram_pool.h"   // StreamHandle

#include <cstdint>
#include <cstddef>
#include <string>

namespace streamllm_ext {

// Per-chunk (per-tensor) ABC. The pool calls ``disk_to_kernel`` on
// the SSD-stream path and ``after_load`` once the kernel-format
// bytes have been uploaded to VRAM. The scheduler / kernel calls
// ``after_evict`` when the chunk is removed from the pool so any
// pointer-table the kernel reads can be cleared.
class ChunkedTensor {
public:
    virtual ~ChunkedTensor() = default;

    // Stable identifier — typically the gguf tensor name. Used as
    // the key in the pool's residency map. Must be unique within
    // a runtime instance.
    virtual const std::string & id() const = 0;

    // Chunk count.
    virtual int n_chunks() const = 0;

    // Per-chunk byte sizes. ``disk_bytes`` = bytes to pread; may be
    // variable per chunk (any-prec). ``kernel_bytes`` = bytes the
    // pool allocates in VRAM after transform.
    virtual size_t disk_bytes  (int chunk_idx) const = 0;
    virtual size_t kernel_bytes(int chunk_idx) const = 0;

    // Encoder-private transform: convert one chunk's disk bytes
    // into the kernel-format VRAM payload. Caller passes a host
    // scratch buffer of size ``kernel_bytes(chunk_idx)`` for output.
    virtual void disk_to_kernel(int chunk_idx,
                                 const uint8_t * disk_in,
                                 uint8_t       * kernel_out) const = 0;

    // Encoder-private post-load hook: chunk has just been uploaded
    // to ``device_ptr``; subclass typically writes per-plane
    // pointer-table entries the kernel will read. ``stream`` is the
    // pool's copy stream (the upload completed on it); the subclass
    // should enqueue any pointer-table writes on this same stream.
    virtual void after_load (int chunk_idx,
                              const void * device_ptr,
                              StreamHandle stream)        = 0;

    // Encoder-private post-evict hook: the chunk's VRAM slot has
    // been freed by the pool; subclass clears any pointer-table
    // entry that referenced it so the kernel doesn't read a stale
    // pointer if the slot is later reused.
    virtual void after_evict(int chunk_idx)               = 0;
};

}  // namespace streamllm_ext
