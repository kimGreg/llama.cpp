// streamllm-ext / decoder / anybcq — per-plane device pointer layout.
//
// AnyBCQ stores each plane as one contiguous (signs | alpha) blob in
// the VRAM pool. The NAVER GEMV kernel wants two **arrays of pointers**
// — one per plane index — addressable on the device, so the launcher
// can index them directly without a per-call H2D memcpy.
//
// These two arrays are owned per-managed-tensor (one pair per Entry in
// the runtime). Their lifecycle is:
//
//   - allocated at install-time inside ``register_layout`` via
//     ``alloc_per_plane_arrays``,
//   - mutated each time ``move_chunk`` lands a chunk (the relevant
//     plane index gets its qw / alpha base pointers written) via
//     ``update_per_plane_after_load``,
//   - freed at runtime teardown via ``free_per_plane_arrays``.
//
// All three operations are AnyBCQ-specific (the "qw_bytes_per_chunk"
// offset between the qw and alpha base addresses is a property of the
// AnyBCQ chunk layout, not of the framework). They live here, not in
// core/runtime, so adding a second decoder doesn't drag AnyBCQ
// bookkeeping into framework files.

#pragma once

#include "vram_pool.h"   // StreamHandle

#include <cstddef>

namespace streamllm_ext { namespace anybcq {

// Allocate the two device-side pointer arrays of length ``max_planes``,
// zero-initialised. Called once per managed tensor at install time.
// Both ``d_qw`` and ``d_alpha`` are out-params and must be nullptr on
// entry; the function ignores existing non-null values to avoid leaks.
// Throws on cudaMalloc failure.
void alloc_per_plane_arrays(void **& d_qw, void **& d_alpha,
                            int max_planes);

// Free the two device-side pointer arrays and reset both to nullptr.
// Safe to call with already-null arguments. Called per Entry at
// runtime teardown.
void free_per_plane_arrays(void *& d_qw, void *& d_alpha);

// Update the per-plane device-side pointer arrays after a chunk lands
// in the VRAM pool. Writes both ``d_qw[plane]`` (= chunk_device_ptr)
// and ``d_alpha[plane]`` (= chunk_device_ptr + qw_bytes_per_chunk).
//
// Two variants:
//
//   * ``update_per_plane_after_load`` does two synchronous 8-byte
//     H2Ds on the default stream — simple but blocks the calling
//     worker for ~13 µs/call. At MoE scale this dominates wall time.
//
//   * ``update_per_plane_after_load_async`` launches a 1-thread
//     kernel on the supplied stream that writes both pointers in a
//     single device-side store pair, then returns. Caller is
//     responsible for re-recording the chunk's ready event on the
//     same stream so ``wait_on_stream`` correctly orders downstream
//     compute behind the pointer-table update (without that
//     re-record, ``ready_event`` only captures the H2D bytes; a
//     kernel launched after the event would write the table after
//     ``wait_on_stream`` succeeds — racing the matmul kernel).
void update_per_plane_after_load(void ** d_qw, void ** d_alpha,
                                 int plane,
                                 const void * chunk_device_ptr,
                                 size_t qw_bytes_per_chunk);
void update_per_plane_after_load_async(void ** d_qw, void ** d_alpha,
                                       int plane,
                                       const void * chunk_device_ptr,
                                       size_t qw_bytes_per_chunk,
                                       StreamHandle stream);

// Zero the per-plane pointer slot after the underlying VRAM chunk has
// been evicted from the pool. Required because pool::evict() only
// frees the slot — without clearing d_qw[plane]/d_alpha[plane], the
// kernel will read the now-stale pointer (the slot may be reused by
// another chunk's load), corrupting outputs. Writes happen on the
// default stream synchronously: eviction is the slow path, so
// ordering simplicity is worth more than the µs.
void clear_per_plane_after_evict(void ** d_qw, void ** d_alpha,
                                 int plane);

}}  // namespace streamllm_ext::anybcq
