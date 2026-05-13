// streamllm-ext / decoder / anybcq — public GEMV interface.
//
// Single-token LUT-GEMV (decode path) plus the per-plane device-pointer
// table the runtime updates whenever a chunk lands in VRAM.
//
// The GEMV kernel reads
//
//     w · x ≈ β + Σ α_i × (sign_i · x)   for i ∈ [0, precision)
//
// from per-plane pointer tables. ``update_per_plane_after_load*`` and
// ``update_anyprec_after_load*`` populate those tables for shortcut
// and any-prec layouts respectively — encoder-specific staging only.
//
// MoE-fused dispatch is NOT here. Fusion across MoE experts is an
// architecture-specific optimisation; see ``qwen3/qwen3_moe_fused.h``.
//
// Callers in core/, qwen3/, and tests include only this header.

#pragma once

#include "vram_pool.h"   // StreamHandle

#include <cstddef>
#include <cstdint>

struct CUstream_st;

namespace streamllm_ext {

// =====================================================================
// Single-token LUT-GEMV (decode hot path).
// =====================================================================

// Per-call precision cap. Independently set here (algorithm layer) and
// in runtime.h (framework layer); both are 16 and a static_assert in
// runtime.h ties them.
constexpr int kNaverMaxPrecision = 16;

// Pre-allocated scratch buffers the GEMV launcher reuses across calls.
// Without a scratch, each call does 3× cudaMallocAsync + 3× cudaFreeAsync,
// ~10 µs of per-call overhead — non-trivial at P<8 where the kernel
// itself is ~50 µs. Owned by StreamllmRuntime; pass nullptr for
// standalone tests / microbench.
struct NaverKernelScratch {
    void **  d_qw_ptrs      = nullptr;   // device [kNaverMaxPrecision × void*]
    void **  d_alpha_ptrs   = nullptr;   // device [kNaverMaxPrecision × void*]
    float *  d_acc_f32      = nullptr;   // device [acc_capacity_elems × float]
    size_t   acc_capacity_elems = 0;
    int      device         = 0;

    bool init(int device, size_t acc_capacity_elems);
    void destroy();
    bool ensure_acc(size_t needed_elems);
};

// Y = X @ decode(q_weight_planes[:P], alpha_planes[:P], q_bias).T, fp16.
// See header comment above for the AnyBCQ math.
//   d_input_fp16       (K,)             fp16 activations
//   d_output_fp16      (M,)             fp16 output (caller's dst)
//   d_q_weight_planes  (precision,)     array of device pointers, each
//                                       at a [K/32, M] uint32 sign pack.
//   d_alpha_planes     (precision,)     array of device pointers, each
//                                       at a [K_groups, M] fp16 α.
//   d_q_bias_fp16      (K_groups, M)    fp16, shared across planes
// ``precision`` ≤ kNaverMaxPrecision. Setting precision < P_total runs
// the kernel at runtime-tunable bpw. group_size % 64 == 0, K % 64 == 0.
// If d_q_weight_planes_device / d_alpha_planes_device are non-null they
// are used directly (must be device-resident, ≥ precision entries) and
// the per-call H2D copy of the pointer arrays is skipped — the host-
// pointer args may then be nullptr.
void naver_gemv_launch(
    const void *        d_input_fp16,
    void *              d_output_fp16,
    const void * const *d_q_weight_planes,
    const void * const *d_alpha_planes,
    const void *        d_q_bias_fp16,
    int                 M,
    int                 K,
    int                 precision,
    int                 group_size,
    CUstream_st *       stream = nullptr,
    NaverKernelScratch *scratch = nullptr,
    const void * const *d_q_weight_planes_device = nullptr,
    const void * const *d_alpha_planes_device    = nullptr);

// Native chunked LUT-GEMM (no dense W materialisation) for N input
// vectors. Layout matches ggml: input/output col-major with K/M inner.
// For N == 1 prefer naver_gemv_launch — this kernel is faster only for
// N > 1.
void naver_gemm_launch(
    const void *        d_input_fp16,     // [K × N]
    void *              d_output_fp16,    // [M × N]
    const void * const *d_q_weight_planes,
    const void * const *d_alpha_planes,
    const void *        d_q_bias_fp16,
    int                 M,
    int                 K,
    int                 N,
    int                 precision,
    int                 group_size,
    CUstream_st *       stream = nullptr);


namespace anybcq {

// =====================================================================
// Per-plane pointer-table updates (encoder-specific staging).
//
// Each managed tensor owns two device-side pointer arrays of length
// kNaverMaxPrecision: ``d_qw[]`` and ``d_alpha[]``. When a chunk lands
// in VRAM the runtime calls one of the helpers below to write the
// relevant entries.
// =====================================================================

// Allocate / free the per-Entry pointer arrays.
void alloc_per_plane_arrays(void **& d_qw, void **& d_alpha,
                            int max_planes);
void free_per_plane_arrays(void *& d_qw, void *& d_alpha);

// Shortcut layout: one plane per chunk. ``d_qw[plane]`` ← chunk_device_ptr
// and ``d_alpha[plane]`` ← chunk_device_ptr + qw_bytes_per_chunk.
// The async variant launches a 1-thread kernel and the caller must
// re-record the chunk's ready event on ``stream`` so wait_on_stream
// orders compute behind the table update.
void update_per_plane_after_load(void ** d_qw, void ** d_alpha,
                                 int plane,
                                 const void * chunk_device_ptr,
                                 size_t qw_bytes_per_chunk);
void update_per_plane_after_load_async(void ** d_qw, void ** d_alpha,
                                       int plane,
                                       const void * chunk_device_ptr,
                                       size_t qw_bytes_per_chunk,
                                       StreamHandle stream);

// Zero a freed slot's per-plane pointers after pool eviction so the
// kernel doesn't read a stale pointer (the slot may be reused by
// another chunk under tight VRAM caps).
//
// IMPORTANT: this is the inverse of ``update_per_plane_after_load`` /
// ``update_anyprec_after_load`` — it MUST clear every plane those
// writers populated. For shortcut, that is one plane per chunk
// (chunk_idx == plane). For any-prec chunk 0, ``base_p`` planes
// ([0, base_p)) — leaving planes 1..base_p-1 dangling makes a
// freed-and-reused slot readable through a stale d_qw[plane>=1]
// entry and corrupts kernel output (the T5 race bug, 2026-05-13).
//
// The async variant runs as a 1-thread kernel on the pool's
// copy_stream — preferred mid-run. The sync variant is for install
// teardown where no copy_stream is available.
void clear_per_plane_after_evict(void ** d_qw, void ** d_alpha,
                                 int plane_first, int n_planes);
void clear_per_plane_after_evict_async(void ** d_qw, void ** d_alpha,
                                       int plane_first, int n_planes,
                                       StreamHandle stream);

// Any-prec layout: chunk i packs [n_planes signs | precision_at_chunk α
// blocks | β]. Writes the n_planes sign pointers at [plane_idx_first..]
// and ALL precision_at_chunk α pointers (this chunk holds the optimal
// α^(p) for its precision tier; kernel running at p reads alpha[0..p-1]).
// β is written into a separate single-pointer slot ``d_qbias_slot[0]``;
// pass nullptr to leave it untouched.
void update_anyprec_after_load_async(
    void **      d_qw_ptrs,
    void **      d_alpha_ptrs,
    void **      d_qbias_slot,
    int          plane_idx_first,
    int          n_planes_this_chunk,
    int          precision_at_chunk,
    const void * chunk_device_ptr,
    size_t       qw_bytes_per_chunk,
    size_t       alpha_bytes_per_chunk,
    size_t       q_bias_bytes_per_chunk,
    StreamHandle stream);
void update_anyprec_after_load(
    void **      d_qw_ptrs,
    void **      d_alpha_ptrs,
    void **      d_qbias_slot,
    int          plane_idx_first,
    int          n_planes_this_chunk,
    int          precision_at_chunk,
    const void * chunk_device_ptr,
    size_t       qw_bytes_per_chunk,
    size_t       alpha_bytes_per_chunk,
    size_t       q_bias_bytes_per_chunk);

}  // namespace anybcq
}  // namespace streamllm_ext
