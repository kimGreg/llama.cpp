// streamllm-ext — raw-pointer launcher for NAVER's LUT-GEMV.

#pragma once

#include <cstddef>

struct CUstream_st;

namespace streamllm_ext {

// NAVER LUT-GEMV per-call precision cap — upper bound on the number of
// AnyBCQ data chunks this kernel can consume in one launch. Independently
// set here (algorithm layer) and in runtime.h (framework layer); both
// are 16 and a static_assert in runtime.h ties them.
constexpr int kNaverMaxPrecision = 16;

// Pre-allocated scratch buffers the GEMV launcher reuses across calls.
// Without a scratch, each call does 3× cudaMallocAsync + 3× cudaFreeAsync,
// which is ~10 μs of per-call overhead — non-trivial at P<8 where the
// kernel itself is ~50 μs. Owned by StreamllmRuntime; pass nullptr for
// standalone tests / microbench.
struct NaverKernelScratch {
    void **  d_qw_ptrs      = nullptr;   // device [kMaxChunksPerTensor × void*]
    void **  d_alpha_ptrs   = nullptr;   // device [kMaxChunksPerTensor × void*]
    float *  d_acc_f32      = nullptr;   // device [acc_capacity_elems × float]
    size_t   acc_capacity_elems = 0;
    int      device         = 0;

    bool init(int device, size_t acc_capacity_elems);
    void destroy();
    // Grow d_acc_f32 on demand. Returns false on CUDA failure.
    bool ensure_acc(size_t needed_elems);
};

// Y = X @ decode(q_weight_planes[:P], alpha_planes[:P], q_bias).T, fp16.
//
// Pointers must all be valid on the current CUDA device.
//   d_input_fp16       (K,)                  fp16 activations
//   d_output_fp16      (M,)                  fp16 output (caller's dst)
//   d_q_weight_planes  (precision,)          array of device pointers;
//                                            each points at a
//                                            [K/32, M] uint32 sign pack.
//   d_alpha_planes     (precision,)          array of device pointers;
//                                            each points at a
//                                            [K_groups, M] fp16 α.
//   d_q_bias_fp16      (K_groups, M) fp16    shared across planes
//
// ``precision`` is the number of planes consumed and ≤ kMaxChunksPerTensor. M8
// allows precision < P_total so the kernel runs at runtime-tunable bpw.
// Constraints: group_size % 64 == 0, K % 64 == 0. Throws on violation.
// If ``d_q_weight_planes_device`` and ``d_alpha_planes_device`` are both
// non-null, they are used as the plane-pointer arrays directly (must be
// device-resident, contain at least ``precision`` valid entries) and the
// per-call H2D memcpy of the pointer arrays is skipped. In that case
// the host-pointer arguments may be nullptr.
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

// Y = X @ decode(q_weight_planes, alpha_planes, q_bias).T for a batch
// of N input vectors — native chunked LUT-GEMM (no dense W
// materialisation). Layout convention matches ggml:
//
//   d_input_fp16    : [K, N]  col-major (K inner, contiguous)
//   d_output_fp16   : [M, N]  col-major (M inner, contiguous)
//
// Precision dial and plane-pointer contract identical to naver_gemv_launch.
// For N == 1 the per-token GEMV path (naver_gemv_launch) is strictly
// faster — use this only when N > 1.
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

} // namespace streamllm_ext
