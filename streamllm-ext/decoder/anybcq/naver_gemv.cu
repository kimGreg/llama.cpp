// streamllm-ext — raw-pointer launcher for NAVER's LUT-GEMV kernel.
//
// Takes arrays of per-plane pointers for q_weight and alpha; the
// kernel picks ``precision`` of them at runtime.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "naver_gemv.h"

// Forward-decls.
// GEMV body is defined in naver_kernel_copy.cu.
__global__ void nqmv_bias_planes(
    const uint32_t * const * q_weight_planes,
    const __half   * const * alpha_planes,
    const __half   *         q_bias,
    const __half   *         input,
    float          *         output,
    const int                M,
    const int                K,
    const int                precision,
    const int                group_size);

// Batched GEMM body lives in naver_gemm_batched.cu.
__global__ void nqmv_bias_planes_batched(
    const uint32_t * const * q_weight_planes,
    const __half   * const * alpha_planes,
    const __half   *         q_bias,
    const __half   *         input,
    float          *         output,
    const int                M,
    const int                K,
    const int                N,
    const int                precision,
    const int                group_size);

constexpr int kNTileSizeBatched = 8;  // matches naver_gemm_batched.cu
constexpr int kBatchedSharedBytes = 64 * 1024;  // 8 × 8 × 256 × sizeof(float)

namespace {

__global__ void cast_f32_to_f16(const float * __restrict__ src,
                                __half * __restrict__ dst,
                                int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

constexpr int kKTileSize  = 64;
constexpr int kMTileSize  = 1024;
constexpr int kNumThreads = 256;

inline void check_cuda(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        throw std::runtime_error(
            std::string("naver_gemv: CUDA error in ") + what + ": " +
            cudaGetErrorString(e));
    }
}

} // anonymous


namespace streamllm_ext {

bool NaverKernelScratch::init(int dev, size_t acc_cap_elems) {
    int prev_dev = -1;
    cudaError_t e = cudaGetDevice(&prev_dev);
    if (e != cudaSuccess) return false;
    if (dev != prev_dev) {
        e = cudaSetDevice(dev);
        if (e != cudaSuccess) return false;
    }
    device = dev;

    e = cudaMalloc(&d_qw_ptrs,    kNaverMaxPrecision * sizeof(void *));
    if (e != cudaSuccess) { destroy(); return false; }
    e = cudaMalloc(&d_alpha_ptrs, kNaverMaxPrecision * sizeof(void *));
    if (e != cudaSuccess) { destroy(); return false; }

    if (acc_cap_elems > 0) {
        e = cudaMalloc(&d_acc_f32, acc_cap_elems * sizeof(float));
        if (e != cudaSuccess) { destroy(); return false; }
        acc_capacity_elems = acc_cap_elems;
    }

    if (dev != prev_dev) cudaSetDevice(prev_dev);
    return true;
}

void NaverKernelScratch::destroy() {
    if (d_qw_ptrs)    { cudaFree(d_qw_ptrs);    d_qw_ptrs    = nullptr; }
    if (d_alpha_ptrs) { cudaFree(d_alpha_ptrs); d_alpha_ptrs = nullptr; }
    if (d_acc_f32)    { cudaFree(d_acc_f32);    d_acc_f32    = nullptr; }
    acc_capacity_elems = 0;
}

bool NaverKernelScratch::ensure_acc(size_t needed_elems) {
    if (needed_elems <= acc_capacity_elems) return true;
    // Grow. Free old, alloc new sized to 1.5× for amortized growth.
    const size_t new_cap = needed_elems + needed_elems / 2;
    float * new_buf = nullptr;
    int prev_dev = -1;
    cudaGetDevice(&prev_dev);
    if (device != prev_dev) cudaSetDevice(device);
    cudaError_t e = cudaMalloc(&new_buf, new_cap * sizeof(float));
    if (device != prev_dev) cudaSetDevice(prev_dev);
    if (e != cudaSuccess) return false;
    if (d_acc_f32) cudaFree(d_acc_f32);
    d_acc_f32 = new_buf;
    acc_capacity_elems = new_cap;
    return true;
}

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
    CUstream_st *       stream_opaque,
    NaverKernelScratch *scratch,
    const void * const *d_q_weight_planes_device,
    const void * const *d_alpha_planes_device)
{
    if (precision < 1 || precision > kNaverMaxPrecision) {
        throw std::runtime_error(
            "naver_gemv_launch: precision (" + std::to_string(precision) +
            ") out of range [1, " + std::to_string(kNaverMaxPrecision) + "]");
    }
    if (group_size % kKTileSize != 0) {
        throw std::runtime_error(
            "naver_gemv_launch: group_size (" + std::to_string(group_size) +
            ") must be a multiple of K_TILE_SIZE=64");
    }
    if (K % kKTileSize != 0) {
        throw std::runtime_error(
            "naver_gemv_launch: K (" + std::to_string(K) +
            ") must be a multiple of K_TILE_SIZE=64");
    }

    // Ptr-array source: either pre-uploaded on-device arrays (no H2D
    // this call, hot path for eager) or host arrays we stage via scratch.
    const bool use_device_ptrs =
        d_q_weight_planes_device != nullptr &&
        d_alpha_planes_device    != nullptr;
    if (!use_device_ptrs && (d_q_weight_planes == nullptr || d_alpha_planes == nullptr)) {
        throw std::runtime_error(
            "naver_gemv_launch: either host or device plane-pointer arrays required");
    }

    cudaStream_t stream = (cudaStream_t) stream_opaque;

    // Scratch path: pre-allocated ptr arrays + acc_f32 owned by runtime.
    // Fallback: per-call cudaMallocAsync/cudaFreeAsync (microbench / tests).
    const size_t ptr_arr_bytes = (size_t)precision * sizeof(void *);
    void ** d_qw_ptrs = nullptr;
    void ** d_a_ptrs  = nullptr;
    float * d_acc_f32 = nullptr;
    const bool use_scratch =
        scratch != nullptr &&
        scratch->d_qw_ptrs != nullptr &&
        scratch->d_alpha_ptrs != nullptr &&
        scratch->ensure_acc((size_t)M);

    if (use_scratch) {
        d_qw_ptrs = scratch->d_qw_ptrs;
        d_a_ptrs  = scratch->d_alpha_ptrs;
        d_acc_f32 = scratch->d_acc_f32;
    } else if (use_device_ptrs) {
        // Still need an acc_f32 buffer. Allocate per call.
        check_cuda(
            cudaMallocAsync((void **)&d_acc_f32, (size_t)M * sizeof(float), stream),
            "cudaMallocAsync(acc_f32)");
    } else {
        check_cuda(
            cudaMallocAsync((void **)&d_qw_ptrs, ptr_arr_bytes, stream),
            "cudaMallocAsync(qw_ptrs)");
        check_cuda(
            cudaMallocAsync((void **)&d_a_ptrs,  ptr_arr_bytes, stream),
            "cudaMallocAsync(a_ptrs)");
        check_cuda(
            cudaMallocAsync((void **)&d_acc_f32, (size_t)M * sizeof(float), stream),
            "cudaMallocAsync(acc_f32)");
    }

    // Pointer-array source for the kernel: prefer already-on-device
    // arrays (skip the H2D); fall back to host-array staging.
    const uint32_t * const * kernel_qw_ptrs;
    const __half   * const * kernel_a_ptrs;
    if (use_device_ptrs) {
        kernel_qw_ptrs = (const uint32_t * const *) d_q_weight_planes_device;
        kernel_a_ptrs  = (const __half   * const *) d_alpha_planes_device;
    } else {
        check_cuda(
            cudaMemcpyAsync(d_qw_ptrs, d_q_weight_planes, ptr_arr_bytes,
                            cudaMemcpyHostToDevice, stream),
            "cudaMemcpyAsync(qw_ptrs)");
        check_cuda(
            cudaMemcpyAsync(d_a_ptrs,  d_alpha_planes, ptr_arr_bytes,
                            cudaMemcpyHostToDevice, stream),
            "cudaMemcpyAsync(a_ptrs)");
        kernel_qw_ptrs = (const uint32_t * const *) d_qw_ptrs;
        kernel_a_ptrs  = (const __half   * const *) d_a_ptrs;
    }
    check_cuda(
        cudaMemsetAsync(d_acc_f32, 0, (size_t)M * sizeof(float), stream),
        "cudaMemsetAsync(acc_f32)");

    dim3 grid(
        (M + kMTileSize - 1) / kMTileSize,
        (K + kKTileSize - 1) / kKTileSize);
    dim3 block(kNumThreads);

    nqmv_bias_planes<<<grid, block, 0, stream>>>(
        kernel_qw_ptrs,
        kernel_a_ptrs,
        (const __half *)           d_q_bias_fp16,
        (const __half *)           d_input_fp16,
        d_acc_f32,
        M, K, precision, group_size);

    cudaError_t last = cudaGetLastError();
    if (last != cudaSuccess) {
        if (!use_scratch) {
            cudaFreeAsync(d_acc_f32, stream);
            if (!use_device_ptrs) {
                cudaFreeAsync(d_qw_ptrs, stream);
                cudaFreeAsync(d_a_ptrs,  stream);
            }
        }
        throw std::runtime_error(
            std::string("naver_gemv_launch: kernel launch failed: ") +
            cudaGetErrorString(last));
    }

    {
        const int cast_block = 256;
        const int cast_grid  = (M + cast_block - 1) / cast_block;
        cast_f32_to_f16<<<cast_grid, cast_block, 0, stream>>>(
            d_acc_f32, (__half *) d_output_fp16, M);
    }

    if (!use_scratch) {
        cudaFreeAsync(d_acc_f32, stream);
        if (!use_device_ptrs) {
            cudaFreeAsync(d_qw_ptrs, stream);
            cudaFreeAsync(d_a_ptrs,  stream);
        }
    }
}

// --- Batched chunked LUT-GEMM ---------------------------------------------
// Y[M, N] = X[K, N] @ decode(planes).T, fp16 → fp16.
// Mirrors naver_gemv_launch but with an extra N dimension. Output is
// written into an [M × N] fp32 scratch via atomicAdd, then cast to fp16.

void naver_gemm_launch(
    const void *        d_input_fp16,
    void *              d_output_fp16,
    const void * const *d_q_weight_planes,
    const void * const *d_alpha_planes,
    const void *        d_q_bias_fp16,
    int                 M,
    int                 K,
    int                 N,
    int                 precision,
    int                 group_size,
    CUstream_st *       stream_opaque)
{
    if (precision < 1 || precision > kNaverMaxPrecision) {
        throw std::runtime_error(
            "naver_gemm_launch: precision (" + std::to_string(precision) +
            ") out of range [1, " + std::to_string(kNaverMaxPrecision) + "]");
    }
    if (group_size % kKTileSize != 0) {
        throw std::runtime_error(
            "naver_gemm_launch: group_size (" + std::to_string(group_size) +
            ") must be a multiple of K_TILE_SIZE=64");
    }
    if (K % kKTileSize != 0) {
        throw std::runtime_error(
            "naver_gemm_launch: K (" + std::to_string(K) +
            ") must be a multiple of K_TILE_SIZE=64");
    }
    if (N < 1) {
        throw std::runtime_error(
            "naver_gemm_launch: N must be >= 1 (got " +
            std::to_string(N) + ")");
    }
    if (d_q_weight_planes == nullptr || d_alpha_planes == nullptr) {
        throw std::runtime_error(
            "naver_gemm_launch: plane pointer arrays must be non-null");
    }

    cudaStream_t stream = (cudaStream_t) stream_opaque;

    // Stage per-plane pointer arrays on device (same idiom as GEMV).
    const size_t ptr_arr_bytes = (size_t) precision * sizeof(void *);
    void ** d_qw_ptrs = nullptr;
    void ** d_a_ptrs  = nullptr;
    check_cuda(
        cudaMallocAsync((void **)&d_qw_ptrs, ptr_arr_bytes, stream),
        "cudaMallocAsync(qw_ptrs)");
    check_cuda(
        cudaMallocAsync((void **)&d_a_ptrs,  ptr_arr_bytes, stream),
        "cudaMallocAsync(a_ptrs)");
    check_cuda(
        cudaMemcpyAsync(d_qw_ptrs, d_q_weight_planes, ptr_arr_bytes,
                        cudaMemcpyHostToDevice, stream),
        "cudaMemcpyAsync(qw_ptrs)");
    check_cuda(
        cudaMemcpyAsync(d_a_ptrs,  d_alpha_planes, ptr_arr_bytes,
                        cudaMemcpyHostToDevice, stream),
        "cudaMemcpyAsync(a_ptrs)");

    // fp32 output scratch, [M × N]. Zeroed before atomicAdds.
    float * d_acc_f32 = nullptr;
    const size_t acc_elems = (size_t) M * (size_t) N;
    check_cuda(
        cudaMallocAsync((void **)&d_acc_f32, acc_elems * sizeof(float), stream),
        "cudaMallocAsync(acc_f32)");
    check_cuda(
        cudaMemsetAsync(d_acc_f32, 0, acc_elems * sizeof(float), stream),
        "cudaMemsetAsync(acc_f32)");

    // Opt-in to 64 KB shared memory per block (Ada default is 48 KB,
    // cap is 99 KB). Must be set before the first launch per kernel
    // function — doing it every call is cheap (driver caches).
    cudaFuncSetAttribute(
        (const void *) &nqmv_bias_planes_batched,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        kBatchedSharedBytes);

    dim3 grid(
        (M + kMTileSize - 1) / kMTileSize,
        (K + kKTileSize - 1) / kKTileSize,
        (N + kNTileSizeBatched - 1) / kNTileSizeBatched);
    dim3 block(kNumThreads);

    nqmv_bias_planes_batched<<<grid, block, kBatchedSharedBytes, stream>>>(
        (const uint32_t * const *) d_qw_ptrs,
        (const __half   * const *) d_a_ptrs,
        (const __half *)           d_q_bias_fp16,
        (const __half *)           d_input_fp16,
        d_acc_f32,
        M, K, N, precision, group_size);

    cudaError_t last = cudaGetLastError();
    if (last != cudaSuccess) {
        cudaFreeAsync(d_acc_f32, stream);
        cudaFreeAsync(d_qw_ptrs, stream);
        cudaFreeAsync(d_a_ptrs,  stream);
        throw std::runtime_error(
            std::string("naver_gemm_launch: kernel launch failed: ") +
            cudaGetErrorString(last));
    }

    {
        const int cast_block = 256;
        const int cast_grid  = (int)((acc_elems + cast_block - 1) / cast_block);
        cast_f32_to_f16<<<cast_grid, cast_block, 0, stream>>>(
            d_acc_f32, (__half *) d_output_fp16, (int) acc_elems);
    }

    cudaFreeAsync(d_acc_f32, stream);
    cudaFreeAsync(d_qw_ptrs, stream);
    cudaFreeAsync(d_a_ptrs,  stream);
}

} // namespace streamllm_ext
