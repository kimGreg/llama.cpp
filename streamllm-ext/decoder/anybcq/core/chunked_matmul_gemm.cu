// streamllm-ext / decoder / anybcq / core — prefill / dense-fp16 GEMM path.
//
// For n_tokens > 1 the per-token GEMV is wasteful — a 2048-token
// prefill would be 2048 kernel launches per tensor and 2048× the same
// weight reread from VRAM. This translation unit holds the alternative:
//
//   1. ``launch_dequant_planes_f16``  — materialise W[M,K] in fp16
//                                       from bitplane signs + α + β,
//                                       honouring the runtime precision
//                                       dial.
//   2. ``batched_gemm_f16``           — Y = W @ X via cublasGemmEx.
//   3. ``naver_gemm_launch``          — alternative fused chunked LUT-GEMM
//                                       (no W materialisation) used when
//                                       ``STREAMLLM_BATCHED_BACKEND=fused``
//                                       is set.
//   4. ``launch_f32_to_f16`` / ``launch_f16_to_f32`` — cast helpers used
//                                       by the hook to bridge ggml's f32
//                                       activations into the kernel's
//                                       f16 inputs.
//
// Bundled into one .cu since they all serve the prefill path and share
// the same ``anybcq_gemm.h`` public header.

#include "anybcq_gemm.h"
#include "anybcq_gemv.h"   // kNaverMaxPrecision (used by naver_gemm_launch)

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>

// =====================================================================
// Fused chunked LUT-GEMM kernel (alternative N>1 backend).
//
// Extension of the per-token NAVER LUT-GEMV (``nqmv_bias_planes``) to
// batched X. Lives in the global namespace because the kernel body is
// vendored from NAVER. Forward-declared by naver_gemm_launch below.
// =====================================================================

#define K_TILE_SIZE 64
#define M_TILE_SIZE 1024
#define NUM_THREADS 256
#define N_TILE_SIZE 8     // 8 tokens per block — 64 KB shared LUT (opt-in
                          // via cudaFuncSetAttribute).

__global__ void nqmv_bias_planes_batched(
    const uint32_t * const * __restrict__ q_weight_planes,
    const __half   * const * __restrict__ alpha_planes,
    const __half   *                     q_bias,
    const __half   *                     input,
    float          *                     output,
    const int      M,
    const int      K,
    const int      N,
    const int      precision,
    const int      group_size)
{
    extern __shared__ float lut_raw[];
    float (*lut)[K_TILE_SIZE/8][256] =
        (float (*)[K_TILE_SIZE/8][256]) lut_raw;

    const int n_tile_base = blockIdx.z * N_TILE_SIZE;
    const int lut_x_size = blockDim.x / (K_TILE_SIZE/8);
    const int lut_y = threadIdx.x / lut_x_size;
    const int lut_x = threadIdx.x % lut_x_size;

    const int K_over_32_offset = blockIdx.y * (K_TILE_SIZE / 32);
    const int group_idx        = (blockIdx.y * K_TILE_SIZE) / group_size;

    const float sgn[8] = {
        (float)(2 * ((lut_x>>0) & 1) - 1),
        (float)(2 * ((lut_x>>1) & 1) - 1),
        (float)(2 * ((lut_x>>2) & 1) - 1),
        (float)(2 * ((lut_x>>3) & 1) - 1),
        (float)(2 * ((lut_x>>4) & 1) - 1),
        (float)(2 * ((lut_x>>5) & 1) - 1),
        (float)(2 * ((lut_x>>6) & 1) - 1),
        (float)(2 * ((lut_x>>7) & 1) - 1),
    };
    const int s = (lut_x_size==1)  ?0:
                  (lut_x_size==2)  ?1:
                  (lut_x_size==4)  ?2:
                  (lut_x_size==8)  ?3:
                  (lut_x_size==16) ?4:
                  (lut_x_size==32) ?5:
                  (lut_x_size==64) ?6:
                  (lut_x_size==128)?7: 8;

    #pragma unroll
    for (int n_off = 0; n_off < N_TILE_SIZE; ++n_off) {
        const int n = n_tile_base + n_off;
        if (n >= N) break;

        const __half * _inp = &input[(size_t)n * K
                                     + blockIdx.y * K_TILE_SIZE
                                     + lut_y * 8];
        float4 inp_vec = ((const float4 *)_inp)[0];
        const __half2 * inp_half2 = (const __half2 *)&inp_vec;
        const float inp_f32[8] = {
            __half2float(inp_half2[0].x), __half2float(inp_half2[0].y),
            __half2float(inp_half2[1].x), __half2float(inp_half2[1].y),
            __half2float(inp_half2[2].x), __half2float(inp_half2[2].y),
            __half2float(inp_half2[3].x), __half2float(inp_half2[3].y),
        };

        float base = sgn[0]*inp_f32[0] + sgn[1]*inp_f32[1]
                   + sgn[2]*inp_f32[2] + sgn[3]*inp_f32[3]
                   + sgn[4]*inp_f32[4] + sgn[5]*inp_f32[5]
                   + sgn[6]*inp_f32[6] + sgn[7]*inp_f32[7];
        lut[n_off][lut_y][lut_x] = base;

        #pragma unroll
        for (int s_iter = s; s_iter < 8; ++s_iter) {
            const float iValue = 2.f * inp_f32[s_iter];
            #pragma unroll
            for (int i = (1 << s_iter); i < (1 << (s_iter + 1)); i += lut_x_size) {
                lut[n_off][lut_y][i + lut_x] =
                    lut[n_off][lut_y][i + lut_x - (1 << s_iter)] + iValue;
            }
        }
    }
    __syncthreads();

    const int m_start = blockIdx.x * M_TILE_SIZE + threadIdx.x * 2;
    const int m_end   = min((blockIdx.x + 1) * M_TILE_SIZE, M);
    const int m_step  = blockDim.x * 2;

    const int n_valid = min(N_TILE_SIZE, N - n_tile_base);

    for (int m = m_start; m < m_end; m += m_step) {
        float acc_lo[N_TILE_SIZE];
        float acc_hi[N_TILE_SIZE];
        #pragma unroll
        for (int n_off = 0; n_off < N_TILE_SIZE; ++n_off) {
            acc_lo[n_off] = 0.f;
            acc_hi[n_off] = 0.f;
        }

        // q_bias term (constant across planes).
        {
            const __half2 qb = ((const __half2 *)&q_bias[group_idx*M + m])[0];
            const float qb_lo = __half2float(qb.x);
            const float qb_hi = __half2float(qb.y);

            #pragma unroll
            for (int n_off = 0; n_off < N_TILE_SIZE; ++n_off) {
                float t_sum = 0.f;
                #pragma unroll
                for (int kt = 0; kt < K_TILE_SIZE/32; ++kt) {
                    t_sum += lut[n_off][kt*4+0][255] + lut[n_off][kt*4+1][255]
                           + lut[n_off][kt*4+2][255] + lut[n_off][kt*4+3][255];
                }
                acc_lo[n_off] = fmaf(qb_lo, t_sum, acc_lo[n_off]);
                acc_hi[n_off] = fmaf(qb_hi, t_sum, acc_hi[n_off]);
            }
        }

        // For each plane, read weight bytes once, apply to all N tokens.
        for (int b = 0; b < precision; ++b) {
            const uint32_t * __restrict__ bW_p = q_weight_planes[b] +
                                                 (size_t)K_over_32_offset * M;
            const __half   * __restrict__ alpha_p = alpha_planes[b];

            uchar4 by0_arr[K_TILE_SIZE/32];
            uchar4 by1_arr[K_TILE_SIZE/32];
            #pragma unroll
            for (int kt = 0; kt < K_TILE_SIZE/32; ++kt) {
                const uint64_t w_pair = ((const uint64_t *)&bW_p[kt*M + m])[0];
                const uint32_t w0 = (uint32_t)w_pair;
                const uint32_t w1 = (uint32_t)(w_pair >> 32);
                by0_arr[kt] = *reinterpret_cast<const uchar4 *>(&w0);
                by1_arr[kt] = *reinterpret_cast<const uchar4 *>(&w1);
            }

            const __half2 a = ((const __half2 *)&alpha_p[group_idx*M + m])[0];
            const float a_lo = __half2float(a.x);
            const float a_hi = __half2float(a.y);

            #pragma unroll
            for (int n_off = 0; n_off < N_TILE_SIZE; ++n_off) {
                float t_lo = 0.f;
                float t_hi = 0.f;
                #pragma unroll
                for (int kt = 0; kt < K_TILE_SIZE/32; ++kt) {
                    const uchar4 by0 = by0_arr[kt];
                    const uchar4 by1 = by1_arr[kt];
                    t_lo += lut[n_off][kt*4+0][by0.x] + lut[n_off][kt*4+1][by0.y]
                         + lut[n_off][kt*4+2][by0.z] + lut[n_off][kt*4+3][by0.w];
                    t_hi += lut[n_off][kt*4+0][by1.x] + lut[n_off][kt*4+1][by1.y]
                         + lut[n_off][kt*4+2][by1.z] + lut[n_off][kt*4+3][by1.w];
                }
                acc_lo[n_off] = fmaf(a_lo, t_lo, acc_lo[n_off]);
                acc_hi[n_off] = fmaf(a_hi, t_hi, acc_hi[n_off]);
            }
        }

        #pragma unroll
        for (int n_off = 0; n_off < N_TILE_SIZE; ++n_off) {
            if (n_off >= n_valid) break;
            const int n = n_tile_base + n_off;
            atomicAdd(&output[(size_t)n * M + m],     acc_lo[n_off]);
            atomicAdd(&output[(size_t)n * M + m + 1], acc_hi[n_off]);
        }
    }
}


// =====================================================================
// streamllm_ext launchers + helper kernels.
// =====================================================================

namespace streamllm_ext {

namespace {

// CUDA error guard used across this TU.
inline void check_cuda(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "anybcq/gemm: CUDA error in %s: %s\n",
                     what, cudaGetErrorString(e));
        std::exit(1);
    }
}

// --- F32 ↔ F16 cast kernels ----------------------------------------
__global__ void k_f32_to_f16(const float * __restrict__ src,
                             __half      * __restrict__ dst,
                             int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

__global__ void k_f16_to_f32(const __half * __restrict__ src,
                             float        * __restrict__ dst,
                             int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __half2float(src[i]);
}

// --- Plane → fp16 dense W kernel -----------------------------------
//
// One thread per output element. precision ≤ 16 so the inner loop is
// short; weights are read with M-innermost stride to coalesce across
// a warp striding in m.
__global__ void k_dequant_planes_f16(
    const uint32_t * const * __restrict__ q_weight_planes,
    const __half   * const * __restrict__ alpha_planes,
    const __half   * __restrict__ q_bias,
    __half *         __restrict__ W_out,
    int M, int K, int precision, int group_size)
{
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M || k >= K) return;

    const int g  = k / group_size;
    const int kt = k / 32;
    const int s  = k % 32;

    float w = __half2float(q_bias[(size_t)g * M + m]);
    for (int p = 0; p < precision; ++p) {
        uint32_t word = q_weight_planes[p][(size_t)kt * M + m];
        int      bit  = (int)((word >> s) & 1u);
        float    sign = bit ? 1.0f : -1.0f;
        float    a    = __half2float(alpha_planes[p][(size_t)g * M + m]);
        w += sign * a;
    }
    W_out[(size_t)m * K + k] = __float2half(w);
}

// --- F32 → F16 cast used by naver_gemm_launch's atomicAdd output ---
__global__ void cast_f32_to_f16(const float * __restrict__ src,
                                __half * __restrict__ dst,
                                int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

// --- cuBLAS handle lifecycle ---------------------------------------
cublasHandle_t g_cublas = nullptr;
std::mutex     g_cublas_mu;

const char * cublas_err_str(cublasStatus_t s) {
    switch (s) {
        case CUBLAS_STATUS_SUCCESS:          return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:  return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:     return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:    return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:    return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:    return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:   return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED:    return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR:    return "CUBLAS_STATUS_LICENSE_ERROR";
        default:                             return "CUBLAS_STATUS_UNKNOWN";
    }
}

constexpr int kBatchedSharedBytes = 64 * 1024;  // N_TILE_SIZE × 8 × 256 × 4

}  // anonymous namespace


// --- Public: cast launchers ---------------------------------------

void launch_f32_to_f16(const void * src_f32, void * dst_f16,
                       int n, cudaStream_t stream) {
    const int block = 256;
    const int grid  = (n + block - 1) / block;
    k_f32_to_f16<<<grid, block, 0, stream>>>(
        (const float *) src_f32, (__half *) dst_f16, n);
}

void launch_f16_to_f32(const void * src_f16, void * dst_f32,
                       int n, cudaStream_t stream) {
    const int block = 256;
    const int grid  = (n + block - 1) / block;
    k_f16_to_f32<<<grid, block, 0, stream>>>(
        (const __half *) src_f16, (float *) dst_f32, n);
}


// --- Public: plane → fp16 dense W ---------------------------------

void launch_dequant_planes_f16(
    const void * const * d_q_weight_planes,
    const void * const * d_alpha_planes,
    const void *         d_q_bias,
    void *               d_W_out,
    int M, int K, int precision, int group_size,
    cudaStream_t         stream)
{
    const size_t ptr_bytes = (size_t)precision * sizeof(void *);
    void ** d_qw_ptrs = nullptr;
    void ** d_a_ptrs  = nullptr;
    check_cuda(cudaMallocAsync((void **)&d_qw_ptrs, ptr_bytes, stream),
               "cudaMallocAsync(qw_ptrs)");
    check_cuda(cudaMallocAsync((void **)&d_a_ptrs,  ptr_bytes, stream),
               "cudaMallocAsync(a_ptrs)");
    check_cuda(
        cudaMemcpyAsync(d_qw_ptrs, d_q_weight_planes, ptr_bytes,
                        cudaMemcpyHostToDevice, stream),
        "cudaMemcpyAsync(qw_ptrs)");
    check_cuda(
        cudaMemcpyAsync(d_a_ptrs, d_alpha_planes, ptr_bytes,
                        cudaMemcpyHostToDevice, stream),
        "cudaMemcpyAsync(a_ptrs)");

    dim3 block(32, 8);
    dim3 grid((K + block.x - 1) / block.x,
              (M + block.y - 1) / block.y);

    k_dequant_planes_f16<<<grid, block, 0, stream>>>(
        (const uint32_t * const *) d_qw_ptrs,
        (const __half   * const *) d_a_ptrs,
        (const __half *)           d_q_bias,
        (__half *)                 d_W_out,
        M, K, precision, group_size);

    check_cuda(cudaFreeAsync(d_qw_ptrs, stream), "cudaFreeAsync(qw_ptrs)");
    check_cuda(cudaFreeAsync(d_a_ptrs,  stream), "cudaFreeAsync(a_ptrs)");
}


// --- Public: cuBLAS F16 GEMM --------------------------------------

bool batched_gemm_init() {
    std::lock_guard<std::mutex> lk(g_cublas_mu);
    if (g_cublas) return true;
    auto st = cublasCreate(&g_cublas);
    if (st != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "batched_gemm: cublasCreate failed: %s\n",
                     cublas_err_str(st));
        g_cublas = nullptr;
        return false;
    }
    cublasSetMathMode(g_cublas, CUBLAS_DEFAULT_MATH);
    return true;
}

void batched_gemm_shutdown() {
    std::lock_guard<std::mutex> lk(g_cublas_mu);
    if (g_cublas) cublasDestroy(g_cublas);
    g_cublas = nullptr;
}

bool batched_gemm_f16(
    const void * W_f16, const void * X_f16, void * Y_f16,
    int M, int K, int N,
    cudaStream_t stream)
{
    if (g_cublas == nullptr) {
        if (!batched_gemm_init()) return false;
    }

    const __half alpha_h = __float2half(1.0f);
    const __half beta_h  = __float2half(0.0f);

    cublasSetStream(g_cublas, stream);
    auto st = cublasGemmEx(
        g_cublas,
        CUBLAS_OP_T, CUBLAS_OP_N,
        M, N, K,
        &alpha_h,
        W_f16, CUDA_R_16F, K,
        X_f16, CUDA_R_16F, K,
        &beta_h,
        Y_f16, CUDA_R_16F, M,
        CUBLAS_COMPUTE_16F,
        CUBLAS_GEMM_DEFAULT);
    if (st != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "batched_gemm_f16: cublasGemmEx failed: %s "
                             "(M=%d N=%d K=%d)\n",
                     cublas_err_str(st), M, N, K);
        return false;
    }
    return true;
}


// --- Public: fused chunked LUT-GEMM (alternative N>1 backend) -----

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
    if (group_size % K_TILE_SIZE != 0) {
        throw std::runtime_error(
            "naver_gemm_launch: group_size (" + std::to_string(group_size) +
            ") must be a multiple of K_TILE_SIZE=64");
    }
    if (K % K_TILE_SIZE != 0) {
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

    float * d_acc_f32 = nullptr;
    const size_t acc_elems = (size_t) M * (size_t) N;
    check_cuda(
        cudaMallocAsync((void **)&d_acc_f32, acc_elems * sizeof(float), stream),
        "cudaMallocAsync(acc_f32)");
    check_cuda(
        cudaMemsetAsync(d_acc_f32, 0, acc_elems * sizeof(float), stream),
        "cudaMemsetAsync(acc_f32)");

    cudaFuncSetAttribute(
        (const void *) &nqmv_bias_planes_batched,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        kBatchedSharedBytes);

    dim3 grid(
        (M + M_TILE_SIZE - 1) / M_TILE_SIZE,
        (K + K_TILE_SIZE - 1) / K_TILE_SIZE,
        (N + N_TILE_SIZE - 1) / N_TILE_SIZE);
    dim3 block(NUM_THREADS);

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

}  // namespace streamllm_ext
