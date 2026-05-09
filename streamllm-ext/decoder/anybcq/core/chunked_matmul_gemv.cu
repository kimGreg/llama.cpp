// streamllm-ext / decoder / anybcq / core — decode hot path GEMV kernels.
//
// Bundled here:
//
//   1. ``nqmv_bias_planes``            — vendored NAVER LUT-GEMV (single-token).
//   2. ``naver_gemv_launch`` + scratch — host-side launcher that picks
//                                        ``precision`` planes and stages
//                                        pointer arrays.
//   3. Per-plane device-pointer table updates (alloc, free, write,
//      clear) for both shortcut and any-prec layouts.
//
// All three served by the same ``anybcq_gemv.h`` public header.
//
// MoE-fused dispatch (the ``naver_gemv_moe_launch`` kernel + the
// ``MoeExpertTable`` it reads) is NOT here. Fusion across MoE experts
// is an architecture decision; see ``qwen3/moe_fused.cu``.

#include "anybcq_gemv.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#define K_TILE_SIZE 64
#define M_TILE_SIZE 1024
#define NUM_THREADS 256


// =====================================================================
// Vendored NAVER LUT-GEMV kernel (single-token).
//
// Source: naver-aics/anybcq third-party/anybcq/.../anybcq.cu
// Modifications: per-M fp32 accumulator + per-plane device pointers.
// Lives in the global namespace because the kernel body is vendored.
// =====================================================================

__global__ void nqmv_bias_planes(
    const uint32_t * const * __restrict__ q_weight_planes,
    const __half   * const * __restrict__ alpha_planes,
    const __half   *                     q_bias,
    const __half   *                     input,
    float          *                     output,
    const int      M,
    const int      K,
    const int      precision,
    const int      group_size)
{
    __shared__ float lut[K_TILE_SIZE/8][256];
    const int lut_x_size = blockDim.x / (K_TILE_SIZE/8);

    const int lut_y = threadIdx.x / lut_x_size;
    const int lut_x = threadIdx.x % lut_x_size;

    const __half * _inp = &input[blockIdx.y * K_TILE_SIZE + lut_y * 8];
    float4 inp_vec = ((float4 *)_inp)[0];
    const __half2 * inp_half2 = (const __half2 *)&inp_vec;
    float inp_f32[8] = {
        __half2float(inp_half2[0].x), __half2float(inp_half2[0].y),
        __half2float(inp_half2[1].x), __half2float(inp_half2[1].y),
        __half2float(inp_half2[2].x), __half2float(inp_half2[2].y),
        __half2float(inp_half2[3].x), __half2float(inp_half2[3].y),
    };

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
    float base = sgn[0]*inp_f32[0] + sgn[1]*inp_f32[1]
               + sgn[2]*inp_f32[2] + sgn[3]*inp_f32[3]
               + sgn[4]*inp_f32[4] + sgn[5]*inp_f32[5]
               + sgn[6]*inp_f32[6] + sgn[7]*inp_f32[7];
    lut[lut_y][lut_x] = base;

    const int s = (lut_x_size==1)  ?0:
                  (lut_x_size==2)  ?1:
                  (lut_x_size==4)  ?2:
                  (lut_x_size==8)  ?3:
                  (lut_x_size==16) ?4:
                  (lut_x_size==32) ?5:
                  (lut_x_size==64) ?6:
                  (lut_x_size==128)?7: 8;

    #pragma unroll
    for(int s_iter = s; s_iter < 8; s_iter++){
        const float iValue = 2.f * inp_f32[s_iter];
        #pragma unroll
        for (int i = (1 << s_iter); i < (1 << (s_iter + 1)); i += lut_x_size) {
            lut[lut_y][i + lut_x] = lut[lut_y][i + lut_x - (1 << s_iter)] + iValue;
        }
    }
    __syncthreads();

    const int m_start = blockIdx.x * M_TILE_SIZE + threadIdx.x * 2;
    const int m_end   = min((blockIdx.x + 1) * M_TILE_SIZE, M);
    const int m_step  = blockDim.x * 2;

    const int K_over_32_offset = blockIdx.y * (K_TILE_SIZE / 32);
    const int group_idx = (blockIdx.y * K_TILE_SIZE) / group_size;

    for (int m = m_start; m < m_end; m += m_step) {
        float acc_lo = 0.f;
        float acc_hi = 0.f;

        {
            const __half2 qb = ((const __half2*)&q_bias[group_idx*M + m])[0];
            const float qb_lo = __half2float(qb.x);
            const float qb_hi = __half2float(qb.y);

            float t_sum = 0.f;
            #pragma unroll
            for (int kt = 0; kt < K_TILE_SIZE/32; ++kt) {
                t_sum += lut[kt*4+0][255] + lut[kt*4+1][255]
                       + lut[kt*4+2][255] + lut[kt*4+3][255];
            }
            acc_lo = fmaf(qb_lo, t_sum, acc_lo);
            acc_hi = fmaf(qb_hi, t_sum, acc_hi);
        }

        for (int b = 0; b < precision; ++b) {
            const uint32_t * __restrict__ bW_p = q_weight_planes[b] +
                                                 (size_t)K_over_32_offset * M;
            const __half   * __restrict__ alpha_p = alpha_planes[b];

            float t_lo = 0.f;
            float t_hi = 0.f;

            #pragma unroll
            for (int kt = 0; kt < K_TILE_SIZE/32; ++kt) {
                const uint64_t w_pair = ((const uint64_t*)&bW_p[kt*M + m])[0];
                const uint32_t w0 = (uint32_t)w_pair;
                const uint32_t w1 = (uint32_t)(w_pair >> 32);

                const uchar4 by0 = *reinterpret_cast<const uchar4*>(&w0);
                const uchar4 by1 = *reinterpret_cast<const uchar4*>(&w1);

                t_lo += lut[kt*4+0][by0.x] + lut[kt*4+1][by0.y]
                      + lut[kt*4+2][by0.z] + lut[kt*4+3][by0.w];
                t_hi += lut[kt*4+0][by1.x] + lut[kt*4+1][by1.y]
                      + lut[kt*4+2][by1.z] + lut[kt*4+3][by1.w];
            }

            const __half2 a = ((const __half2*)&alpha_p[group_idx*M + m])[0];
            acc_lo = fmaf(__half2float(a.x), t_lo, acc_lo);
            acc_hi = fmaf(__half2float(a.y), t_hi, acc_hi);
        }
        atomicAdd(&output[m],     acc_lo);
        atomicAdd(&output[m + 1], acc_hi);
    }
}


namespace streamllm_ext { namespace anybcq {

namespace {

inline void check_cuda(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        throw std::runtime_error(
            std::string("anybcq/gemv: CUDA error in ") + what + ": " +
            cudaGetErrorString(e));
    }
}

}  // anonymous
// =====================================================================
// Per-plane device-pointer table updates.
// =====================================================================

void alloc_per_plane_arrays(void **& d_qw, void **& d_alpha,
                            int max_planes) {
    if (max_planes <= 0) {
        throw std::runtime_error(
            "alloc_per_plane_arrays: max_planes must be > 0");
    }
    if (d_qw == nullptr) {
        check_cuda(cudaMalloc((void **)&d_qw,
                              (size_t)max_planes * sizeof(void *)),
                   "cudaMalloc(d_qw)");
        check_cuda(cudaMemset(d_qw, 0,
                              (size_t)max_planes * sizeof(void *)),
                   "cudaMemset(d_qw)");
    }
    if (d_alpha == nullptr) {
        check_cuda(cudaMalloc((void **)&d_alpha,
                              (size_t)max_planes * sizeof(void *)),
                   "cudaMalloc(d_alpha)");
        check_cuda(cudaMemset(d_alpha, 0,
                              (size_t)max_planes * sizeof(void *)),
                   "cudaMemset(d_alpha)");
    }
}

void free_per_plane_arrays(void *& d_qw, void *& d_alpha) {
    if (d_qw)    { cudaFree(d_qw);    d_qw    = nullptr; }
    if (d_alpha) { cudaFree(d_alpha); d_alpha = nullptr; }
}

void clear_per_plane_after_evict(void ** d_qw, void ** d_alpha, int plane) {
    if (d_qw == nullptr || d_alpha == nullptr) return;
    if (plane < 0) return;
    void * null_ptr = nullptr;
    cudaMemcpy((uint8_t *)d_qw    + (size_t)plane * sizeof(void *),
               &null_ptr, sizeof(void *), cudaMemcpyHostToDevice);
    cudaMemcpy((uint8_t *)d_alpha + (size_t)plane * sizeof(void *),
               &null_ptr, sizeof(void *), cudaMemcpyHostToDevice);
}

void update_per_plane_after_load(void ** d_qw, void ** d_alpha,
                                 int plane,
                                 const void * chunk_device_ptr,
                                 size_t qw_bytes_per_chunk) {
    if (d_qw == nullptr || d_alpha == nullptr) return;
    if (plane < 0) return;
    const void * qw_p    = chunk_device_ptr;
    const void * alpha_p = static_cast<const uint8_t *>(chunk_device_ptr) +
                            qw_bytes_per_chunk;
    cudaMemcpy((uint8_t *)d_qw    + (size_t)plane * sizeof(void *),
               &qw_p,    sizeof(void *), cudaMemcpyHostToDevice);
    cudaMemcpy((uint8_t *)d_alpha + (size_t)plane * sizeof(void *),
               &alpha_p, sizeof(void *), cudaMemcpyHostToDevice);
}

namespace {
__global__ void k_set_two_ptrs(void ** d_qw, void ** d_alpha,
                                int plane,
                                void * qw_p, void * alpha_p) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_qw[plane]    = qw_p;
        d_alpha[plane] = alpha_p;
    }
}
}  // anon

void update_per_plane_after_load_async(void ** d_qw, void ** d_alpha,
                                       int plane,
                                       const void * chunk_device_ptr,
                                       size_t qw_bytes_per_chunk,
                                       StreamHandle stream) {
    if (d_qw == nullptr || d_alpha == nullptr) return;
    if (plane < 0) return;
    void * qw_p    = const_cast<void *>(chunk_device_ptr);
    void * alpha_p = const_cast<void *>(
        static_cast<const void *>(
            static_cast<const uint8_t *>(chunk_device_ptr) +
            qw_bytes_per_chunk));
    auto s = (cudaStream_t) stream;
    k_set_two_ptrs<<<1, 1, 0, s>>>(d_qw, d_alpha, plane, qw_p, alpha_p);
}

namespace {
__global__ void k_set_anyprec_ptrs(
    void **      d_qw,
    void **      d_alpha,
    void **      d_qbias_slot,
    int          plane_idx_first,
    int          n_planes_signs,
    int          precision_alpha,
    void *       chunk_base,
    size_t       qw_bytes,
    size_t       alpha_bytes,
    size_t       /*qbias_bytes_offset_into_chunk*/)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    uint8_t * base = (uint8_t *)chunk_base;
    for (int p = 0; p < n_planes_signs; ++p) {
        d_qw[plane_idx_first + p] = base + (size_t)p * qw_bytes;
    }
    uint8_t * alpha_base = base + (size_t)n_planes_signs * qw_bytes;
    for (int p = 0; p < precision_alpha; ++p) {
        d_alpha[p] = alpha_base + (size_t)p * alpha_bytes;
    }
    if (d_qbias_slot != nullptr) {
        uint8_t * beta_base = alpha_base + (size_t)precision_alpha * alpha_bytes;
        d_qbias_slot[0] = beta_base;
    }
}
}  // anon

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
    size_t       /*q_bias_bytes_per_chunk*/,
    StreamHandle stream)
{
    if (d_qw_ptrs == nullptr || d_alpha_ptrs == nullptr) return;
    if (chunk_device_ptr == nullptr) return;
    auto s = (cudaStream_t)stream;
    k_set_anyprec_ptrs<<<1, 1, 0, s>>>(
        d_qw_ptrs, d_alpha_ptrs, d_qbias_slot,
        plane_idx_first, n_planes_this_chunk, precision_at_chunk,
        const_cast<void *>(chunk_device_ptr),
        qw_bytes_per_chunk, alpha_bytes_per_chunk,
        /*qbias_bytes_offset_into_chunk=*/0);
}

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
    size_t       q_bias_bytes_per_chunk)
{
    update_anyprec_after_load_async(
        d_qw_ptrs, d_alpha_ptrs, d_qbias_slot,
        plane_idx_first, n_planes_this_chunk, precision_at_chunk,
        chunk_device_ptr, qw_bytes_per_chunk, alpha_bytes_per_chunk,
        q_bias_bytes_per_chunk, /*stream=*/nullptr);
    cudaDeviceSynchronize();
}

}}  // namespace streamllm_ext::anybcq


// =====================================================================
// Single-token GEMV launcher (lives in streamllm_ext::, not anybcq::).
// =====================================================================

namespace {

__global__ void cast_f32_to_f16(const float * __restrict__ src,
                                __half * __restrict__ dst,
                                int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

inline void check_cuda_gemv(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        throw std::runtime_error(
            std::string("naver_gemv: CUDA error in ") + what + ": " +
            cudaGetErrorString(e));
    }
}

}  // anonymous


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
    if (group_size % K_TILE_SIZE != 0) {
        throw std::runtime_error(
            "naver_gemv_launch: group_size (" + std::to_string(group_size) +
            ") must be a multiple of K_TILE_SIZE=64");
    }
    if (K % K_TILE_SIZE != 0) {
        throw std::runtime_error(
            "naver_gemv_launch: K (" + std::to_string(K) +
            ") must be a multiple of K_TILE_SIZE=64");
    }

    const bool use_device_ptrs =
        d_q_weight_planes_device != nullptr &&
        d_alpha_planes_device    != nullptr;
    if (!use_device_ptrs && (d_q_weight_planes == nullptr || d_alpha_planes == nullptr)) {
        throw std::runtime_error(
            "naver_gemv_launch: either host or device plane-pointer arrays required");
    }

    cudaStream_t stream = (cudaStream_t) stream_opaque;

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
        check_cuda_gemv(
            cudaMallocAsync((void **)&d_acc_f32, (size_t)M * sizeof(float), stream),
            "cudaMallocAsync(acc_f32)");
    } else {
        check_cuda_gemv(
            cudaMallocAsync((void **)&d_qw_ptrs, ptr_arr_bytes, stream),
            "cudaMallocAsync(qw_ptrs)");
        check_cuda_gemv(
            cudaMallocAsync((void **)&d_a_ptrs,  ptr_arr_bytes, stream),
            "cudaMallocAsync(a_ptrs)");
        check_cuda_gemv(
            cudaMallocAsync((void **)&d_acc_f32, (size_t)M * sizeof(float), stream),
            "cudaMallocAsync(acc_f32)");
    }

    const uint32_t * const * kernel_qw_ptrs;
    const __half   * const * kernel_a_ptrs;
    if (use_device_ptrs) {
        kernel_qw_ptrs = (const uint32_t * const *) d_q_weight_planes_device;
        kernel_a_ptrs  = (const __half   * const *) d_alpha_planes_device;
    } else {
        check_cuda_gemv(
            cudaMemcpyAsync(d_qw_ptrs, d_q_weight_planes, ptr_arr_bytes,
                            cudaMemcpyHostToDevice, stream),
            "cudaMemcpyAsync(qw_ptrs)");
        check_cuda_gemv(
            cudaMemcpyAsync(d_a_ptrs,  d_alpha_planes, ptr_arr_bytes,
                            cudaMemcpyHostToDevice, stream),
            "cudaMemcpyAsync(a_ptrs)");
        kernel_qw_ptrs = (const uint32_t * const *) d_qw_ptrs;
        kernel_a_ptrs  = (const __half   * const *) d_a_ptrs;
    }
    check_cuda_gemv(
        cudaMemsetAsync(d_acc_f32, 0, (size_t)M * sizeof(float), stream),
        "cudaMemsetAsync(acc_f32)");

    dim3 grid(
        (M + M_TILE_SIZE - 1) / M_TILE_SIZE,
        (K + K_TILE_SIZE - 1) / K_TILE_SIZE);
    dim3 block(NUM_THREADS);

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

}  // namespace streamllm_ext
