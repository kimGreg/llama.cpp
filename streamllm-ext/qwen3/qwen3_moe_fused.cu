// streamllm-ext / qwen3 — fused MoE LUT-GEMV impl.
//
// Architecture-specific kernel-fusion. See moe_fused.h for the layer
// placement rationale.

#include "qwen3_moe_fused.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <stdexcept>
#include <string>

#define K_TILE_SIZE 64
#define M_TILE_SIZE 1024
#define NUM_THREADS 256

namespace streamllm_ext { namespace qwen3 {

using ::streamllm_ext::MoeExpertTable;

namespace {

inline void check_cuda(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        throw std::runtime_error(
            std::string("qwen3_moe_fused: CUDA error in ") + what + ": " +
            cudaGetErrorString(e));
    }
}

}  // anon


// shared_x = true:  X_fp16 has shape [n_tokens, K]
//                   block reads X_fp16 + t * K
// shared_x = false: X_fp16 has shape [n_tokens, n_used, K]
//                   block reads X_fp16 + tu * K
__global__ void nqmv_bias_planes_moe_fused(
    const __half * __restrict__ X_fp16,
    float *        __restrict__ Y_dst_f32,
    const int32_t * __restrict__ ids,
    const uint32_t * const * const * __restrict__ qw_planes_per_expert,
    const __half   * const * const * __restrict__ alpha_planes_per_expert,
    const __half   *               * __restrict__ q_bias_per_expert,
    const int M,
    const int K,
    const int n_used,
    const int     uniform_precision,
    const int * __restrict__ prec_per_tu,
    const int group_size,
    const int shared_x)
{
    const int tu  = blockIdx.z;
    const int t   = tu / n_used;
    const int eid = ids[tu];
    if (eid < 0) return;
    const int precision = prec_per_tu ? prec_per_tu[tu] : uniform_precision;
    if (precision <= 0) return;

    const uint32_t * const * qw    = qw_planes_per_expert   [eid];
    const __half   * const * alpha = alpha_planes_per_expert[eid];
    const __half   *         q_bias = q_bias_per_expert     [eid];
    if (qw == nullptr || alpha == nullptr || q_bias == nullptr) return;

    const __half * X_t = shared_x
        ? X_fp16 + (size_t)t  * K
        : X_fp16 + (size_t)tu * K;
    float * Y_tu = Y_dst_f32 + (size_t)tu * M;

    __shared__ float lut[K_TILE_SIZE/8][256];
    const int lut_x_size = blockDim.x / (K_TILE_SIZE/8);

    const int lut_y = threadIdx.x / lut_x_size;
    const int lut_x = threadIdx.x % lut_x_size;

    const __half * _inp = &X_t[blockIdx.y * K_TILE_SIZE + lut_y * 8];
    float4 inp_vec = ((float4 *)_inp)[0];
    const __half2 * inp_half2 = (const __half2 *)&inp_vec;
    const float inp_f32[8] = {
        __half2float(inp_half2[0].x), __half2float(inp_half2[0].y),
        __half2float(inp_half2[1].x), __half2float(inp_half2[1].y),
        __half2float(inp_half2[2].x), __half2float(inp_half2[2].y),
        __half2float(inp_half2[3].x), __half2float(inp_half2[3].y),
    };

    const float sgn[8] = {
        (float)(2 * ((lut_x >> 0) & 1) - 1),
        (float)(2 * ((lut_x >> 1) & 1) - 1),
        (float)(2 * ((lut_x >> 2) & 1) - 1),
        (float)(2 * ((lut_x >> 3) & 1) - 1),
        (float)(2 * ((lut_x >> 4) & 1) - 1),
        (float)(2 * ((lut_x >> 5) & 1) - 1),
        (float)(2 * ((lut_x >> 6) & 1) - 1),
        (float)(2 * ((lut_x >> 7) & 1) - 1),
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
    for (int s_iter = s; s_iter < 8; ++s_iter) {
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
    const int group_idx        = (blockIdx.y * K_TILE_SIZE) / group_size;

    for (int m = m_start; m < m_end; m += m_step) {
        float acc_lo = 0.f;
        float acc_hi = 0.f;

        {
            const __half2 qb = ((const __half2 *)&q_bias[group_idx*M + m])[0];
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
            if (qw[b] == nullptr || alpha[b] == nullptr) continue;
            const uint32_t * __restrict__ bW_p = qw[b] +
                                                 (size_t)K_over_32_offset * M;
            const __half   * __restrict__ alpha_p = alpha[b];

            float t_lo = 0.f;
            float t_hi = 0.f;

            #pragma unroll
            for (int kt = 0; kt < K_TILE_SIZE/32; ++kt) {
                const uint64_t w_pair = ((const uint64_t *)&bW_p[kt*M + m])[0];
                const uint32_t w0 = (uint32_t)w_pair;
                const uint32_t w1 = (uint32_t)(w_pair >> 32);

                const uchar4 by0 = *reinterpret_cast<const uchar4 *>(&w0);
                const uchar4 by1 = *reinterpret_cast<const uchar4 *>(&w1);

                t_lo += lut[kt*4+0][by0.x] + lut[kt*4+1][by0.y]
                      + lut[kt*4+2][by0.z] + lut[kt*4+3][by0.w];
                t_hi += lut[kt*4+0][by1.x] + lut[kt*4+1][by1.y]
                      + lut[kt*4+2][by1.z] + lut[kt*4+3][by1.w];
            }

            const __half2 a = ((const __half2 *)&alpha_p[group_idx*M + m])[0];
            acc_lo = fmaf(__half2float(a.x), t_lo, acc_lo);
            acc_hi = fmaf(__half2float(a.y), t_hi, acc_hi);
        }

        atomicAdd(&Y_tu[m],     acc_lo);
        atomicAdd(&Y_tu[m + 1], acc_hi);
    }
}


void naver_gemv_moe_launch(
    const void *           X_fp16,
    void *                 Y_dst_f32,
    const int32_t *        ids_d,
    const MoeExpertTable & table,
    int                    M,
    int                    K,
    int                    n_tokens,
    int                    n_used,
    int                    uniform_precision,
    const int *            prec_per_tu_d,
    int                    group_size,
    int                    shared_x,
    StreamHandle           stream_opaque)
{
    if (prec_per_tu_d == nullptr) {
        if (uniform_precision < 1 || uniform_precision > 8) {
            throw std::runtime_error(
                "qwen3_moe_fused: uniform_precision out of range [1, 8]: " +
                std::to_string(uniform_precision));
        }
    }
    if (K % K_TILE_SIZE != 0) {
        throw std::runtime_error(
            "qwen3_moe_fused: K must be a multiple of K_TILE_SIZE=64");
    }
    if (n_tokens <= 0 || n_used <= 0) {
        return;
    }

    auto stream = (cudaStream_t) stream_opaque;

    dim3 grid(
        (M + M_TILE_SIZE - 1) / M_TILE_SIZE,
        (K + K_TILE_SIZE - 1) / K_TILE_SIZE,
        n_tokens * n_used);
    dim3 block(NUM_THREADS);

    nqmv_bias_planes_moe_fused<<<grid, block, 0, stream>>>(
        (const __half *)   X_fp16,
        (float *)          Y_dst_f32,
        ids_d,
        (const uint32_t * const * const *) table.d_qw_planes_per_expert,
        (const __half   * const * const *) table.d_alpha_planes_per_expert,
        (const __half   *               *) table.d_q_bias_per_expert,
        M, K, n_used, uniform_precision, prec_per_tu_d,
        group_size, shared_x);

    cudaError_t last = cudaGetLastError();
    if (last != cudaSuccess) {
        throw std::runtime_error(
            std::string("qwen3_moe_fused: kernel launch failed: ") +
            cudaGetErrorString(last));
    }
}


// ----- expert-pointer table alloc/free + β refresh -----

void alloc_moe_expert_table(
    MoeExpertTable & out,
    const void * const * host_qw,
    const void * const * host_alpha,
    const void * const * host_q_bias,
    int                  n_experts)
{
    if (n_experts <= 0) {
        throw std::runtime_error(
            "qwen3_moe_fused::alloc_moe_expert_table: n_experts must be > 0");
    }
    out.n_experts = n_experts;

    const size_t bytes = (size_t)n_experts * sizeof(void *);

    check_cuda(cudaMalloc((void **)&out.d_qw_planes_per_expert,    bytes),
               "cudaMalloc(d_qw_planes_per_expert)");
    check_cuda(cudaMalloc((void **)&out.d_alpha_planes_per_expert, bytes),
               "cudaMalloc(d_alpha_planes_per_expert)");
    check_cuda(cudaMalloc((void **)&out.d_q_bias_per_expert,        bytes),
               "cudaMalloc(d_q_bias_per_expert)");

    check_cuda(cudaMemcpy(out.d_qw_planes_per_expert,    host_qw,
                          bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy(d_qw_planes_per_expert)");
    check_cuda(cudaMemcpy(out.d_alpha_planes_per_expert, host_alpha,
                          bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy(d_alpha_planes_per_expert)");
    check_cuda(cudaMemcpy(out.d_q_bias_per_expert,        host_q_bias,
                          bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy(d_q_bias_per_expert)");
}

void free_moe_expert_table(MoeExpertTable & t) {
    if (t.d_qw_planes_per_expert) {
        cudaFree(t.d_qw_planes_per_expert);
        t.d_qw_planes_per_expert = nullptr;
    }
    if (t.d_alpha_planes_per_expert) {
        cudaFree(t.d_alpha_planes_per_expert);
        t.d_alpha_planes_per_expert = nullptr;
    }
    if (t.d_q_bias_per_expert) {
        cudaFree(t.d_q_bias_per_expert);
        t.d_q_bias_per_expert = nullptr;
    }
    if (t.d_qbias_slot_per_expert) {
        cudaFree(t.d_qbias_slot_per_expert);
        t.d_qbias_slot_per_expert = nullptr;
    }
    t.needs_qbias_refresh = false;
    t.n_experts = 0;
}

namespace {
__global__ void k_refresh_qbias(
    void **      d_q_bias_per_expert,
    void * const * d_qbias_slot_per_expert,
    int          n_experts)
{
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_experts) return;
    void ** slot = (void **) d_qbias_slot_per_expert[e];
    d_q_bias_per_expert[e] = slot ? slot[0] : nullptr;
}
}  // anon

void refresh_q_bias_for_anyprec_launch(
    const MoeExpertTable & table,
    StreamHandle stream)
{
    if (!table.needs_qbias_refresh) return;
    if (table.d_q_bias_per_expert == nullptr) return;
    if (table.d_qbias_slot_per_expert == nullptr) return;
    if (table.n_experts <= 0) return;
    const int n = table.n_experts;
    const int block = 64;
    const int grid  = (n + block - 1) / block;
    auto s = (cudaStream_t) stream;
    k_refresh_qbias<<<grid, block, 0, s>>>(
        table.d_q_bias_per_expert,
        (void * const *) table.d_qbias_slot_per_expert,
        n);
}

}}  // namespace streamllm_ext::qwen3
