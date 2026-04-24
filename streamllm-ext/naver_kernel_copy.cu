// streamllm-ext — NAVER-derived nqmv_bias kernel with fp32 accumulation
// and per-plane base pointers.
//
// Source: naver-aics/anybcq
//   third-party/anybcq/anybcq/inference/custom_kernel/anybcq.cu
//   commit c316a4d06fc7714aa6b500b4d607a08ec6fe0b24
//
// Copyright (c) 2025-present NAVER Cloud Corp.
// Licensed under the Apache License, Version 2.0.
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Modifications vs upstream:
//
//   * Per-M accumulator + LUT in ``float`` (M4b-integration.4). Cross-
//     block reduction is ``atomicAdd<float>`` into an M-sized fp32
//     scratch the launcher allocates + casts back to fp16.
//
//   * ``q_weight`` and ``alpha`` are arrays of device pointers indexed
//     by plane id (M8). Each plane is its own contiguous [K/32, M]
//     (signs) or [K_groups, M] (α) buffer, so the VRAM pool can move
//     planes independently and the kernel picks ``precision`` of them
//     at launch time.
//
// Why a copy rather than compiling upstream directly? Upstream pulls
// in libtorch via its helper headers; fatal in a llama.cpp build.

#include <cuda_fp16.h>
#include <cstdint>

#define K_TILE_SIZE 64
#define M_TILE_SIZE 1024
#define NUM_THREADS 256

__global__ void nqmv_bias_planes(
    const uint32_t * const * __restrict__ q_weight_planes,  // [precision][K/32 * M]
    const __half   * const * __restrict__ alpha_planes,     // [precision][K_groups * M]
    const __half   *                     q_bias,            // [K_groups * M]
    const __half   *                     input,             // [K]
    float          *                     output,            // [M] fp32 scratch
    const int      M,
    const int      K,
    const int      precision,
    const int      group_size
) {
    __shared__ float lut[K_TILE_SIZE/8][256];
    const int lut_x_size = blockDim.x / (K_TILE_SIZE/8);

    const int lut_y = threadIdx.x / lut_x_size;
    const int lut_x = threadIdx.x % lut_x_size;

    const __half *_inp = &input[blockIdx.y * K_TILE_SIZE + lut_y * 8];

    float4 inp_vec = ((float4*)_inp)[0];
    const __half2 *inp_half2 = (const __half2*)&inp_vec;
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

    // blockIdx.y strides the outer K dim in K_TILE_SIZE blocks.
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
