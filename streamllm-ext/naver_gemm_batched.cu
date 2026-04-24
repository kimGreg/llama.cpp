// streamllm-ext — batched (n_tokens > 1) chunked LUT-GEMM.
//
// Extension of the per-token NAVER LUT-GEMV kernel
// (``nqmv_bias_planes`` in naver_kernel_copy.cu) to batched X. Input
// is a contiguous [K × N] column-major block (ggml src1 layout); output
// is [M × N] column-major.
//
// Why a fused batched kernel at all:
//   * Dequant → cuBLAS (M12's path) materialises a dense W[M, K] in
//     fp16 for every mul_mat — a 40-50 MB roundtrip per call. That's
//     half the DRAM budget on our shapes.
//   * Upstream AnyBCQ does the same thing in python
//     (AnyBCQLinear._gemm = anybcq_dequant + torch.matmul). So the
//     dense-W path is the reference, not a novelty.
//   * A fused kernel keeps compute on the compressed bytes directly —
//     signs + α read once per K-tile per M-tile, amortised across the
//     N-tile of tokens. Matches the chunked-matmul abstraction: the
//     kernel consumes exactly the resident planes, no dense W is ever
//     produced.
//
// Shape & layout (same convention as naver_gemv_launch):
//   q_weight_planes[p]  : [K/32, M]  uint32   column-major (M inner)
//   alpha_planes[p]     : [K_groups, M] __half column-major
//   q_bias              : [K_groups, M] __half column-major
//   input X             : [K, N]      __half   column-major (ggml src1)
//   output_scratch      : [M, N]      float    column-major (atomicAdd target)
//
// Grid / block:
//   grid  = (M / M_TILE_SIZE, K / K_TILE_SIZE, ceil(N / N_TILE_SIZE))
//   block = NUM_THREADS
//
// Shared memory:
//   float lut[N_TILE_SIZE][K_TILE_SIZE/8][256]   — one LUT per token in
//     the tile, built once per (K_tile, N_tile) and reused across every
//     M output this block produces.
//   At N_TILE_SIZE=4, K_TILE_SIZE=64, fp32: 4 × 8 × 256 × 4 = 32 KB.
//
// Each thread processes 2 M outputs for all N_TILE_SIZE tokens in the
// tile (pairs via __half2 alignment on the weight side). acc[n][lo/hi]
// lives in registers.
//
// Cross-K-tile reduction: atomicAdd into the fp32 output scratch. The
// launcher zero-inits the scratch and casts back to fp16 after.

#include <cuda_fp16.h>
#include <cstdint>

#define K_TILE_SIZE   64
#define M_TILE_SIZE  1024
#define NUM_THREADS   256
#define N_TILE_SIZE     8    // 8 tokens per block — 64 KB shared LUT.
                             // Opt-in via cudaFuncSetAttribute in the launcher.


__global__ void nqmv_bias_planes_batched(
    const uint32_t * const * __restrict__ q_weight_planes,  // [precision][K/32 * M]
    const __half   * const * __restrict__ alpha_planes,     // [precision][K_groups * M]
    const __half   *                     q_bias,            // [K_groups * M]
    const __half   *                     input,             // [K * N] (K inner)
    float          *                     output,            // [M * N] (M inner) fp32 scratch
    const int      M,
    const int      K,
    const int      N,
    const int      precision,
    const int      group_size
) {
    // Shared LUTs, one per token in the N-tile.
    // N_TILE × 8 × 256 × 4 = 64 KB at N_TILE=8 — exceeds the 48 KB
    // static-shmem default; declared as dynamic shared memory and
    // reinterpreted as a 3-D array.
    extern __shared__ float lut_raw[];
    float (*lut)[K_TILE_SIZE/8][256] =
        (float (*)[K_TILE_SIZE/8][256]) lut_raw;

    const int n_tile_base = blockIdx.z * N_TILE_SIZE;
    const int lut_x_size = blockDim.x / (K_TILE_SIZE/8);  // = 32 for NUM_THREADS=256
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

    // ---- Build N_TILE_SIZE LUTs (one per token in this N-tile) ----------
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

    // ---- Main M-loop: for each (m, m+1) output, apply all planes to all N --

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

        // q_bias term (shared across planes — constant in b).
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

            // Cache weight bytes per K-tile in registers — shared across N tokens.
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

        // atomicAdd into output[M, N] col-major (M inner → output[m + n*M]).
        #pragma unroll
        for (int n_off = 0; n_off < N_TILE_SIZE; ++n_off) {
            if (n_off >= n_valid) break;
            const int n = n_tile_base + n_off;
            atomicAdd(&output[(size_t)n * M + m],     acc_lo[n_off]);
            atomicAdd(&output[(size_t)n * M + m + 1], acc_hi[n_off]);
        }
    }
}
