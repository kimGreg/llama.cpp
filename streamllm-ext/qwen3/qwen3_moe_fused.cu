// streamllm-ext / qwen3 — fused MoE LUT-GEMV impl.
//
// Architecture-specific kernel-fusion. See moe_fused.h for the layer
// placement rationale.

#include "qwen3_moe_fused.h"
#include "launch_diag.h"

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
    const int * __restrict__ prec_per_eid,
    const int group_size,
    const int shared_x)
{
    const int tu  = blockIdx.z;
    const int t   = tu / n_used;
    const int eid = ids[tu];
    if (eid < 0) return;
    const int precision = prec_per_eid ? prec_per_eid[eid] : uniform_precision;
    if (precision <= 0) return;

    // Required-non-null contract (SSOT §6.9 M1).  When precision > 0 and
    // eid >= 0 the scheduler must have reserved + loaded all planes in
    // [0, precision) for this expert.  Per-expert tables and q_bias must
    // be non-null; per-plane pointers in [0, precision) must be non-null.
    // Null here is a loader/synchronization bug, not "skip plane".
    const uint32_t * const * qw    = qw_planes_per_expert   [eid];
    const __half   * const * alpha = alpha_planes_per_expert[eid];
    const __half   *         q_bias = q_bias_per_expert     [eid];
    if (qw == nullptr || alpha == nullptr || q_bias == nullptr) __trap();

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
            // Required-non-null (M1): plane b ∈ [0, precision) must be
            // resident.  Null → loader/sync bug → trap.
            if (qw[b] == nullptr || alpha[b] == nullptr) __trap();
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
    const int *            prec_per_eid_d,
    int                    group_size,
    int                    shared_x,
    StreamHandle           stream_opaque)
{
    if (prec_per_eid_d == nullptr) {
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
        M, K, n_used, uniform_precision, prec_per_eid_d,
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
    streamllm_ext::launch_diag::note_launch(
        streamllm_ext::launch_diag::Kind::QbiasRefresh);
    k_refresh_qbias<<<grid, block, 0, s>>>(
        table.d_q_bias_per_expert,
        (void * const *) table.d_qbias_slot_per_expert,
        n);
}


namespace {
// Rebuild the per-expert α and β pointer tables from the top resident
// chunk's qw base, indexed by the per-expert plane count for THIS
// dispatch.  Runs once per MoE dispatch on the compute stream right
// before the fused MoE kernel.
//
// Why both α and β: any-prec stores precision-dependent α + β in each
// chunk's payload.  ``update_anyprec_after_load`` wrote the kernel
// table to point into whichever chunk loaded LAST — under varying-
// precision dial + cap pressure, that chunk may have been evicted,
// or it may simply be the wrong precision for THIS dispatch.  We
// rebuild from the still-resident top chunk by:
//   1. Reading d_qw_e[top_plane] to get the top chunk's slot base
//      (its qw section starts here).
//   2. α section = qw_base + n_planes_signs(top) × qw_bytes_per_chunk
//   3. β section = α section + P × alpha_bytes_per_chunk
//                  (chunk c stores P=base_p+c α values, then β)
//
// This makes the kernel-facing pointer table a pure function of
// (residency state, per-expert precision) computed at launch time,
// so stale-load / eviction-race effects can't propagate.
__global__ void k_refresh_alpha_beta_anyprec(
    void * const * const * d_alpha_planes_per_expert,   // [n_experts] → [kMaxChunks] α ptrs
    void **                d_q_bias_per_expert,         // [n_experts] β ptrs
    const void * const * const * d_qw_planes_per_expert,// [n_experts] → [kMaxChunks] qw ptrs
    const int *  prec_per_eid_d,
    int          base_p,
    int          n_experts,
    size_t       qw_bytes_per_chunk,
    size_t       alpha_bytes_per_chunk)
{
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_experts) return;
    const int P = prec_per_eid_d[e];
    if (P < 1) return;

    void ** d_alpha_e = (void **) d_alpha_planes_per_expert[e];
    const void * const * d_qw_e = d_qw_planes_per_expert[e];
    if (d_alpha_e == nullptr || d_qw_e == nullptr) return;

    // P = base_p + (N − 1) where N = number of chunks loaded for this
    // expert. The top resident chunk owns plane (P − 1) for N >= 2, or
    // planes [0, base_p) for N == 1. In both cases its qw section
    // starts at d_qw_e[top_plane].
    int top_plane;
    int n_planes_signs;
    if (P <= base_p) {
        // N == 1: chunk 0 only.
        top_plane      = 0;
        n_planes_signs = base_p;
    } else {
        top_plane      = P - 1;
        n_planes_signs = 1;
    }

    const uint8_t * qw_top =
        (const uint8_t *) d_qw_e[top_plane];
    if (qw_top == nullptr) return;

    const uint8_t * alpha_base =
        qw_top + (size_t) n_planes_signs * qw_bytes_per_chunk;

    for (int p = 0; p < P; ++p) {
        d_alpha_e[p] = (void *)(alpha_base +
                                 (size_t) p * alpha_bytes_per_chunk);
    }

    // β section follows the precision-P α array inside the top chunk.
    if (d_q_bias_per_expert != nullptr) {
        const uint8_t * beta_base =
            alpha_base + (size_t) P * alpha_bytes_per_chunk;
        d_q_bias_per_expert[e] = (void *) beta_base;
    }
}
}  // anon

void refresh_alpha_for_anyprec_launch(
    const MoeExpertTable & table,
    const int *  prec_per_eid_d,
    int          base_p,
    size_t       qw_bytes_per_chunk,
    size_t       alpha_bytes_per_chunk,
    StreamHandle stream)
{
    if (table.d_alpha_planes_per_expert == nullptr) return;
    if (table.d_qw_planes_per_expert    == nullptr) return;
    if (prec_per_eid_d == nullptr) return;
    if (table.n_experts <= 0) return;
    if (base_p <= 0) return;
    // Debug: confirm we're running with sane layout params.
    // Fires once per process via static guard.
    static bool logged = false;
    if (!logged) {
        std::fprintf(stderr,
            "streamllm-ext: refresh_alpha_for_anyprec_launch active "
            "(base_p=%d qw_bytes=%zu alpha_bytes=%zu n_experts=%d)\n",
            base_p, qw_bytes_per_chunk, alpha_bytes_per_chunk,
            table.n_experts);
        logged = true;
    }
    const int n = table.n_experts;
    const int block = 64;
    const int grid  = (n + block - 1) / block;
    auto s = (cudaStream_t) stream;
    streamllm_ext::launch_diag::note_launch(
        streamllm_ext::launch_diag::Kind::AlphaBetaRefresh);
    k_refresh_alpha_beta_anyprec<<<grid, block, 0, s>>>(
        (void * const * const *) table.d_alpha_planes_per_expert,
        table.d_q_bias_per_expert,
        (const void * const * const *) table.d_qw_planes_per_expert,
        prec_per_eid_d,
        base_p,
        n,
        qw_bytes_per_chunk,
        alpha_bytes_per_chunk);
}

// Mode A milestone 1, S4 — weighted reduce over the per-top-k slot axis.
//
// layer_out[t, m] = sum_{u=0..n_used} weights[t, u] * slot_out[t, u, m]
//
// One block per (t, M-tile). Threads in the block stride over the M-tile
// (each thread owns a subset of m); for its m, each thread loops over u
// in [0, n_used) accumulating the weighted sum.  Reads from slot_out are
// strided by M across u — coalesced within a warp (consecutive threads
// read consecutive m), serialised across u.  Reads from weights are
// broadcast within a warp (all threads read the same u).
//
// ────────────────────────────────────────────────────────────────────
// SwiGLU * up — element-wise silu(gate) * up. M1 cutover S6.
// ────────────────────────────────────────────────────────────────────

namespace {

constexpr int kSwigluThreads = 256;
constexpr int kSwigluPerThread = 4;

__global__ void k_swiglu_mul(
    const float * __restrict__ slot_gate,
    const float * __restrict__ slot_up,
    float       * __restrict__ slot_out,
    std::size_t                N)
{
    const std::size_t base =
        (std::size_t) blockIdx.x * kSwigluThreads * kSwigluPerThread
        + threadIdx.x;
    #pragma unroll
    for (int i = 0; i < kSwigluPerThread; ++i) {
        const std::size_t idx = base + (std::size_t) i * kSwigluThreads;
        if (idx >= N) return;
        const float g = slot_gate[idx];
        const float u = slot_up[idx];
        // silu(g) = g * sigmoid(g) = g / (1 + exp(-g)).
        // Use the standard form; CUDA's __expf is fast-path single-precision.
        const float sig = 1.0f / (1.0f + __expf(-g));
        slot_out[idx]   = g * sig * u;
    }
}

}  // anon

void launch_swiglu_mul(
    const float * slot_gate,
    const float * slot_up,
    float       * slot_out,
    std::size_t   N,
    StreamHandle  stream)
{
    if (N == 0 || slot_gate == nullptr || slot_up == nullptr || slot_out == nullptr) return;
    const std::size_t elems_per_block =
        (std::size_t) kSwigluThreads * kSwigluPerThread;
    const std::size_t n_blocks = (N + elems_per_block - 1) / elems_per_block;
    streamllm_ext::launch_diag::note_launch(
        streamllm_ext::launch_diag::Kind::SwigluMul);
    k_swiglu_mul<<<(unsigned) n_blocks, (unsigned) kSwigluThreads,
                   0, (cudaStream_t) stream>>>(
        slot_gate, slot_up, slot_out, N);
}

// ────────────────────────────────────────────────────────────────────
// Weighted reduce over the per-top-k slot axis. M1 cutover S4.
// ────────────────────────────────────────────────────────────────────
//
// Tunables — small for M1's typical shapes (Qwen3-30B-A3B: M = 768,
// n_used = 8). M_TILE = 256 keeps each block manageable; threads = 256
// gives one m per thread per iteration. The kernel handles M not
// divisible by M_TILE via the boundary check.
namespace {

constexpr int kWRMTile  = 256;
constexpr int kWRThreads = 256;

__global__ void k_weighted_reduce_slots(
    const float * __restrict__ slot_out,    // [n_tokens, n_used, M]
    const float * __restrict__ weights,     // [n_tokens, n_used]
    float       * __restrict__ layer_out,   // [n_tokens, M]
    int                        n_tokens,
    int                        n_used,
    int                        M)
{
    const int t      = blockIdx.x;
    const int m_base = blockIdx.y * kWRMTile;
    if (t >= n_tokens) return;

    const float * w_row = weights + (size_t) t * n_used;
    const float * s_tok = slot_out + (size_t) t * n_used * M;
    float       * y_row = layer_out + (size_t) t * M;

    for (int m_off = threadIdx.x; m_off < kWRMTile; m_off += kWRThreads) {
        const int m = m_base + m_off;
        if (m >= M) break;
        float acc = 0.f;
        // Linear scan over u; broadcast-style read of weights.
        for (int u = 0; u < n_used; ++u) {
            acc += w_row[u] * s_tok[(size_t) u * M + m];
        }
        y_row[m] = acc;
    }
}

}  // anon

void launch_weighted_reduce_slots(
    const float * slot_out,
    const float * weights,
    float       * layer_out,
    int           n_tokens,
    int           n_used,
    int           M,
    StreamHandle  stream)
{
    if (n_tokens <= 0 || n_used <= 0 || M <= 0) return;
    if (slot_out == nullptr || weights == nullptr || layer_out == nullptr) return;
    const int n_m_tiles = (M + kWRMTile - 1) / kWRMTile;
    dim3 grid((unsigned) n_tokens, (unsigned) n_m_tiles, 1u);
    dim3 block((unsigned) kWRThreads, 1u, 1u);
    streamllm_ext::launch_diag::note_launch(
        streamllm_ext::launch_diag::Kind::WeightedReduce);
    k_weighted_reduce_slots<<<grid, block, 0, (cudaStream_t) stream>>>(
        slot_out, weights, layer_out, n_tokens, n_used, M);
}

// ─── Capture-mode plan kernel ────────────────────────────────────────
//
// One thread block per expert. Threads in the block stride-scan the
// (n_tokens × n_used) ids matrix looking for routes to `eid`; for each
// match they compute the gate score (weights[t,u] if available, else
// probs[t, eid], else a deterministic fallback that matches the host
// plan's "no real scores" branch), and reduce per-block via shared-
// memory atomicMax on the float bit pattern. After the reduction, thread
// 0 maps the max gate to chunks → planes and writes
// planes_per_eid_d[eid].
//
// `g` is non-negative (renormalised weights / probs from softmax), so
// the `__float_as_uint` ordering matches the actual float ordering for
// atomicMax. Sentinel "no route found" leaves the block's s_max_bits at
// 0 → falls into the "unrouted: max planes" branch, which matches the
// host pre-fill behaviour in MoEMatMulComp::on_install.
namespace {

__global__ void k_plan_per_expert_planes(
    const int32_t * __restrict__ ids,
    const float   * __restrict__ weights,
    const float   * __restrict__ probs,
    const float   * __restrict__ thresholds,
    int n_tokens,
    int n_used,
    int n_expert,
    int n_tiers,
    int n_chunks_max,
    int base_p,
    int any_precision,
    int * __restrict__ planes_per_eid)
{
    const int eid = (int) blockIdx.x;
    if (eid >= n_expert) return;

    __shared__ unsigned int s_max_bits;
    __shared__ int          s_any;
    if (threadIdx.x == 0) {
        s_max_bits = 0u;          // __float_as_uint(0.0f)
        s_any      = 0;
    }
    __syncthreads();

    const int total = n_tokens * n_used;
    for (int idx = (int) threadIdx.x; idx < total; idx += (int) blockDim.x) {
        const int e = ids[idx];
        if (e != eid) continue;
        atomicExch(&s_any, 1);

        float g;
        if (weights != nullptr) {
            g = weights[idx];
        } else if (probs != nullptr) {
            const int t = idx / n_used;
            g = probs[t * n_expert + eid];
        } else {
            const int u = idx % n_used;
            g = 0.45f - 0.05f * (float) u;
            if (g < 0.05f) g = 0.05f;
        }
        if (g < 0.0f) g = 0.0f;   // keep __float_as_uint ordering valid
        atomicMax(&s_max_bits, __float_as_uint(g));
    }
    __syncthreads();

    if (threadIdx.x != 0) return;

    int chunks;
    if (s_any == 0) {
        chunks = n_chunks_max;
    } else {
        const float g = __uint_as_float(s_max_bits);
        int k = -1;
        for (int i = n_tiers - 1; i >= 0; --i) {
            if (g >= thresholds[i]) { k = i; break; }
        }
        chunks = (k < 0) ? n_chunks_max : (k + 1);
        if (chunks < 1)            chunks = 1;
        if (chunks > n_chunks_max) chunks = n_chunks_max;
    }
    int planes = any_precision ? (base_p + chunks - 1) : chunks;
    if (planes < 1) planes = 1;
    planes_per_eid[eid] = planes;
}

}  // anon

void launch_plan_per_expert_planes(
    const int32_t * ids_d,
    const float   * weights_d,
    const float   * probs_d,
    const float   * thresholds_d,
    int             n_tokens,
    int             n_used,
    int             n_expert,
    int             n_tiers,
    int             n_chunks_max,
    int             base_p,
    bool            any_precision,
    int *           planes_per_eid_d,
    StreamHandle    stream)
{
    if (n_expert <= 0 || planes_per_eid_d == nullptr) return;
    if (ids_d == nullptr || thresholds_d == nullptr) return;
    if (n_tokens <= 0 || n_used <= 0 || n_tiers <= 0) return;
    const unsigned grid  = (unsigned) n_expert;
    const unsigned block = 256u;
    k_plan_per_expert_planes<<<grid, block, 0, (cudaStream_t) stream>>>(
        ids_d, weights_d, probs_d, thresholds_d,
        n_tokens, n_used, n_expert, n_tiers,
        n_chunks_max, base_p, any_precision ? 1 : 0,
        planes_per_eid_d);
}

}}  // namespace streamllm_ext::qwen3
