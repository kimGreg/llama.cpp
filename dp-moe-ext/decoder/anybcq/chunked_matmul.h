// DPMoE / decoder / anybcq — MoE chunked-matmul dispatch.
//
// ────────────────────────────────────────────────────────────────────
// Owns the plane↔chunk translation for the AnyBCQ encoder family.
// ────────────────────────────────────────────────────────────────────
//
// Layering contract (mirrors decoder/ss_anybcq/chunked_matmul.h):
//
//   * Caller (qwen3/MoEMatMulComp) passes CHUNK COUNTS per expert.
//     The model layer never thinks in planes; the score-dial maps
//     gate-score tiers to chunk counts and that's the only unit it
//     uses (cf. SSOT §6.1.5 ideal flow: "current-batch routing →
//     scheduler.plan(required chunks) → cap-aware residency").
//
//   * This decoder helper converts CHUNKS → PLANES internally using
//     ``UpstreamLayoutDevice::base_precision`` and ``::n_chunks``,
//     prepares the per-expert plane-count array the fused kernel
//     reads (``prec_per_eid_d``), refreshes any-prec q_bias slots,
//     and launches ``naver_gemv_moe_launch``.
//
//   * Asserts (constraint 6):
//       1 <= host_n_chunks_per_eid[eid] <= layout.n_chunks
//       planes := base_p + n_chunks - 1  for any-prec
//                 n_chunks                for ss_anybcq/static
//       1 <= planes <= kMaxChunksPerTensor
//     Out-of-range entries abort in debug, clamp + count in release.
//
//   * Counters (constraint 7):
//       g_anybcq_moe_chunk_matmul_calls    incremented on every entry
//       g_anybcq_moe_kernel_launch_calls   incremented before
//                                          naver_gemv_moe_launch
//       g_anybcq_n_chunks_oob              chunk-count out-of-range
//
// All CUDA work goes on the caller-supplied compute stream. No
// per-call cudaMalloc — uses pinned host scratch carried by the
// caller (host_n_chunks_per_eid) and device scratch allocated by
// the caller (prec_per_eid_d). The function is reentrant per stream.

#pragma once

#include "fused_kernels.h"   // MoeExpertTable, naver_gemv_moe_launch decl
#include "vram_pool.h"         // StreamHandle

#include <cstddef>
#include <cstdint>
#include <string>

namespace dp_moe_ext {

class DPMoERuntime;
struct UpstreamLayoutDevice;

namespace anybcq {

// MoE per-canonical fused LUT-GEMV launcher.
//
// PRE: caller has externally:
//   - reserved + loaded + wait-on-stream'd every chunk in
//     ``required_set`` (cids 0..host_n_chunks_per_eid[eid]-1 per
//     routed expert),
//   - validated required_set residency (the M1 contract; the kernel
//     traps on null required plane as the device-side backstop),
//   - cast src1 (F32) into the caller's ``X_fp16`` scratch,
//   - zeroed ``Y_slot_f32``.
//
// IN:
//   rt                    : runtime (for diagnostics; not consulted for layout)
//   canonical             : tensor wid (e.g. "blk.5.ffn_gate_exps.weight"),
//                           used only for log messages
//   layout                : the canonical's UpstreamLayoutDevice (carries
//                           base_precision, n_chunks, any_precision).
//                           Caller (MoEMatMulComp) already holds this from
//                           construction; rt.layout(canonical) wouldn't
//                           resolve because layouts register per-expert.
//   table                 : the canonical's MoeExpertTable (per-expert
//                           plane pointer arrays).
//   prec_per_eid_d        : device buffer [n_experts × int]. This
//                           function writes per-expert PLANE counts
//                           here via H2D (chunks→planes done inside).
//                           Must be non-null.
//   host_n_chunks_per_eid : pinned host buffer [n_experts × int]. The
//                           caller's per-expert chunk count, from
//                           plan(). Asserted in [1, layout.n_chunks].
//   ids_d                 : [n_tokens × n_used] int32 device tensor
//                           (the routing).
//   X_fp16                : shared_x=1 → [n_tokens, K] half device
//                           tensor, else [n_tokens, n_used, K]
//   Y_slot_f32            : [n_tokens, n_used, M] f32, pre-zeroed
//   n_tokens, n_used      : current packed batch shape
//   n_experts             : table size
//   M, K                  : canonical's static shape
//   group_size            : encoder param (from layout)
//   shared_x              : routing-fusion flag
//   routed_chunk_span_hint
//                         : optional K_max-K_min over routed experts in
//                           CHUNKS; -1 if unknown. The decoder translates
//                           this to a plane span for kernel selection.
//   routed_max_chunk_hint
//                         : optional K_max over routed experts in CHUNKS;
//                           -1 if unknown. Used only to size mixed
//                           flat-kernel plane grids after decoder-side
//                           chunks->planes conversion.
//   stream                : caller's compute stream
//
// Returns true on success (kernel launched). Returns false if the
// layout cannot be found, the prec buffer is null, or a chunk count
// is out of range and cannot be clamped. On a false return the kernel
// is NOT launched and the caller is responsible for diagnostic logging.
bool moe_chunk_matmul(
    DPMoERuntime &           rt,
    const std::string &          canonical,
    const UpstreamLayoutDevice & layout,
    const MoeExpertTable &       table,
    int *                    prec_per_eid_d,
    const int *              host_n_chunks_per_eid,
    const int32_t *          ids_d,
    const void *             X_fp16,
    void *                   Y_slot_f32,
    int                      n_tokens,
    int                      n_used,
    int                      n_experts,
    int                      M,
    int                      K,
    int                      group_size,
    int                      shared_x,
    int                      routed_chunk_span_hint,
    int                      routed_max_chunk_hint,
    StreamHandle             stream);


// =====================================================================
// Correctness counters (constraint 7).  Process-wide atomics; reset by
// the M1 ``clear()`` path alongside the other DPMoE counters.
// =====================================================================

// Number of entries to ``moe_chunk_matmul`` (whether or not the kernel
// actually launched).  A non-zero value at end-of-run confirms the
// model layer's execute() is calling through the decoder interface.
unsigned long long counter_chunk_matmul_calls();

// Number of times ``naver_gemv_moe_launch`` actually fired.  Should
// equal ``counter_chunk_matmul_calls()`` minus early returns (bad
// chunk counts, missing layout).  Non-zero ⇒ the kernel ran.
unsigned long long counter_kernel_launch_calls();

// Number of (eid, n_chunks) entries seen with n_chunks_per_eid outside
// [1, layout.n_chunks].  Should be zero in a healthy run; non-zero
// indicates a planner bug.
unsigned long long counter_n_chunks_oob();

// Resets all three to zero.  Called from the DPMoE clear path.
void reset_counters();

}  // namespace anybcq
}  // namespace dp_moe_ext
