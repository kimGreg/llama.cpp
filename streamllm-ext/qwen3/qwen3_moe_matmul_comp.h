// streamllm-ext / qwen3 — Qwen3-MoE mul_mat_id as a ChunkedComputation.
//
// One ``MoEMatMulComp`` per managed MoE canonical (one of
// ``blk.<L>.ffn_{gate,up,down}_exps.weight``).  ``Runtime::run``
// drives the universal lifecycle:
//
//   pre_inputs() — captured D2H of ids / probs / renorm-weights into
//                  this comp's pinned buffers.  Called once at graph
//                  recording time; replays re-fire the captured copies.
//   plan()       — pure host.  Reads the pinned input buffers,
//                  computes per-expert max gate score, asks the
//                  scheduler for a per-expert chunk plan, fills the
//                  ``host_n_chunks_per_expert_`` array (size n_experts).
//                  Returns the chunk-indexed load_set and required_set.
//   execute()    — F32→F16 cast of src1, ids D2D into the per-stream
//                  scratch, zero dst, then delegates the H2D-prec-array
//                  + kernel launch to ``anybcq::moe_chunk_matmul``,
//                  which owns the chunks→planes translation and the
//                  fused MoE GEMV launch.
//
// Layer boundary (SSOT §6.1.5 ideal flow; constraints 1 + 5):
//
//   * This TU is in the **model** layer (``qwen3/``) and works in
//     CHUNKS only.  It never names planes, never reads base_p, and
//     never owns the kernel-facing per-expert precision array's
//     contents — that array is decoder-fillable scratch, opaque
//     here.
//
//   * The plane↔chunk translation, the H2D of the kernel-facing
//     precision array, the any-prec q_bias refresh, and the fused
//     MoE GEMV launch all live in ``decoder/anybcq/chunked_matmul.cu``.

#pragma once

#include "computation.h"
#include "qwen3_moe_fused.h"   // MoeExpertTable
#include "vram_pool.h"          // StreamHandle / EventHandle
#include "scheduler.h"          // streamllm_ext::Scheduler

#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

struct ggml_tensor;

namespace streamllm_ext {

class StreamllmRuntime;
struct UpstreamLayoutDevice;

namespace qwen3 {

// Per-dispatch payload built by the dispatch shim from the
// ggml-cuda mul_mat_id call's tensors.  The runtime hands a ref to
// ``MoEMatMulComp::pre_inputs / plan / execute`` via the
// ``ComputationInput`` base.
struct MoEInput : ComputationInput {
    const ggml_tensor * src0    = nullptr;  // canonical expert-stack [K, M, n_experts]
    const ggml_tensor * src1    = nullptr;  // routed activations
    const ggml_tensor * ids     = nullptr;  // [n_used_per_tok, n_tokens] int32
    const ggml_tensor * probs   = nullptr;  // [n_expert_in_probs, n_tokens] float (or null)
    const ggml_tensor * weights = nullptr;  // [n_used_per_tok, n_tokens] float (or null)
    int  K              = 0;
    int  M              = 0;
    int  n_tokens       = 0;
    int  n_used_per_tok = 0;
    int  n_expert_in_probs = 0;
    bool shared_x       = true;
    int  layer_index    = -1;
};

struct MoEOutput : ComputationOutput {
    ggml_tensor * dst = nullptr;
};

class MoEMatMulComp : public ChunkedComputation {
public:
    MoEMatMulComp(StreamllmRuntime &           rt,
                   Scheduler &                 sched,
                   std::string                 canonical,
                   const MoeExpertTable *      fuse_table,
                   const UpstreamLayoutDevice * any_layout,
                   int                         n_experts,
                   int                         max_n_tokens,
                   int                         max_n_used);
    ~MoEMatMulComp() override;

    MoEMatMulComp(const MoEMatMulComp &) = delete;
    MoEMatMulComp & operator=(const MoEMatMulComp &) = delete;

    // ChunkedComputation contract.
    std::string             state_key() const override { return canonical_; }
    std::vector<MemcpySpec> pre_inputs(const ComputationInput & in) override;
    ChunkPlan               plan(const ComputationInput & in)        override;
    void                    execute(const ComputationInput & in,
                                     ComputationOutput &      out,
                                     StreamHandle             stream) override;

    bool valid() const { return fuse_table_ != nullptr && any_layout_ != nullptr; }

    // Public read-only views of the canonical's static shape. The
    // executor's per-layer entry point (forward_moe_layer) queries
    // these on the three canonicals (gate/up/down) for a layer so it
    // can size the per-slot scratch and shape the synthesized
    // src1/dst views without exposing the per-comp internals.
    int K() const { return K_; }
    int M() const { return M_; }
    int n_experts() const { return n_experts_; }

    // Capture-mode hooks. The on-device plan kernel writes per-expert
    // PLANE counts directly into ``prec_per_eid_d_``; ``execute`` is
    // then called with host_n_chunks_per_expert_ unused. The accessor
    // surfaces the device pointer + the layout fields the kernel needs.
    int *                         prec_per_eid_device() { return (int *) prec_per_eid_d_; }
    const UpstreamLayoutDevice *  any_layout()    const { return any_layout_; }
    int                           n_chunks_max()  const { return n_chunks_; }
    int                           group_size()    const { return group_size_; }

private:
    StreamllmRuntime *           rt_         = nullptr;
    Scheduler *                  sched_      = nullptr;
    std::string                  canonical_;
    int                          layer_index_ = -1;

    const MoeExpertTable *       fuse_table_ = nullptr;
    const UpstreamLayoutDevice * any_layout_ = nullptr;

    // Cached canonical config (constant for this comp).  Only
    // chunk-side metadata is kept here; plane-side fields (base_p,
    // any_precision flag) belong to the decoder and are pulled from
    // ``rt_->layout(canonical_)`` inside ``decoder/anybcq``.
    int  n_experts_         = 0;
    int  K_                 = 0;
    int  M_                 = 0;
    int  n_chunks_          = 0;  // encoder max chunks; clamps score_lookup
    int  group_size_        = 0;

    // Pinned host buffers.  Sized at construction; never reallocated.
    int32_t * ids_pinned_              = nullptr;  // [max_n_tokens × max_n_used] int32
    float   * probs_pinned_            = nullptr;  // [max_n_tokens × n_experts] float
    float   * weights_pinned_          = nullptr;  // [max_n_tokens × max_n_used] float

    // Per-expert chunk count the dispatch will require resident
    // (constraint 1 rename).  plan() writes one entry per routed
    // expert; execute() passes this to anybcq::moe_chunk_matmul.
    int     * host_n_chunks_per_expert_ = nullptr;  // [n_experts] int

    // Device scratch the decoder fills with kernel-facing per-expert
    // precision (planes).  Opaque to the model — the decoder owns
    // both the H2D and the chunks→planes conversion that populates
    // it.  Allocated here so it lives for the comp's lifetime.
    void    * prec_per_eid_d_           = nullptr;  // device, [n_experts × int]

    int max_n_tokens_ = 0;
    int max_n_used_   = 0;

    // Per-dispatch shape state — set by pre_inputs() at graph recording
    // time, read by plan() at every replay and execute() at recording.
    // Captured graphs re-instantiate when shapes change, so each
    // recording resets these.
    int  cur_n_tokens_       = 0;
    int  cur_n_used_         = 0;
    int  cur_n_expert_       = 0;
    bool have_real_scores_   = false;
    bool have_renorm_weights_ = false;
};

// Free-fn lookup the dispatch shim uses to find a comp from a canonical
// tensor name.  Returns nullptr if no comp is registered for that name
// (i.e. the canonical isn't a managed MoE expert-stack tensor).
MoEMatMulComp * scheduler_lookup_moe_comp(
    Scheduler &         sched,
    const std::string & canonical_wid);

}  // namespace qwen3
}  // namespace streamllm_ext
