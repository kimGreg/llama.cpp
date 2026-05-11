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
//                  ``host_prec_per_expert_`` array (size n_experts —
//                  SSOT §6.9 M1).  Returns the load_set the loader
//                  thread should bring resident.
//   execute()    — kernel launches: F32→F16 cast of src1, ids D2D
//                  into the per-stream scratch, dst zero, H2D of
//                  host_prec_per_expert_ → prec_per_eid_d_,
//                  ``refresh_q_bias_for_anyprec_launch``, and the
//                  fused ``naver_gemv_moe_launch``.  Kernel traps
//                  on null required plane (M1).

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

private:
    StreamllmRuntime *           rt_         = nullptr;
    Scheduler *                  sched_      = nullptr;
    std::string                  canonical_;
    int                          layer_index_ = -1;

    const MoeExpertTable *       fuse_table_ = nullptr;
    const UpstreamLayoutDevice * any_layout_ = nullptr;

    // Cached canonical config (constant for this comp).
    int  n_experts_         = 0;
    int  K_                 = 0;
    int  M_                 = 0;
    int  n_chunks_          = 0;
    int  group_size_        = 0;
    int  base_precision_    = 0;
    bool any_precision_     = false;
    // Static fallback uniform_precision passed to the fused kernel
    // launcher.  When ``prec_per_eid_d_`` is non-null (always, in
    // this path) the kernel ignores ``uniform_precision``; we still
    // pass a valid value so the launcher's [1, 8] range check is
    // satisfied.
    int  uniform_precision_static_ = 0;

    // Pinned host buffers.  Sized at construction; never reallocated.
    int32_t * ids_pinned_           = nullptr;  // [max_n_tokens × max_n_used] int32
    float   * probs_pinned_         = nullptr;  // [max_n_tokens × n_experts] float
    float   * weights_pinned_       = nullptr;  // [max_n_tokens × max_n_used] float
    int     * host_prec_per_expert_ = nullptr;  // [n_experts] int (SSOT §6.9 M1)
    void    * prec_per_eid_d_       = nullptr;  // device, [n_experts × int]

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
