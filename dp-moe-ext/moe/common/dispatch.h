// DPMoE — Qwen3-MoE ggml-cuda hook bodies.
//
// Owned by the scheduler architecturally: MoEScheduler::handle_mul_mat,
// handle_mul_mat_id and on_topk_moe_observed forward to the *_impl
// functions declared here. qwen3/runtime_glue.cpp's extern-C shims also
// reach the scheduler via Scheduler::handle_*; the scheduler then calls
// these impls.
//
// Lifecycle helpers (size_scratch, free_scratch, print_profile,
// clear_topk_weights) are called by qwen3/runtime_glue.cpp at install
// and clear time.

#pragma once

#include <cuda_runtime.h>
#include <cstddef>

struct ggml_tensor;

namespace dp_moe_ext {

class StreamReader;

namespace moe_dispatch {

// Per-stream cast / staging scratch.  Allocated lazily by
// scratch_for_stream(stream) on the first hook firing for that
// stream; sized once at install_for_gguf via size_scratch_for(). The
// dispatch shim and ``MoEMatMulComp::execute`` both consume these
// buffers.
//
// Note: per-expert precision buffer (``prec_per_eid_d``) is owned per
// MoEMatMulComp, not per stream — sized n_experts × int per comp
// (SSOT §6.9 M1).  The dispatch shim does not need a slot for it.
struct StreamScratch {
    void *  x_f16          = nullptr;
    void *  y_f16          = nullptr;
    void *  xb_f16         = nullptr;
    void *  yb_f16         = nullptr;
    void *  w_f16          = nullptr;
    void *  ids_d          = nullptr;
};

StreamScratch * scratch_for_stream(cudaStream_t stream);

// Byte counts used by the comps to size their pinned buffers / the
// kernel arguments.  Stable for the runtime's lifetime once
// size_scratch_for() has been called at install.
size_t scratch_xb_bytes_total();
size_t scratch_yb_bytes_total();
size_t scratch_ids_bytes_total();
int    scratch_batch_n_max();

// Mode A M1 cleanup retired ``handle_mul_mat_impl`` (dense streaming
// body), ``handle_mul_mat_id_impl``, ``on_topk_moe_observed_impl``,
// ``topk_weights_lookup``, and ``clear_topk_weights`` — all part of
// the legacy MUL_MAT_ID / dense interception rails. Sentinel
// dispatch (runtime_glue.cpp ``dp_moe_pre_op`` →
// ``forward_moe_layer``) is the only managed-MoE path.

// Per-stream cast scratch sizing. Called once at install_for_gguf;
// returns false on alloc failure (caller aborts install).
bool size_scratch_for(const StreamReader & r);

// Frees per-stream scratch on clear(). Idempotent.
void free_scratch();

// Dumps the MoE profiling counters when DP_MOE_PROFILE=1. Called
// from clear() before the runtime is torn down.
void print_profile_if_enabled();

// Step 3 (Milestone 1): increment the MoE dispatch entry counter. The
// counter itself stays in dispatch.cpp alongside the rest of
// the profile counters; the executor calls this from
// forward_moe_block's entry now that the dispatch body lives there.
// No-op when DP_MOE_PROFILE is unset (the counter just isn't
// printed).
void profile_inc_hook_calls();

} // namespace moe_dispatch
} // namespace dp_moe_ext
