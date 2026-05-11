// streamllm-ext — Qwen3-MoE ggml-cuda hook bodies.
//
// Owned by the scheduler architecturally: MoEScheduler::handle_mul_mat,
// handle_mul_mat_id and on_topk_moe_observed forward to the *_impl
// functions declared here. qwen3/qwen3_runtime_glue.cpp's extern-C shims also
// reach the scheduler via Scheduler::handle_*; the scheduler then calls
// these impls.
//
// Lifecycle helpers (size_scratch, free_scratch, print_profile,
// clear_topk_weights) are called by qwen3/qwen3_runtime_glue.cpp at install
// and clear time.

#pragma once

#include <cuda_runtime.h>
#include <cstddef>

struct ggml_tensor;

namespace streamllm_ext {

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

bool handle_mul_mat_impl(
    cudaStream_t stream,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    struct ggml_tensor * dst);

bool handle_mul_mat_id_impl(
    cudaStream_t stream,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * ids,
    struct ggml_tensor * dst);

void on_topk_moe_observed_impl(
    cudaStream_t stream,
    const struct ggml_tensor * logits,
    struct ggml_tensor *       weights,
    struct ggml_tensor *       ids);

// Topk-weights side channel.  ggml-cuda's fused topk_moe captures
// (ids, weights) here so the dispatch path can later read the post-
// norm routing weights.  Returns true if a weights tensor is
// associated with ``ids_data`` (stable for the lifetime of one
// forward pass / cuda-graph capture session).
bool topk_weights_lookup(
    const void *               ids_data,
    const struct ggml_tensor ** out_weights,
    int *                       out_n_used);

// Per-stream cast scratch sizing. Called once at install_for_gguf;
// returns false on alloc failure (caller aborts install).
bool size_scratch_for(const StreamReader & r);

// Frees per-stream scratch on clear(). Idempotent.
void free_scratch();

// Dumps the MoE profiling counters when STREAMLLM_PROFILE=1. Called
// from clear() before the runtime is torn down.
void print_profile_if_enabled();

// Drops any captured (ids, weights) handles from the topk_moe side
// channel. Called from clear() so a future install starts fresh.
void clear_topk_weights();

// Step 3 (Milestone 1): increment the MoE dispatch entry counter. The
// counter itself stays in qwen3_moe_dispatch.cpp alongside the rest of
// the profile counters; the executor calls this from
// forward_moe_block's entry now that the dispatch body lives there.
// No-op when STREAMLLM_PROFILE is unset (the counter just isn't
// printed).
void profile_inc_hook_calls();

} // namespace moe_dispatch
} // namespace streamllm_ext
