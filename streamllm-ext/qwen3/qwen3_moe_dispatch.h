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

struct ggml_tensor;

namespace streamllm_ext {

class StreamReader;

namespace moe_dispatch {

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

} // namespace moe_dispatch
} // namespace streamllm_ext
