// streamllm-ext / qwen3 — Qwen3 MoE AnyBCQ executor.
//
// Concrete ModelExecutor implementation for Qwen3-MoE streamed
// through the AnyBCQ chunk encoder.  Selected at llama_model_load
// by ``streamllm.executor = "qwen3_moe_anybcq_v1"`` in the GGUF.
//
// ``forward_moe_block`` is invoked once per managed mul_mat_id
// dispatch by qwen3_runtime_glue's hook surface (today's integration
// point; a future ggml custom-op / arch-builder patch replaces the
// hook entry point without changing the executor body — see
// SSOT §6.4.2).  Inside the call, the scheduler runs with
// current-batch routing, loads missing chunks, waits on copy_stream,
// then launches the fused MoE kernel.

#pragma once

#include "executor.h"

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

namespace streamllm_ext {

struct ChunkKey;        // core/vram_pool.h
struct ChunkPlan;       // core/computation.h

namespace qwen3 {

class Qwen3MoEAnyBcqExecutor : public ModelExecutor {
public:
    Qwen3MoEAnyBcqExecutor() = default;

    const char * name() const override { return "qwen3_moe_anybcq_v1"; }

    void bind_to_model(StreamllmRuntime &  rt,
                        const StreamReader & reader,
                        const std::string &  gguf_path) override;

    bool forward_moe_block(StreamHandle               stream,
                            const ggml_tensor *        src0,
                            const ggml_tensor *        src1,
                            const ggml_tensor *        ids,
                            ggml_tensor *              dst) override;

    // Step 6 (Milestone 1): pre-launch host-side validation.
    // Walks ``plan.required_set`` and asserts every chunk is at
    // ChunkState::POINTER_TABLE_READY before the fused MoE kernel
    // launches. Debug builds GGML_ABORT on the first miss with a
    // detailed message; release builds increment the global counter
    // and emit a rate-limited structured log, then return false so
    // the executor surfaces the failure to the caller. The M1
    // kernel trap remains the final guard regardless.
    //
    // Returns true when every required chunk is kernel-ready;
    // false (release build only) when one or more chunks failed
    // the check.
    bool validate_required_set_(const ChunkPlan & plan,
                                 StreamHandle      stream) const;

    // Process-wide counter of required-set misses (chunks that
    // weren't at POINTER_TABLE_READY at validate time). Exposed via
    // the extern-C ``streamllm_stat_required_set_misses`` accessor.
    static std::atomic<std::uint64_t> required_set_misses_;

private:
    // Walk one or two parents up from ``ids`` to find the F32 probs
    // tensor produced by the router. Returns nullptr when the topology
    // doesn't match (e.g. ids->src[0] is not the gate output).
    // Topology is graph-stable so this probe is cheap on every call.
    static const ggml_tensor * probe_probs_tensor_(const ggml_tensor * ids,
                                                    int n_tokens);

    StreamllmRuntime * rt_ = nullptr;
};

// Static-init registration.  Called from qwen3_runtime_glue's
// ``install_for_gguf`` (idempotent) so the executor is registered
// before ``make_executor("qwen3_moe_anybcq_v1")`` runs.
void register_qwen3_moe_anybcq_executor();

}  // namespace qwen3
}  // namespace streamllm_ext
