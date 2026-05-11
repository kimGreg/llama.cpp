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

#include <string>

namespace streamllm_ext { namespace qwen3 {

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

private:
    StreamllmRuntime * rt_ = nullptr;
};

// Static-init registration.  Called from qwen3_runtime_glue's
// ``install_for_gguf`` (idempotent) so the executor is registered
// before ``make_executor("qwen3_moe_anybcq_v1")`` runs.
void register_qwen3_moe_anybcq_executor();

}}  // namespace streamllm_ext::qwen3
