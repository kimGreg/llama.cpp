// streamllm-ext / qwen3 — Qwen3 MoE AnyBCQ executor implementation.
//
// First-milestone delivery: ``forward_moe_block`` delegates to the
// existing ``moe_dispatch::handle_mul_mat_id_impl``, which already
// runs the canonical Mode A flow (current-batch routing → plan →
// reserve → load → wait → kernel → release).  The executor
// abstraction is the integration boundary — future passes can move
// that body into this file without touching the call sites
// (qwen3_runtime_glue.cpp and any future ggml custom-op handler).

#include "qwen3_moe_executor.h"
#include "qwen3_moe_dispatch.h"
#include "runtime.h"

#include <cuda_runtime.h>

namespace streamllm_ext { namespace qwen3 {

void Qwen3MoEAnyBcqExecutor::bind_to_model(
    StreamllmRuntime &  rt,
    const StreamReader & /*reader*/,
    const std::string &  /*gguf_path*/)
{
    rt_ = &rt;
}

bool Qwen3MoEAnyBcqExecutor::forward_moe_block(
    StreamHandle stream,
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    const ggml_tensor * ids,
    ggml_tensor *       dst)
{
    if (rt_ == nullptr) return false;
    return moe_dispatch::handle_mul_mat_id_impl(
        (cudaStream_t) stream, src0, src1, ids, dst);
}

void register_qwen3_moe_anybcq_executor() {
    static bool once = false;
    if (once) return;
    once = true;
    register_executor(
        "qwen3_moe_anybcq_v1",
        []() -> std::unique_ptr<ModelExecutor> {
            return std::unique_ptr<ModelExecutor>(
                new Qwen3MoEAnyBcqExecutor());
        });
}

}}  // namespace streamllm_ext::qwen3
