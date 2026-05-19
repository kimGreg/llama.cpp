// streamllm-ext / moe / qwen3 — Qwen3-MoE executor registration.

#include "qwen3_moe_executor.h"
#include "executor.h"        // register_executor

#include <memory>

namespace streamllm_ext { namespace qwen3 {

void register_qwen3_moe_executor() {
    static bool once = false;
    if (once) return;
    once = true;
    auto factory = []() -> std::unique_ptr<ModelExecutor> {
        return std::unique_ptr<ModelExecutor>(new Qwen3MoEExecutor());
    };
    register_executor("qwen3_anybcq_v1",     factory);
    register_executor("qwen3_moe_anybcq_v1", factory);   // legacy alias
    register_executor("qwen35_anybcq_v1",    factory);
    register_executor("qwen36_anybcq_v1",    factory);
}

}}  // namespace streamllm_ext::qwen3
