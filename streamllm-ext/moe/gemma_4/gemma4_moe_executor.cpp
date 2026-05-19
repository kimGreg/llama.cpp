// streamllm-ext / moe / gemma_4 — registration.

#include "gemma4_moe_executor.h"
#include "executor.h"

#include <memory>

namespace streamllm_ext { namespace gemma_4 {

void register_gemma4_moe_executor() {
    static bool once = false;
    if (once) return;
    once = true;
    register_executor(
        "gemma4_anybcq_v1",
        []() -> std::unique_ptr<ModelExecutor> {
            return std::unique_ptr<ModelExecutor>(new Gemma4MoEExecutor());
        });
}

}}  // namespace streamllm_ext::gemma_4
