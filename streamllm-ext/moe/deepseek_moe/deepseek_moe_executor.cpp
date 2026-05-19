// streamllm-ext / moe / deepseek_moe — registration.

#include "deepseek_moe_executor.h"
#include "executor.h"

#include <memory>

namespace streamllm_ext { namespace deepseek_moe {

void register_deepseek_moe_executor() {
    static bool once = false;
    if (once) return;
    once = true;
    register_executor(
        "deepseek_anybcq_v1",
        []() -> std::unique_ptr<ModelExecutor> {
            return std::unique_ptr<ModelExecutor>(new DeepSeekMoEExecutor());
        });
}

}}  // namespace streamllm_ext::deepseek_moe
