// streamllm-ext / moe / deepseek_moe — per-arch executor subclass.
//
// DeepSeek-MoE V1 (DeepseekForCausalLM, e.g. deepseek-moe-16b-chat)
// shares the arch-neutral ``MoEAnyBcqExecutor`` implementation in
// ``moe/common/``: same chunked AnyBCQ matmul pipeline, SiLU
// activation, same canonical names.  This subclass is the registry
// hookup point + a place for V1-specific extensions if/when they
// arise.  DeepSeek-V2/V3 will be separate subclasses (different
// MLA attention path; V3 needs biased-sigmoid routing).

#pragma once

#include "moe_executor.h"

namespace streamllm_ext { namespace deepseek_moe {

class DeepSeekMoEExecutor : public qwen3::MoEAnyBcqExecutor {
public:
    DeepSeekMoEExecutor() : qwen3::MoEAnyBcqExecutor() {
        // DeepSeek-MoE V1 uses SwiGLU + SiLU (hidden_act=silu).
        set_activation(qwen3::Activation::SiLU);
    }
    const char * name() const override { return "deepseek_anybcq_v1"; }
};

// Registry hookup.  Idempotent.  Registers:
//   deepseek_anybcq_v1   (V1: DeepseekForCausalLM)
void register_deepseek_moe_executor();

}}  // namespace streamllm_ext::deepseek_moe
