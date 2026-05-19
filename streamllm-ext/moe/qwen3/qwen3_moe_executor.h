// streamllm-ext / moe / qwen3 — per-arch executor subclass.
//
// Today this subclass is a one-liner over the arch-neutral
// ``MoEAnyBcqExecutor`` in ``moe/common/``: it just sets the
// gated-activation default (SiLU).  When per-arch math diverges from
// the common base — e.g. Qwen3.5+ adds a routing-bias term, or a
// future Qwen variant ships a different MoE kernel — the override
// lands here.

#pragma once

#include "moe_executor.h"

namespace streamllm_ext { namespace qwen3 {

class Qwen3MoEExecutor : public MoEAnyBcqExecutor {
public:
    Qwen3MoEExecutor() : MoEAnyBcqExecutor() {
        // Qwen3-MoE / Qwen3.5 / Qwen3.6 — all SwiGLU + SiLU.
        set_activation(Activation::SiLU);
    }
    const char * name() const override { return "qwen3_anybcq_v1"; }
};

// Registry hookup. Idempotent.  Registers:
//   qwen3_anybcq_v1       (current Qwen3-MoE name)
//   qwen3_moe_anybcq_v1   (legacy alias — old Qwen3 GGUFs)
//   qwen35_anybcq_v1      (Qwen3.5 family, same arch)
//   qwen36_anybcq_v1      (Qwen3.6 family, same arch)
void register_qwen3_moe_executor();

}}  // namespace streamllm_ext::qwen3
