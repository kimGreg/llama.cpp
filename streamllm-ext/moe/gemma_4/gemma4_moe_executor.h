// streamllm-ext / moe / gemma_4 — per-arch executor subclass.
//
// Gemma 4 MoE (e.g. google/gemma-4-26B-A4B-it) shares the arch-
// neutral ``MoEAnyBcqExecutor`` pipeline but uses GELU (tanh-approx,
// matching HF transformers' ``gelu_pytorch_tanh``) for the SwiGLU
// stage instead of SiLU.  This subclass is where future per-arch
// divergence will land — Gemma 4 has parallel dense MLP on MoE
// layers and a custom logit projection, both of which are handled
// caller-side today (in ``src/models/gemma4-iswa.cpp``) outside the
// sentinel rail.

#pragma once

#include "moe_executor.h"

namespace streamllm_ext { namespace gemma_4 {

class Gemma4MoEExecutor : public qwen3::MoEAnyBcqExecutor {
public:
    Gemma4MoEExecutor() : qwen3::MoEAnyBcqExecutor() {
        // Gemma 4 uses GELU (tanh-approx).
        set_activation(qwen3::Activation::GELU);
    }
    const char * name() const override { return "gemma4_anybcq_v1"; }
};

// Registry hookup.  Idempotent.
void register_gemma4_moe_executor();

}}  // namespace streamllm_ext::gemma_4
