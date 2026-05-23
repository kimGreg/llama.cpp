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

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace streamllm_ext { namespace qwen3 {

class Qwen3MoEExecutor : public MoEAnyBcqExecutor {
public:
    Qwen3MoEExecutor() : MoEAnyBcqExecutor() {
        // Qwen3-MoE / Qwen3.5 / Qwen3.6 — all SwiGLU + SiLU.
        set_activation(Activation::SiLU);
    }
    const char * name() const override { return "qwen3_ss_anybcq_v1"; }
};

class Qwen3BaselineExecutor : public ModelExecutor {
public:
    const char * name() const override { return "qwen3_direct_stock_v1"; }
    bool uses_stock_moe_graph() const override { return true; }

    void bind_to_model(StreamllmRuntime & rt,
                       const StreamReader & reader,
                       const std::string & gguf_path) override;

    bool forward_moe_layer(StreamHandle,
                           const ggml_tensor *,
                           const ggml_tensor *,
                           const ggml_tensor *,
                           const ggml_tensor *,
                           ggml_tensor *,
                           int) override;

    bool prepare_moe_mul_mat_id(StreamHandle stream,
                                ggml_tensor * dst) override;

private:
    struct LayerState {
        const void * ids_data = nullptr;
        int n_used = 0;
        int n_tokens = 0;
        std::vector<int> active_experts;
        size_t slice_bytes[3] = {0, 0, 0};
    };

    StreamllmRuntime * rt_ = nullptr;
    std::unordered_map<int, LayerState> layers_;
    std::string staging_wid_[3];
    void * staging_[3] = {nullptr, nullptr, nullptr};
    size_t staging_bytes_[3] = {0, 0, 0};

    void allocate_staging_(int kind, size_t bytes);
};

// Registry hookup. Idempotent.
void register_qwen3_moe_executor();

}}  // namespace streamllm_ext::qwen3
