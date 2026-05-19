// streamllm-ext / qwen3 — Qwen3 MoE AnyBCQ executor.
//
// Concrete ModelExecutor implementation for Qwen3-MoE streamed
// through the AnyBCQ chunk encoder.  Selected at llama_model_load
// by ``streamllm.executor = "qwen3_moe_anybcq_v1"`` in the GGUF.
//
// ``forward_moe_layer`` is invoked once per managed MoE layer by the
// per-layer sentinel rail (qwen3_runtime_glue's ``streamllm_pre_op``
// matches on the ``"streamllm.moe_layer_<L>"`` name set by the
// arch-builder helper ``llm_build_moe_sentinel``).  Inside the call,
// the executor runs three chunked matmuls (gate / up / down) with
// current-batch routing — D2H ids/probs/weights, plan, reserve,
// load, wait, validate, fused kernel — plus the in-place SwiGLU
// elementwise and the final weighted reduce into ``layer_out``.

#pragma once

#include "executor.h"          // ModelExecutor base
#include "fused_kernels.h"     // Activation

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace streamllm_ext {

struct ChunkKey;        // core/vram_pool.h
struct ChunkPlan;       // core/computation.h

namespace qwen3 {
class MoEMatMulComp;    // qwen3/matmul_comp.h

// Per-stream slot scratch consumed by forward_moe_layer (S6). One
// pair of F32 device buffers per compute stream: ``slot_a`` is the
// gate output AND, after the down matmul, the per-slot M=hidden_dim
// output that the weighted reduce consumes; ``slot_b`` is the up
// output and, after SwiGLU writes in-place, the per-slot M=n_ff f32
// down input. Sizes are upper bounds (max_n_tokens × max_n_used ×
// max-M). Both are F32 because the comp's execute() does the F32→F16
// cast of src1 internally — keeping the SwiGLU+swap dance in F32
// avoids a separate gated_f16 buffer.
struct SlotScratch {
    void *  slot_a       = nullptr;
    void *  slot_b       = nullptr;
    size_t  slot_a_bytes = 0;
    size_t  slot_b_bytes = 0;
};

class Qwen3MoEAnyBcqExecutor : public ModelExecutor {
public:
    Qwen3MoEAnyBcqExecutor() = default;
    ~Qwen3MoEAnyBcqExecutor() override;

    const char * name() const override { return "qwen3_moe_anybcq_v1"; }

    void bind_to_model(StreamllmRuntime &  rt,
                        const StreamReader & reader,
                        const std::string &  gguf_path) override;

    // S8 (Mode A): forward_moe_block override retired with the
    // legacy MUL_MAT_ID rail. Only forward_moe_layer remains.

    bool forward_moe_layer(StreamHandle               stream,
                            const ggml_tensor *        layer_in,
                            const ggml_tensor *        ids,
                            const ggml_tensor *        probs,
                            const ggml_tensor *        weights,
                            ggml_tensor *              layer_out,
                            int                        layer_idx) override;

    // M1 cutover S2: router-gate binding (criterion 5). Future-proof
    // plumbing — M1 correctness does NOT depend on the executor
    // having direct access to ffn_gate_inp (the arch-builder Mode A
    // branch in S5 emits router/topk via the shared helper). This
    // override caches the pointers for a post-M1 move that
    // relocates router/topk construction into the executor at
    // execute time.
    void attach_router_gates(
        const struct ggml_tensor * const * ffn_gate_inp_per_layer,
        int                                n_layer) override;

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

    // Per-arch activation for the SwiGLU stage between gate/up and
    // down matmuls.  Default SiLU matches Qwen3-MoE + DeepSeek-V2;
    // Gemma 4 MoE installs GELU at bind time.  Setter is host-only
    // and read once per ``forward_moe_layer`` call.
    void set_activation(Activation a) { activation_ = a; }
    Activation activation() const { return activation_; }

private:
    // ``probe_probs_tensor_`` was retired with the legacy
    // ``forward_moe_block`` body — the sentinel carries probs through
    // ``src[2]`` directly.

    StreamllmRuntime * rt_ = nullptr;
    Activation activation_ = Activation::SiLU;

    // M1 cutover S2: cached per-layer ffn_gate_inp pointers from
    // attach_router_gates. Empty until S5 wires the install path
    // to call attach_router_gates with the model's layers; the
    // attach is optional and M1 doesn't read from this vector
    // (router/topk is built by the arch builder in S5).
    std::vector<const struct ggml_tensor *> ffn_gate_inp_;

    // M1 cutover S6: per-layer slot scratch. One pair of F32 device
    // buffers per compute stream. Allocated lazily on the first
    // forward_moe_layer call for a given stream (we need observed
    // n_tokens / n_used to size them) and never freed until the
    // executor is destroyed. Sized as upper bounds — a later call
    // with a larger shape reallocates.
    std::unordered_map<unsigned long long, SlotScratch> slot_scratch_;
    std::mutex                                          slot_scratch_mu_;

    SlotScratch * acquire_slot_scratch_(unsigned long long stream_key,
                                         int                n_tokens,
                                         int                n_used,
                                         int                M_gate_up,
                                         int                M_down);

    // Batched per-layer dispatch.  Plans, loads, waits, and validates
    // gate/up/down in a single coalesced phase, then runs the three
    // matmuls (with SwiGLU between up and down + the final weighted
    // reduce) back-to-back.  One ``cudaStreamSynchronize`` and one
    // ``wait_async_load_batch`` per forward_moe_layer instead of one
    // per canonical; the io-worker pool also sees a single larger
    // batch to parallelise across.  This is the only dispatch path.
    bool dispatch_three_canonicals_(
        qwen3::MoEMatMulComp & gate_comp,
        qwen3::MoEMatMulComp & up_comp,
        qwen3::MoEMatMulComp & down_comp,
        const std::string &    gate_canonical,
        const std::string &    up_canonical,
        const std::string &    down_canonical,
        StreamHandle           stream_h,
        const struct ggml_tensor * cur_3d,
        const struct ggml_tensor * gated_3d,
        const struct ggml_tensor * ids,
        const struct ggml_tensor * probs,
        const struct ggml_tensor * weights,
        const struct ggml_tensor * layer_in,    // for n_embd
        struct ggml_tensor *       layer_out,
        void *                     slot_a_data,
        void *                     slot_b_data,
        int                        n_tokens,
        int                        n_used,
        int                        n_expert_in_probs,
        int                        n_ff,
        int                        n_embd,
        int                        layer_idx);
};

// Static-init registration.  Called from qwen3_runtime_glue's
// ``install_for_gguf`` (idempotent) so the executor is registered
// before ``make_executor("qwen3_moe_anybcq_v1")`` runs.
void register_qwen3_moe_anybcq_executor();

}  // namespace qwen3
}  // namespace streamllm_ext
