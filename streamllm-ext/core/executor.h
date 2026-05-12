// streamllm-ext / core — ModelExecutor ABC.
//
// Model-specific forward executor selected at llama_model_load by the
// GGUF's ``streamllm.executor`` key.  Owns the forward path for the
// model's streamed (chunked) blocks: scheduler invocation with
// current-batch routing, chunk reserve/load/wait, fused kernel
// launch, post-kernel release.  See SSOT §6.6.0.
//
// Invocation point (Mode A M1): ggml-cuda's pre_op_hook matches on
// the per-layer ``"streamllm.moe_layer_<L>"`` sentinel emitted by
// the arch-builder helper ``llm_build_moe_sentinel`` and routes the
// dispatch into ``forward_moe_layer``. No per-canonical interception
// remains.

#pragma once

#include "vram_pool.h"   // StreamHandle

#include <memory>
#include <string>

struct ggml_tensor;

namespace streamllm_ext {

class StreamllmRuntime;
class StreamReader;

class ModelExecutor {
public:
    virtual ~ModelExecutor() = default;

    // Registry name.  Must match the GGUF's ``streamllm.executor``
    // key.  Versioned (e.g. ``"qwen3_moe_anybcq_v1"``) so the executor
    // and encoder can evolve in lockstep.
    virtual const char * name() const = 0;

    // One-time setup at install.  ``rt`` is the owning runtime;
    // ``reader`` is the parsed streamllm.* metadata; ``gguf_path``
    // is the file the managed tensors live in (for SSD-stream).
    virtual void bind_to_model(StreamllmRuntime &  rt,
                                const StreamReader & reader,
                                const std::string &  gguf_path) = 0;

    // Per managed MoE dispatch.  ``src0`` names the canonical
    // expert-stack tensor (e.g. ``"blk.0.ffn_gate_exps.weight"``);
    // ``src1`` is the routed input; ``ids`` is [n_tokens × n_used]
    // expert ids; ``dst`` is the output.  All work goes on
    // ``stream``.  Returns true when the executor handled this op;
    // false to fall through to stock dispatch (unmanaged tensor,
    // shape mismatch, etc.).
    //
    // The per-op-shaped ``forward_moe_block`` virtual was retired in
    // S8 along with the legacy MUL_MAT_ID dispatch surface. Managed
    // MoE now flows exclusively through ``forward_moe_layer`` below.

    // Per-layer MoE dispatch (Mode A milestone 1, criterion 3).
    // Execution-time entry point — invoked once per managed MoE
    // layer from the arch-builder's sentinel dispatch (S5+S6+S7).
    // The router/topk computation has already produced concrete
    // ids/probs/weights tensors at the call site; this entry point
    // owns scheduling, load, wait, validate, fused kernel launch,
    // weighted reduce, and release.
    //
    //   layer_in     [n_tokens, hidden_dim]  — post-ffn-norm input
    //   ids          [n_expert_used, n_tokens]  int32  expert ids
    //   probs        [n_expert, n_tokens]       f32    full router probs
    //   weights      [n_expert_used, n_tokens]  f32    renormalised topk weights
    //   layer_out    [n_tokens, hidden_dim]  — F32; the executor's
    //                weighted reduce writes the final block output
    //                consumed by the post-MoE residual.
    //   layer_idx    arch-relative layer index (used for per-layer
    //                MoEMatMulComp lookup).
    //
    // Returns true when the executor handled this layer; false to
    // surface a dispatch error (the call site is expected to fail
    // hard rather than fall back — see criterion 9). For S2 this is
    // a pure virtual with a stub override; the real body lands in
    // S6 along with the call site in S5.
    virtual bool forward_moe_layer(StreamHandle               stream,
                                    const ggml_tensor *        layer_in,
                                    const ggml_tensor *        ids,
                                    const ggml_tensor *        probs,
                                    const ggml_tensor *        weights,
                                    ggml_tensor *              layer_out,
                                    int                        layer_idx) = 0;

    // Optional model-tensor binding (Mode A milestone 1, criterion 5).
    // Future-proof plumbing — M1 correctness does NOT depend on it
    // being exercised. The arch builder in S5 owns router/topk
    // construction via the shared helper factored in S3; the
    // executor receives concrete ids/probs/weights and never needs
    // direct access to ffn_gate_inp for M1.
    //
    // ``ffn_gate_inp_per_layer`` is an array of ``n_layer``
    // ``ggml_tensor *`` pointers, indexed by layer.  Lifetime of
    // the array itself is one call; the executor caches what it
    // wants. Lifetime of each tensor pointer is the model's
    // lifetime — they remain valid until ``llama_model_free``.
    //
    // Default no-op: most executors won't need router gates. Qwen3
    // overrides to cache the pointers for a post-M1 move that
    // relocates router/topk construction into the executor at
    // execute time.
    virtual void attach_router_gates(
        const struct ggml_tensor * const * /*ffn_gate_inp_per_layer*/,
        int /*n_layer*/) {}
};

// Registry.  Executors register their factory at process init via
// a static initialiser; ``install_for_gguf`` looks one up by name
// (the GGUF's ``streamllm.executor`` key, or a default for legacy
// streams without the key).  Returns nullptr for unknown names —
// the loader contract is to refuse-to-load on a hard mismatch
// (SSOT §6.1.2).
using ExecutorFactory = std::unique_ptr<ModelExecutor> (*)();

void register_executor(const char * name, ExecutorFactory factory);
std::unique_ptr<ModelExecutor> make_executor(const char * name);

}  // namespace streamllm_ext
