// streamllm-ext / core — ModelExecutor ABC.
//
// Model-specific forward executor selected at llama_model_load by the
// GGUF's ``streamllm.executor`` key.  Owns the forward path for the
// model's streamed (chunked) blocks: scheduler invocation with
// current-batch routing, chunk reserve/load/wait, fused kernel
// launch, post-kernel release.  See SSOT §6.6.0.
//
// Today's invocation point: ggml-cuda's per-op hook for managed
// mul_mat_id nodes forwards into ``forward_moe_block``.  The
// executor is the canonical owner regardless of how it's reached;
// future ggml custom-op or arch-builder patches replace the hook
// without changing the executor surface.

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
    virtual bool forward_moe_block(StreamHandle               stream,
                                    const ggml_tensor *        src0,
                                    const ggml_tensor *        src1,
                                    const ggml_tensor *        ids,
                                    ggml_tensor *              dst) = 0;
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
