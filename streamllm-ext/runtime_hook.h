// streamllm-ext — glue between llama.cpp's model load + ggml-cuda's
// mul_mat dispatch and our StreamllmRuntime.
//
// Lifecycle (driven from llama_model_load_from_file_impl):
//
//   streamllm_runtime_install_for_gguf(gguf_path)
//     → parses the GGUF once more (no_alloc) to read streamllm.*
//       metadata, builds a StreamllmRuntime, pre-loads all managed
//       tensors into a VRAM pool, and registers our mul_mat hook with
//       ggml-cuda. If the file has no streamllm.* block, this is a
//       no-op — stock GGUFs keep their default behaviour.
//
//   streamllm_runtime_clear()
//     → called at model free; unregisters the hook and tears down the
//       runtime + its VRAM allocations.
//
// ``streamllm_try_cuda_mul_mat`` is the hook itself — registered with
// ggml-cuda via ``ggml_cuda_set_mul_mat_hook``. It inspects ``src0``'s
// name, looks it up in the active runtime, and if matched, dispatches
// to ``naver_gemv_launch``. Anything it doesn't claim returns false and
// falls through to the default CUDA mul_mat.

#pragma once

#include <cuda_runtime.h>

struct ggml_tensor;

namespace streamllm_ext {

// Build a global StreamllmRuntime from ``gguf_path`` if the file carries
// streamllm.* metadata, and register the ggml-cuda hook. Idempotent on
// stock GGUFs (silently does nothing). Aborts on partial streamllm
// files so the caller sees the real error during model load.
//
// Returns true if a runtime was installed, false if the file is stock.
bool install_for_gguf(const char * gguf_path);

// Tear down whatever ``install_for_gguf`` set up. Safe to call on a
// runtime that was never installed.
void clear();

// The hook itself. Registered with ``ggml_cuda_set_mul_mat_hook``;
// not intended to be called directly by anything but ggml-cuda.
extern "C" bool streamllm_try_cuda_mul_mat(
    cudaStream_t stream,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    struct ggml_tensor * dst);

// Fusion-skip predicate: ggml-cuda calls this on the weight tensor
// of a mul_mat that is a candidate for fusion (ffn_up + ffn_gate +
// glu, mul_mat_vec + glu, etc.). When it returns true the fusion is
// disabled, routing the mul_mat to the regular dispatch path so the
// mul_mat hook above can claim it. No-op when the runtime has no
// managed tensor with that name.
extern "C" bool streamllm_claims_tensor(const struct ggml_tensor * w);

} // namespace streamllm_ext
