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

#include <memory>
#include <mutex>
#include <vector>

struct ggml_tensor;
struct ggml_cgraph;

namespace streamllm_ext {

class StreamllmRuntime;

// Internal — owned by qwen3/qwen3_runtime_glue.cpp (install_for_gguf / clear).
// Read by qwen3/qwen3_moe_dispatch.cpp under g_runtime_mu so the active
// runtime is reachable from the scheduler-side dispatch bodies. Not part
// of the extension's public API.
extern std::mutex                            g_runtime_mu;
extern std::unique_ptr<StreamllmRuntime>     g_runtime;

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

// MoE dispatch hook. Registered with ``ggml_cuda_set_mul_mat_id_hook``;
// fires at the top of ggml-cuda's MoE op handler. ``src0`` is the
// stacked expert weight tensor [K, M, n_experts]; ``src1`` is the
// routed activations; ``ids`` is the int32 expert-id tensor selected
// by the router. Returns true when the hook computes the op (Strategy
// A: loop chunk_matmul over (token, expert)); false to fall through to
// upstream's batched MoE GEMM. No-op when src0 isn't a managed
// expert-stack tensor.
extern "C" bool streamllm_try_cuda_mul_mat_id(
    cudaStream_t stream,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * ids,
    struct ggml_tensor * dst);

// Live-tunable precision dial. ``thresholds`` (length n_thresh,
// descending) and ``chunks`` (length n_chunks, same length) replace the
// active scheduler's score-threshold table.  ``thresholds`` is a
// length-N ascending vector where N = max n_chunks across managed
// tensors (the model's "full chunk size").  thresholds[k] is the
// lower-edge gate score for the band that loads (k+1) chunks; the
// chunks count is implicit by index.  Returns true on success,
// false if no runtime is installed or the input is malformed
// (length mismatch, non-ascending, out-of-range value).  Safe to
// call from any thread; the next mul_mat_id dispatch sees the new
// table; in-flight dispatches keep using the snapshot they took
// at entry.
extern "C" bool streamllm_set_score_table(
    const float * thresholds, int n_thresh);

// Snapshot of the active score-threshold table.  Returns false if
// no runtime is installed.  Safe from any thread.
bool streamllm_get_score_table(
    std::vector<float> & out_thresholds);

// Notification before ggml-cuda's fused topk_moe kernel. We stash
// the (ids, weights) pair so streamllm_try_cuda_mul_mat_id can later
// look up the renormalized routing weights (sum=1 over K) by ids
// pointer — without that, the per-(t, u) gate-score path falls back
// to whatever's in the selection-probs buffer, which the fused
// kernel never wrote (in-register softmax).
extern "C" void streamllm_topk_moe_observed(
    cudaStream_t stream,
    const struct ggml_tensor * logits,
    struct ggml_tensor * weights,
    struct ggml_tensor * ids);

// Graph-walk pre/post hooks. Registered with
// ``ggml_cuda_set_graph_compute_{begin,end}_hook``; fire at the top
// and bottom of ggml_backend_cuda_graph_compute. Forward to the
// active scheduler's on_graph_compute_begin / on_graph_compute_end
// virtuals so concrete schedulers can prewalk the cgraph (managed-
// tensor identification, prefetch, marker scan) before any node-
// level dispatch starts.
extern "C" void streamllm_graph_compute_begin(
    cudaStream_t                stream,
    const struct ggml_cgraph *  cgraph);
extern "C" void streamllm_graph_compute_end(
    cudaStream_t                stream,
    const struct ggml_cgraph *  cgraph);

} // namespace streamllm_ext
