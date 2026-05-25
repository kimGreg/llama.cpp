// DPMoE — glue between llama.cpp's model load + ggml-cuda's
// mul_mat dispatch and our DPMoERuntime.
//
// Lifecycle (driven from llama_model_load_from_file_impl):
//
//   dp_moe_runtime_install_for_gguf(gguf_path)
//     → parses the GGUF once more (no_alloc) to read dp_moe.*
//       metadata, builds a DPMoERuntime, pre-loads all managed
//       tensors into a VRAM pool, and registers our mul_mat hook with
//       ggml-cuda. If the file has no dp_moe.* block, this is a
//       no-op — stock GGUFs keep their default behaviour.
//
//   dp_moe_runtime_clear()
//     → called at model free; unregisters the hook and tears down the
//       runtime + its VRAM allocations.
//
// ``dp_moe_try_cuda_mul_mat`` is the hook itself — registered with
// ggml-cuda via ``ggml_cuda_set_mul_mat_hook``. It inspects ``src0``'s
// name, looks it up in the active runtime, and if matched, dispatches
// to ``naver_gemv_launch``. Anything it doesn't claim returns false and
// falls through to the default CUDA mul_mat.

#pragma once

#include "moe_executor.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

struct ggml_tensor;
struct ggml_cgraph;

namespace dp_moe_ext {

class DPMoERuntime;

// Internal — owned by qwen3/runtime_glue.cpp (install_for_gguf / clear).
// Read by qwen3/dispatch.cpp and qwen3/executor.cpp
// under g_runtime_mu so the active runtime + executor are reachable from
// the scheduler-side dispatch bodies and the executor shim. Not part of
// the extension's public API.
extern std::mutex                            g_runtime_mu;
extern std::unique_ptr<DPMoERuntime>     g_runtime;
extern std::unique_ptr<ModelExecutor>        g_executor;

// Mode A milestone 1, S1 — model-scoped executor binding.
//
// DPMoE does NOT depend on the llama_model layout. The
// model-scoped binding API is type-erased over a ``void **`` slot
// address — the caller (llama.cpp loader / llama_model_free) hands
// in the address of the model's ``dp_moe_executor`` field; the ext
// reads/writes nullptr through that address but never dereferences
// the model object beyond it. This keeps DPMoE self-
// contained (no llama-model.h dependency in the lib).
//
// ``current_executor()`` returns the active ``ModelExecutor *`` (or
// nullptr if no dp_moe runtime is installed). Raw pointer; ownership
// stays with ``g_executor``. Lifetime is valid from
// ``install_for_gguf`` success until the next ``clear()``.
//
// ``bind_model_slot(slot)`` records the slot address in the bound-
// slots set so ``clear()`` can null it before tearing down
// ``g_executor``. Idempotent — calling twice with the same slot is a
// no-op.
//
// ``unbind_model_slot(slot)`` removes ``slot`` from the bound-slots
// set AND writes nullptr through it. MUST be called from
// ``llama_model_free`` (or anywhere the slot's owning model is about
// to be destroyed) so ``clear()`` never dereferences a freed slot.
// Idempotent.
//
// Lifetime contract: a slot that's been ``bind_model_slot``ed must
// remain a valid writable address until ``unbind_model_slot`` is
// called. ``clear()`` walks the bound-slots set under ``g_runtime_mu``
// and nulls each slot — it never calls into the model object beyond
// the write, so the worst case is a single store into still-live
// memory.
ModelExecutor * current_executor();
void            bind_model_slot(void ** dp_moe_executor_slot);
void            unbind_model_slot(void ** dp_moe_executor_slot);
bool            executor_uses_stock_moe_graph(void * dp_moe_executor);

// Build a global DPMoERuntime from ``gguf_path`` if the file carries
// dp_moe.* metadata, and register the ggml-cuda hook. Idempotent on
// stock GGUFs (silently does nothing). Aborts on partial dp_moe
// files so the caller sees the real error during model load.
//
// Returns true if a runtime was installed, false if the file is stock.
bool install_for_gguf(const char * gguf_path);

// Tear down whatever ``install_for_gguf`` set up. Safe to call on a
// runtime that was never installed.
void clear();

// The hook itself. Registered with ``ggml_cuda_set_mul_mat_hook``;
// not intended to be called directly by anything but ggml-cuda.
extern "C" bool dp_moe_try_cuda_mul_mat(
    cudaStream_t stream,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    struct ggml_tensor * dst);

// dp_moe_claims_tensor extern-C and dp_moe_try_cuda_mul_mat_id
// were retired together with the legacy fusion_skip / mul_mat_id hook
// installs. Managed MoE dispatch flows through ``dp_moe_pre_op``
// (sentinel rail) only; the internal ``Scheduler::claims_tensor``
// predicate survives as the S10 dense-managed clear-fail's
// is-this-tensor-managed check.

// Mode A milestone-1 generic pre-op claim hook registered via
// ggml_cuda_set_pre_op_hook. Fires before every op's default
// dispatch. Claims nodes whose name starts with
// ``"dp_moe.moe_layer_"`` and routes them to
// Qwen3MoEAnyBcqExecutor::forward_moe_layer. Returning true short-
// circuits ggml_cuda_compute_forward.
extern "C" bool dp_moe_pre_op(
    cudaStream_t stream,
    struct ggml_tensor * dst);

extern "C" bool dp_moe_set_kbar(float kbar, int allocator_mode);
extern "C" bool dp_moe_get_kbar(float * out_kbar, int * out_allocator_mode);
extern "C" bool dp_moe_set_prefill_decay_end(unsigned long long n_tokens, int reset_state);

// S8 (Mode A): ``dp_moe_topk_moe_observed`` was retired with the
// fused-topk_moe hook install. Sentinel ``src[2..3]`` carries probs
// and renormalised weights directly.

// Mode A M1 graph-compute callbacks. Registered with
// ``ggml_cuda_set_graph_compute_{begin,end}_hook``. Strictly audit +
// score-snapshot (begin) and replay-reservation cleanup (end). Must
// not plan routing, reserve chunks, submit loads, dispatch managed
// nodes, use prior-token routing, or select fallback paths — all
// scheduling lives at the per-sentinel dispatch site.
extern "C" void dp_moe_on_graph_audit_and_score_snapshot(
    cudaStream_t                stream,
    const struct ggml_cgraph *  cgraph);
extern "C" void dp_moe_on_graph_audit_and_score_snapshot_end(
    cudaStream_t                stream,
    const struct ggml_cgraph *  cgraph);

// User-managed-node claim predicate. Registered with
// ``ggml_cuda_set_user_node_claims_hook``; ggml-cuda calls this per
// cgraph node to decide whether to disable cuda-graph capture for
// that cgraph compute. Returns true for dp_moe sentinel nodes
// (``"dp_moe.moe_layer_<N>"``) so capture stays off for any
// cgraph that carries managed MoE.
extern "C" bool dp_moe_user_node_claims(
    const struct ggml_tensor * node);

extern "C" uint64_t dp_moe_replay_dial_version(void);

// ── Runtime-mutable gradual-schedule state ────────────────────────────
//
extern "C" bool dp_moe_kbar_schedule_set(
    const int * thresholds, int n_thresh,
    const float * kbars, int n_kbars,
    int allocator_mode);
extern "C" void dp_moe_schedule_clear(void);
extern "C" bool dp_moe_schedule_active(void);

// Bulk fetch the active schedule (output reference args; pass nullptr
// to skip a field).  Always populates from the same atomic snapshot.
// Caller passes in/out vectors; returns false if no schedule is set.
bool dp_moe_schedule_get(
    std::vector<int> *                       out_thresholds,
    std::vector<float> *                     out_kbars,
    int *                                    out_allocator_mode);

} // namespace dp_moe_ext
