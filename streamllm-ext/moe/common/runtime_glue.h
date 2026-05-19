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

#include "qwen3_executor.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

struct ggml_tensor;
struct ggml_cgraph;

namespace streamllm_ext {

class StreamllmRuntime;

// Internal — owned by qwen3/runtime_glue.cpp (install_for_gguf / clear).
// Read by qwen3/dispatch.cpp and qwen3/executor.cpp
// under g_runtime_mu so the active runtime + executor are reachable from
// the scheduler-side dispatch bodies and the executor shim. Not part of
// the extension's public API.
extern std::mutex                            g_runtime_mu;
extern std::unique_ptr<StreamllmRuntime>     g_runtime;
extern std::unique_ptr<ModelExecutor>        g_executor;

// Mode A milestone 1, S1 — model-scoped executor binding.
//
// streamllm-ext does NOT depend on the llama_model layout. The
// model-scoped binding API is type-erased over a ``void **`` slot
// address — the caller (llama.cpp loader / llama_model_free) hands
// in the address of the model's ``streamllm_executor`` field; the ext
// reads/writes nullptr through that address but never dereferences
// the model object beyond it. This keeps streamllm-ext self-
// contained (no llama-model.h dependency in the lib).
//
// ``current_executor()`` returns the active ``ModelExecutor *`` (or
// nullptr if no streamllm runtime is installed). Raw pointer; ownership
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
void            bind_model_slot(void ** streamllm_executor_slot);
void            unbind_model_slot(void ** streamllm_executor_slot);

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

// streamllm_claims_tensor extern-C and streamllm_try_cuda_mul_mat_id
// were retired together with the legacy fusion_skip / mul_mat_id hook
// installs. Managed MoE dispatch flows through ``streamllm_pre_op``
// (sentinel rail) only; the internal ``Scheduler::claims_tensor``
// predicate survives as the S10 dense-managed clear-fail's
// is-this-tensor-managed check.

// Mode A milestone-1 generic pre-op claim hook registered via
// ggml_cuda_set_pre_op_hook. Fires before every op's default
// dispatch. Claims nodes whose name starts with
// ``"streamllm.moe_layer_"`` and routes them to
// Qwen3MoEAnyBcqExecutor::forward_moe_layer. Returning true short-
// circuits ggml_cuda_compute_forward.
extern "C" bool streamllm_pre_op(
    cudaStream_t stream,
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

// S8 (Mode A): ``streamllm_topk_moe_observed`` was retired with the
// fused-topk_moe hook install. Sentinel ``src[2..3]`` carries probs
// and renormalised weights directly.

// Mode A M1 graph-compute callbacks. Registered with
// ``ggml_cuda_set_graph_compute_{begin,end}_hook``. Strictly audit +
// score-snapshot (begin) and replay-reservation cleanup (end). Must
// not plan routing, reserve chunks, submit loads, dispatch managed
// nodes, use prior-token routing, or select fallback paths — all
// scheduling lives at the per-sentinel dispatch site.
extern "C" void streamllm_on_graph_audit_and_score_snapshot(
    cudaStream_t                stream,
    const struct ggml_cgraph *  cgraph);
extern "C" void streamllm_on_graph_audit_and_score_snapshot_end(
    cudaStream_t                stream,
    const struct ggml_cgraph *  cgraph);

// User-managed-node claim predicate. Registered with
// ``ggml_cuda_set_user_node_claims_hook``; ggml-cuda calls this per
// cgraph node to decide whether to disable cuda-graph capture for
// that cgraph compute. Returns true for streamllm sentinel nodes
// (``"streamllm.moe_layer_<N>"``) so capture stays off for any
// cgraph that carries managed MoE.
extern "C" bool streamllm_user_node_claims(
    const struct ggml_tensor * node);

// Score-table version key. Registered with ggml-cuda's
// ``ggml_cuda_set_streamllm_score_version_hook``; the cgraph cache
// reads it once per compute and forces re-capture on any change.
extern "C" uint64_t streamllm_replay_score_table_version(void);

// ── Runtime-mutable gradual-schedule state ────────────────────────────
//
// The common-sampler observer in common/sampling.cpp reads this on
// each common_sampler_init when the runtime API has been populated;
// otherwise it falls back to the STREAMLLM_SCHEDULE_* env vars. The
// HTTP route POST /streamllm/schedule writes via streamllm_schedule_set.
//
// Wire format (C-friendly):
//   thresholds : ascending int array of length n_thresh (generated-
//                token breakpoints).
//   dials_flat : packed float array, length = sum(dial_lens[i]).
//   dial_lens  : per-dial threshold-count array of length n_dials.
//   n_dials    = n_thresh + 1.
//
// Return false on shape/order violations.  Empty state (after
// streamllm_schedule_clear or before any set) is signalled by
// streamllm_schedule_active() == false.
extern "C" bool streamllm_schedule_set(
    const int *   thresholds, int n_thresh,
    const float * dials_flat, const int * dial_lens, int n_dials);
extern "C" void streamllm_schedule_clear(void);
extern "C" bool streamllm_schedule_active(void);

// Bulk fetch the active schedule (output reference args; pass nullptr
// to skip a field).  Always populates from the same atomic snapshot.
// Caller passes in/out vectors; returns false if no schedule is set.
bool streamllm_schedule_get(
    std::vector<int> *                       out_thresholds,
    std::vector<std::vector<float>> *        out_dials);

} // namespace streamllm_ext
