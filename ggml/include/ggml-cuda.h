#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#ifdef GGML_USE_HIP
#define GGML_CUDA_NAME "ROCm"
#define GGML_CUBLAS_NAME "hipBLAS"
#elif defined(GGML_USE_MUSA)
#define GGML_CUDA_NAME "MUSA"
#define GGML_CUBLAS_NAME "muBLAS"
#else
#define GGML_CUDA_NAME "CUDA"
#define GGML_CUBLAS_NAME "cuBLAS"
#endif
#define GGML_CUDA_MAX_DEVICES       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_cuda_init(int device);

GGML_BACKEND_API bool ggml_backend_is_cuda(ggml_backend_t backend);

// device buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);

// conduct allreduce operation between devices
GGML_BACKEND_API bool ggml_backend_cuda_allreduce_tensor(ggml_backend_t * backends, struct ggml_tensor ** tensors, size_t n_backends);

// split tensor buffer that splits matrices by rows across multiple devices
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_split_buffer_type(int main_device, const float * tensor_split);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);

GGML_BACKEND_API int  ggml_backend_cuda_get_device_count(void);
GGML_BACKEND_API void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);

GGML_BACKEND_API bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size);
GGML_BACKEND_API void ggml_backend_cuda_unregister_host_buffer(void * buffer);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cuda_reg(void);

// streamllm-ext integration: GGML_OP_MUL_MAT override. Mode A M1
// uses this *only* as the criterion-10 clear-fail point: if the hook
// detects a managed canonical surfacing as a dense MUL_MAT (which
// should never happen in M1), it GGML_ABORTs loudly. There is no
// dense-managed streaming in M1; future DenseExecutor work would
// replace this entry. Returning true would signal "op handled" (and
// short-circuit stock dispatch); the M1 body never returns true,
// only aborts or returns false.
//
// Declared as a raw ``void *`` to keep this header CUDA-runtime-clean;
// the implementation casts to the concrete function-pointer type.
GGML_BACKEND_API void ggml_cuda_set_mul_mat_hook(void * hook_fn);

// streamllm-ext integration: graph-walk pre/post hooks. Mode A M1
// scope: **audit + score-snapshot only**. The pre-hook fires at the
// top of ggml_backend_cuda_graph_compute and is expected to do at
// most (a) a one-shot cgraph audit (sentinels well-formed, no
// managed MUL_MAT_ID/MUL_MAT leaked) and (b) a snapshot of the
// score-table for the score-dial side channel. It must NOT plan
// routing, reserve chunks, submit loads, dispatch managed nodes, or
// use prior-token routing — all scheduling lives at the per-sentinel
// dispatch site in pre_op_hook. The post-hook does end-of-replay
// cleanup (replay-reservation drop).
//
// Callback signatures:
//   void begin(cudaStream_t, const struct ggml_cgraph *);
//   void end  (cudaStream_t, const struct ggml_cgraph *);
// Either may be null. The cgraph pointer is read-only — modifying
// nodes from inside these hooks is unsupported.
GGML_BACKEND_API void ggml_cuda_set_graph_compute_begin_hook(void * hook_fn);
GGML_BACKEND_API void ggml_cuda_set_graph_compute_end_hook  (void * hook_fn);

// streamllm-ext integration: user-managed-node claim hook. When set, ggml-
// cuda asks this predicate per cgraph node before deciding whether to
// capture the cgraph into a cuda-graph. If the predicate returns true for
// ANY node in the cgraph, capture is disabled for this compute call and
// every node runs eager via its regular dispatch path (where the
// streamllm per-op hooks above can claim it). Capture stays enabled for
// cgraphs with no claimed nodes (the common dense-only case).
//
// Rationale: managed mul_mat_id ops need to run their LOAD walk on every
// invocation, which is incompatible with whole-cgraph cuda-graph capture
// (the captured replay would skip the host-side decision-making). Partial
// per-segment capture is the right long-term answer (see plan
// vigilant-stitching-heron P1+); disabling capture entirely for the
// affected cgraph is the correct interim behaviour that matches what
// llama.cpp does today for MoE models.
//
// Callback signature:
//   bool fn(const struct ggml_tensor * node);
// Set to null to unregister. Independent of the other streamllm hooks.
GGML_BACKEND_API void ggml_cuda_set_user_node_claims_hook(void * hook_fn);

// streamllm-ext integration: score-table version hook. When set,
// ggml-cuda's per-cgraph "update required?" predicate calls this and
// compares the returned uint64_t to a per-graph cached version. A
// version mismatch forces re-capture of the cgraph — used by
// streamllm-ext to invalidate captured graphs whenever the runtime
// score-dial swaps (HTTP /streamllm/score_table or a phase-aware
// reasoning→generation transition). The version is bumped each time
// streamllm_set_score_table() succeeds.
//
// Callback signature:
//   uint64_t fn(void);
// Set to null to unregister.
GGML_BACKEND_API void ggml_cuda_set_streamllm_score_version_hook(void * hook_fn);

// streamllm-ext integration: generic pre-op claim hook. Fires at the
// top of ggml_cuda_compute_forward for EVERY op — the hook inspects
// dst (op type, name, src[i]) and returns true to indicate "I handled
// this node; skip the default dispatch." Used by Mode A milestone-1's
// sentinel-dispatch mechanism (a named GGML_OP_DUP node carries
// pre-built routing tensors in src[1..3]; the streamllm executor
// claims it via this hook and runs the managed MoE block as a host-
// eager call).
//
// Pre-op semantics are critical: the hook MUST be consulted BEFORE
// the default kernel executes. ggml-cuda's existing per-op hooks
// (mul_mat / mul_mat_id) live inside their respective ops' dispatch
// functions and only protect those single ops. The pre-op hook is
// generic — it sees every node and decides per-node.
//
// Callback signature:
//   bool fn(cudaStream_t stream, struct ggml_tensor * dst);
// Set to null to unregister. Independent of the other streamllm hooks.
GGML_BACKEND_API void ggml_cuda_set_pre_op_hook(void * hook_fn);

#ifdef  __cplusplus
}
#endif
