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

// streamllm-ext integration: register an override for GGML_OP_MUL_MAT.
// The hook is called before the default dispatch inside ggml_cuda_mul_mat;
// returning true signals "op handled", false falls through. Stream is the
// current CUDA compute stream; src0/src1/dst are the op's operands (src0
// is the weight, src1 the activations). Set to null to unregister.
//
// Declared as a raw ``void *`` to keep this header CUDA-runtime-clean;
// the implementation casts to the concrete function-pointer type.
GGML_BACKEND_API void ggml_cuda_set_mul_mat_hook(void * hook_fn);

// streamllm-ext integration: fusion-skip query. When set, ggml-cuda calls
// this before fusing a mul_mat into a larger subgraph (ffn_up + ffn_gate
// + glu, mul_mat_vec + glu, etc.). If it returns true for the candidate
// weight tensor, fusion is disabled and the mul_mat is dispatched via the
// normal path where the streamllm hook can claim it. Callback signature:
//   bool fn(const struct ggml_tensor * weight_tensor);
// Set to null to unregister. Independent of the mul_mat hook.
GGML_BACKEND_API void ggml_cuda_set_fusion_skip_hook(void * hook_fn);

// streamllm-ext integration: parallel hook for GGML_OP_MUL_MAT_ID
// (MoE expert dispatch). Called at the top of ggml_cuda_mul_mat_id;
// returning true means the hook handled the op, false falls through.
// Stream is the current CUDA compute stream; src0 is the stacked expert
// weight tensor [K, M, n_experts]; src1 is the routed activations;
// ids is the int32 expert-id tensor selected by the router. Set to
// null to unregister. Independent of the dense mul_mat hook.
GGML_BACKEND_API void ggml_cuda_set_mul_mat_id_hook(void * hook_fn);

// streamllm-ext integration: notification hook fired just before
// ggml_cuda_op_topk_moe (the fused softmax+argsort+(optional)norm
// kernel that some MoE configurations dispatch instead of the
// separate ops). Lets the streamllm hook capture the (logits,
// weights, ids) tensor triple so it can later recover the
// renormalized routing weights at mul_mat_id dispatch time —
// without that side-channel, the buffer reachable via
// ids->src[0]->src[0] is stale (the fused kernel never writes a
// softmax output, so reading it returns whatever last lived in
// that buffer). Callback signature:
//   void fn(cudaStream_t, const ggml_tensor * logits,
//           ggml_tensor * weights, ggml_tensor * ids);
// Set to null to unregister.
GGML_BACKEND_API void ggml_cuda_set_topk_moe_hook(void * hook_fn);

// streamllm-ext integration: graph-walk pre/post hooks. Fire at the
// top and bottom of ggml_backend_cuda_graph_compute, giving the
// streamllm scheduler a chance to walk the cgraph (prefetch managed
// chunks ahead of time, mark per-graph state, etc.). Callback
// signatures:
//   void begin(cudaStream_t, const struct ggml_cgraph *);
//   void end  (cudaStream_t, const struct ggml_cgraph *);
// Either may be null. The cgraph pointer is read-only — modifying
// nodes from inside these hooks is unsupported (the cgraph is the
// scheduler's snapshot of work-to-do).
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

#ifdef  __cplusplus
}
#endif
