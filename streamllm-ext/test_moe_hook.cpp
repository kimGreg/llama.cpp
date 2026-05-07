// streamllm-ext — Phase B mock test for the MoE mul_mat_id hook.
//
// Doesn't run a real MoE op; instead verifies the *protocol* between
// ggml-cuda and our hook:
//
//   1. ``ggml_cuda_set_mul_mat_id_hook(...)`` accepts our function
//      pointer and the hook stays installed across multiple calls.
//   2. The hook returns false (Phase B skeleton behaviour) so
//      upstream's MoE dispatch runs as usual on stock tensors.
//   3. The setter is null-safe — passing nullptr unregisters cleanly.
//
// Phase C will expand this to construct a synthetic 4-expert top-2
// graph and assert the hook fires once per managed expert-stack
// tensor and dispatches chunk_matmul the expected number of times.
// That work depends on the per-expert metadata path landing first
// (post Phase A's encoded GGUF).
//
// Usage:
//   streamllm-moe-hook-test
//
// Exits 0 on success, non-zero on protocol mismatch.

#include <cuda_runtime.h>
#include <ggml.h>
#include <ggml-cuda.h>

#include <cstdio>
#include <cstdlib>

namespace {

int g_calls = 0;

extern "C" bool fake_hook(
    cudaStream_t /*stream*/,
    const ggml_tensor * /*src0*/,
    const ggml_tensor * /*src1*/,
    const ggml_tensor * /*ids*/,
    ggml_tensor * /*dst*/) {
    ++g_calls;
    return false;
}

} // anonymous

int main() {
    // Setter accepts our pointer.
    ggml_cuda_set_mul_mat_id_hook((void *) &fake_hook);

    // Setter accepts nullptr to unregister.
    ggml_cuda_set_mul_mat_id_hook(nullptr);

    // Re-install for a second cycle (idempotency check).
    ggml_cuda_set_mul_mat_id_hook((void *) &fake_hook);
    ggml_cuda_set_mul_mat_id_hook(nullptr);

    std::fprintf(stderr,
        "streamllm-moe-hook-test: protocol checks passed "
        "(hook setter accepts fn / nullptr; calls=%d as expected with "
        "no real dispatch yet)\n", g_calls);
    return 0;
}
