// streamllm-ext — cuBLAS F16 GEMM used for prefill. See batched_gemm.h.

#include "batched_gemm.h"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <mutex>

namespace streamllm_ext {

namespace {

cublasHandle_t g_handle = nullptr;
std::mutex     g_mu;

const char * cublas_err_str(cublasStatus_t s) {
    switch (s) {
        case CUBLAS_STATUS_SUCCESS:          return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:  return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:     return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:    return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:    return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:    return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:   return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED:    return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR:    return "CUBLAS_STATUS_LICENSE_ERROR";
        default:                             return "CUBLAS_STATUS_UNKNOWN";
    }
}

} // anonymous


bool batched_gemm_init() {
    std::lock_guard<std::mutex> lk(g_mu);
    if (g_handle) return true;
    auto st = cublasCreate(&g_handle);
    if (st != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "batched_gemm: cublasCreate failed: %s\n",
                     cublas_err_str(st));
        g_handle = nullptr;
        return false;
    }
    // Tensor-core math path — matters a lot on Ada.
    cublasSetMathMode(g_handle, CUBLAS_DEFAULT_MATH);
    return true;
}

void batched_gemm_shutdown() {
    std::lock_guard<std::mutex> lk(g_mu);
    if (g_handle) cublasDestroy(g_handle);
    g_handle = nullptr;
}

bool batched_gemm_f16(
    const void * W_f16, const void * X_f16, void * Y_f16,
    int M, int K, int N,
    cudaStream_t stream)
{
    if (g_handle == nullptr) {
        if (!batched_gemm_init()) return false;
    }

    // W is row-major [M, K] → viewed as col-major [K, M] (lda = K).
    // op_A = T gives an (M, K) operand in cuBLAS's view.
    // X is col-major [K, N] (lda = K). op_B = N leaves it as (K, N).
    // Y is col-major [M, N] (lda = M). Result shape (M, N).
    const __half alpha_h = __float2half(1.0f);
    const __half beta_h  = __float2half(0.0f);

    // cublasGemmEx is fine with F16 alpha/beta when Atype/Btype are F16.
    cublasSetStream(g_handle, stream);
    auto st = cublasGemmEx(
        g_handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        M, N, K,
        &alpha_h,
        W_f16, CUDA_R_16F, K,   // A: [K, M] col-major, op=T → (M, K)
        X_f16, CUDA_R_16F, K,   // B: [K, N] col-major
        &beta_h,
        Y_f16, CUDA_R_16F, M,   // C: [M, N] col-major
        CUBLAS_COMPUTE_16F,
        CUBLAS_GEMM_DEFAULT);
    if (st != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "batched_gemm_f16: cublasGemmEx failed: %s "
                             "(M=%d N=%d K=%d)\n",
                     cublas_err_str(st), M, N, K);
        return false;
    }
    return true;
}

} // namespace streamllm_ext
