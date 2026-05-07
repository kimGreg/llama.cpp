// streamllm-ext / decoder / anybcq — per-plane device pointer layout impl.

#include "per_plane_layout.h"

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace streamllm_ext { namespace anybcq {

static inline void check_cuda(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        throw std::runtime_error(
            std::string("anybcq/per_plane_layout: CUDA error in ") + what +
            ": " + cudaGetErrorString(e));
    }
}

void alloc_per_plane_arrays(void **& d_qw, void **& d_alpha,
                            int max_planes) {
    if (max_planes <= 0) {
        throw std::runtime_error(
            "alloc_per_plane_arrays: max_planes must be > 0");
    }
    if (d_qw == nullptr) {
        check_cuda(cudaMalloc((void **)&d_qw,
                              (size_t)max_planes * sizeof(void *)),
                   "cudaMalloc(d_qw)");
        // Zero — the kernel's prefix-plan fast path distinguishes
        // null vs non-null per slot.
        check_cuda(cudaMemset(d_qw, 0,
                              (size_t)max_planes * sizeof(void *)),
                   "cudaMemset(d_qw)");
    }
    if (d_alpha == nullptr) {
        check_cuda(cudaMalloc((void **)&d_alpha,
                              (size_t)max_planes * sizeof(void *)),
                   "cudaMalloc(d_alpha)");
        check_cuda(cudaMemset(d_alpha, 0,
                              (size_t)max_planes * sizeof(void *)),
                   "cudaMemset(d_alpha)");
    }
}

void free_per_plane_arrays(void *& d_qw, void *& d_alpha) {
    if (d_qw)    { cudaFree(d_qw);    d_qw    = nullptr; }
    if (d_alpha) { cudaFree(d_alpha); d_alpha = nullptr; }
}

void clear_per_plane_after_evict(void ** d_qw, void ** d_alpha, int plane) {
    if (d_qw == nullptr || d_alpha == nullptr) return;
    if (plane < 0) return;
    void * null_ptr = nullptr;
    cudaMemcpy((uint8_t *)d_qw    + (size_t)plane * sizeof(void *),
               &null_ptr, sizeof(void *), cudaMemcpyHostToDevice);
    cudaMemcpy((uint8_t *)d_alpha + (size_t)plane * sizeof(void *),
               &null_ptr, sizeof(void *), cudaMemcpyHostToDevice);
}

void update_per_plane_after_load(void ** d_qw, void ** d_alpha,
                                 int plane,
                                 const void * chunk_device_ptr,
                                 size_t qw_bytes_per_chunk) {
    if (d_qw == nullptr || d_alpha == nullptr) return;
    if (plane < 0) return;
    const void * qw_p    = chunk_device_ptr;
    const void * alpha_p = static_cast<const uint8_t *>(chunk_device_ptr) +
                            qw_bytes_per_chunk;
    // Synchronous on default stream — infrequent, single 8-byte copy
    // each. Acts as a barrier ordering the per-plane ptr update after
    // the chunk's H2D and before any kernel that reads the arrays.
    cudaMemcpy((uint8_t *)d_qw    + (size_t)plane * sizeof(void *),
               &qw_p,    sizeof(void *), cudaMemcpyHostToDevice);
    cudaMemcpy((uint8_t *)d_alpha + (size_t)plane * sizeof(void *),
               &alpha_p, sizeof(void *), cudaMemcpyHostToDevice);
}

namespace {
// 1-thread kernel that stores both pointers in one launch — replaces
// the two ~13 µs synchronous cudaMemcpy calls with a single ~5 µs
// async kernel launch. The kernel itself takes a few cycles; the
// dominant cost is the launch, paid once per chunk move instead of
// twice per chunk move *plus* the full sync round-trip overhead.
__global__ void k_set_two_ptrs(void ** d_qw, void ** d_alpha,
                                int plane,
                                void * qw_p, void * alpha_p) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_qw[plane]    = qw_p;
        d_alpha[plane] = alpha_p;
    }
}
}  // anon

void update_per_plane_after_load_async(void ** d_qw, void ** d_alpha,
                                       int plane,
                                       const void * chunk_device_ptr,
                                       size_t qw_bytes_per_chunk,
                                       StreamHandle stream)
{
    if (d_qw == nullptr || d_alpha == nullptr) return;
    if (plane < 0) return;
    void * qw_p    = const_cast<void *>(chunk_device_ptr);
    void * alpha_p = const_cast<void *>(
        static_cast<const void *>(
            static_cast<const uint8_t *>(chunk_device_ptr) +
            qw_bytes_per_chunk));
    auto s = (cudaStream_t) stream;
    k_set_two_ptrs<<<1, 1, 0, s>>>(d_qw, d_alpha, plane, qw_p, alpha_p);
    // No cudaGetLastError check on the hot path — a launch failure
    // here is unrecoverable and would surface on the next sync.
}

}}  // namespace streamllm_ext::anybcq
