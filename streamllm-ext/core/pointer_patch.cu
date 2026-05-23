#include "pointer_patch.h"

#include <cuda_runtime.h>

#include <mutex>
#include <stdexcept>
#include <string>

namespace streamllm_ext {
namespace {

inline void check_cuda(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        throw std::runtime_error(
            std::string("pointer_patch: CUDA error in ") + what + ": " +
            cudaGetErrorString(e));
    }
}

__global__ void k_apply_pointer_patches(const PointerPatch * patches,
                                        int n_patches) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_patches) return;

    const PointerPatch p = patches[i];
    if (p.d_qw == nullptr || p.d_alpha == nullptr) return;

    if (p.kind == PointerPatchKind::ClearPlanes) {
        for (int j = 0; j < p.n_planes; ++j) {
            p.d_qw[p.plane_first + j]    = nullptr;
            p.d_alpha[p.plane_first + j] = nullptr;
        }
        return;
    }

    if (p.chunk_base == nullptr) return;
    uint8_t * base = static_cast<uint8_t *>(p.chunk_base);

    if (p.kind == PointerPatchKind::SetSsAnybcq) {
        p.d_qw[p.plane_first]    = base;
        p.d_alpha[p.plane_first] = base + p.qw_bytes;
        return;
    }

    if (p.kind == PointerPatchKind::SetAnyPrec) {
        for (int j = 0; j < p.n_planes; ++j) {
            p.d_qw[p.plane_first + j] = base + (size_t)j * p.qw_bytes;
        }
        uint8_t * alpha_base = base + (size_t)p.n_planes * p.qw_bytes;
        for (int j = 0; j < p.precision; ++j) {
            p.d_alpha[j] = alpha_base + (size_t)j * p.alpha_bytes;
        }
        if (p.d_qbias_slot != nullptr) {
            p.d_qbias_slot[0] =
                alpha_base + (size_t)p.precision * p.alpha_bytes;
        }
    }
}

std::mutex g_patch_mu;
PointerPatch * g_d_patches = nullptr;
size_t g_patch_capacity = 0;

PointerPatch * ensure_patch_buffer(size_t n) {
    if (n <= g_patch_capacity) return g_d_patches;
    PointerPatch * fresh = nullptr;
    check_cuda(cudaMalloc(&fresh, n * sizeof(PointerPatch)),
               "cudaMalloc(patch buffer)");
    if (g_d_patches != nullptr) {
        cudaFree(g_d_patches);
    }
    g_d_patches = fresh;
    g_patch_capacity = n;
    return g_d_patches;
}

}  // namespace

void apply_pointer_patches_async(const std::vector<PointerPatch> & patches,
                                 StreamHandle stream) {
    if (patches.empty()) return;
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    std::lock_guard<std::mutex> lk(g_patch_mu);
    PointerPatch * d_patches = ensure_patch_buffer(patches.size());
    check_cuda(cudaMemcpyAsync(d_patches, patches.data(),
                               patches.size() * sizeof(PointerPatch),
                               cudaMemcpyHostToDevice, s),
               "cudaMemcpyAsync(patches)");
    const int block = 128;
    const int grid = (int)((patches.size() + block - 1) / block);
    k_apply_pointer_patches<<<grid, block, 0, s>>>(
        d_patches, (int)patches.size());
    check_cuda(cudaGetLastError(), "k_apply_pointer_patches");
}

}  // namespace streamllm_ext
