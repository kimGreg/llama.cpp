// streamllm-ext / decoder / anybcq — AnyBCQFamilyTensor impl.
//
// All virtuals delegate to the ``UpstreamLayoutHost`` callbacks the
// upstream-layout parser registered (see decoder/anybcq/upstream_layout.cpp
// :: build_upstream_layout_anyprec and decoder/shortcut_anybcq/upstream_layout
// .cpp :: build_upstream_layout_shortcut).  This keeps the encoder-specific
// byte-format knowledge in the parser layer and makes the tensor a thin
// ABC adapter.

#include "tensor.h"

#include <stdexcept>
#include <utility>

namespace streamllm_ext { namespace anybcq {

AnyBCQFamilyTensor::AnyBCQFamilyTensor(std::string wid,
                                        UpstreamLayoutHost layout)
    : wid_(std::move(wid)), host_(std::move(layout)) {}

size_t AnyBCQFamilyTensor::disk_bytes(int chunk_idx) const {
    // Any-prec stores per-chunk disk sizes (chunk 0 carries base_p
    // planes; later chunks carry one).  Shortcut has a uniform
    // disk_bytes_per_chunk.
    if (host_.any_precision) {
        if (chunk_idx < 0 || chunk_idx >= (int)host_.chunk_planes.size()) {
            throw std::out_of_range(
                "AnyBCQFamilyTensor::disk_bytes: chunk index out of range");
        }
        return host_.chunk_planes[chunk_idx].disk_chunk_bytes;
    }
    return host_.disk_bytes_per_chunk;
}

size_t AnyBCQFamilyTensor::kernel_bytes(int chunk_idx) const {
    if (host_.any_precision) {
        if (chunk_idx < 0 || chunk_idx >= (int)host_.chunk_planes.size()) {
            throw std::out_of_range(
                "AnyBCQFamilyTensor::kernel_bytes: chunk index out of range");
        }
        return host_.chunk_planes[chunk_idx].kernel_chunk_bytes;
    }
    return host_.bytes_per_chunk;
}

void AnyBCQFamilyTensor::disk_to_kernel(int chunk_idx,
                                         const uint8_t * disk_in,
                                         uint8_t       * kernel_out) const {
    if (host_.disk_to_kernel_fn == nullptr) {
        throw std::runtime_error(
            "AnyBCQFamilyTensor::disk_to_kernel: encoder did not register "
            "disk_to_kernel_fn for " + wid_);
    }
    host_.disk_to_kernel_fn(host_, chunk_idx, disk_in, kernel_out);
}

void AnyBCQFamilyTensor::after_load(int chunk_idx,
                                     const void * device_ptr,
                                     StreamHandle stream) {
    if (host_.after_load_fn == nullptr) return;
    host_.after_load_fn(host_, chunk_idx, device_ptr,
                        d_qw_ptrs_, d_alpha_ptrs_, d_qbias_slot_,
                        stream);
}

void AnyBCQFamilyTensor::after_evict(int chunk_idx) {
    if (host_.after_evict_fn == nullptr) return;
    // The encoder maps chunk_idx → plane_idx differently per family:
    //   - shortcut: chunk_idx == plane_idx
    //   - any-prec: plane_idx_first stored on chunk_planes[chunk_idx]
    int plane_idx = chunk_idx;
    if (host_.any_precision &&
        chunk_idx >= 0 &&
        chunk_idx < (int)host_.chunk_planes.size()) {
        plane_idx = host_.chunk_planes[chunk_idx].plane_idx_first;
    }
    host_.after_evict_fn(d_qw_ptrs_, d_alpha_ptrs_, plane_idx);
}

void AnyBCQFamilyTensor::set_device_state(void ** d_qw_ptrs,
                                           void ** d_alpha_ptrs,
                                           void ** d_qbias_slot) {
    d_qw_ptrs_    = d_qw_ptrs;
    d_alpha_ptrs_ = d_alpha_ptrs;
    d_qbias_slot_ = d_qbias_slot;
}

}}  // namespace streamllm_ext::anybcq
