// streamllm-ext / decoder / direct_matrix — one native GGML expert chunk.

#pragma once

#include "chunked_tensor.h"
#include "upstream_layout.h"

#include <string>

namespace streamllm_ext { namespace direct_matrix {

class DirectMatrixTensor : public ChunkedTensor {
public:
    DirectMatrixTensor(std::string wid, UpstreamLayoutHost layout);

    const std::string & id() const override { return wid_; }
    int n_chunks() const override { return host_.n_chunks; }
    UpstreamLayoutHost & host() override { return host_; }
    const UpstreamLayoutHost & host() const override { return host_; }

    size_t disk_bytes(int chunk_idx) const override;
    size_t kernel_bytes(int chunk_idx) const override;
    void disk_to_kernel(int chunk_idx,
                        const uint8_t * disk_in,
                        uint8_t * kernel_out) const override;
    void after_load(int chunk_idx, const void * device_ptr,
                    StreamHandle stream) override;
    void after_evict(int chunk_idx, StreamHandle stream) override;
    void set_device_state(void ** d_qw_ptrs,
                          void ** d_alpha_ptrs,
                          void ** d_qbias_slot) override;

    void ** d_weight_slot() const { return d_weight_slot_; }

private:
    std::string wid_;
    UpstreamLayoutHost host_;
    void ** d_weight_slot_ = nullptr;
};

UpstreamLayoutHost build_upstream_layout_direct_matrix(
    const TensorLayout & layout,
    const uint8_t * tensor_data);

UpstreamLayoutHost build_upstream_layout_direct_expert_block(
    const TensorLayout & layout,
    const uint8_t * tensor_data);

}}  // namespace streamllm_ext::direct_matrix
