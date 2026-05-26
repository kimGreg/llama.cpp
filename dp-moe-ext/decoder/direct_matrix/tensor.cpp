// DPMoE / decoder / direct_matrix — one native GGML expert chunk.

#include "tensor.h"

#include <cuda_runtime.h>

#include <cstring>
#include <stdexcept>
#include <utility>

namespace dp_moe_ext { namespace direct_matrix {
namespace {

constexpr uint32_t kMagic = 0x54414d44u; // "DMAT"
constexpr uint8_t  kVersion = 1;
constexpr size_t   kHeaderSize = 32;
constexpr uint32_t kExpertMagic = 0x4b4c4245u; // "EBLK"
constexpr uint8_t  kExpertVersion = 1;
constexpr uint8_t  kExpertRecords = 3;
constexpr size_t   kExpertHeaderSize = 136;

#pragma pack(push, 1)
struct Header {
    uint32_t magic;
    uint8_t  version;
    uint8_t  ggml_type;
    uint8_t  n_dims;
    uint8_t  reserved0;
    uint32_t ne[4];
    uint64_t payload_nbytes;
};

struct ExpertRecord {
    uint32_t kind;
    uint32_t ggml_type;
    uint32_t ne[4];
    uint64_t offset;
    uint64_t nbytes;
};

struct ExpertHeader {
    uint32_t magic;
    uint8_t  version;
    uint8_t  n_records;
    uint16_t reserved0;
    uint32_t layer;
    uint32_t expert;
    ExpertRecord records[3];
};
#pragma pack(pop)
static_assert(sizeof(Header) == kHeaderSize, "direct_matrix header size");
static_assert(sizeof(ExpertRecord) == 40, "direct expert record size");
static_assert(sizeof(ExpertHeader) == kExpertHeaderSize, "direct expert header size");

void check_cuda(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("direct_matrix: CUDA error in ") +
                                 what + ": " + cudaGetErrorString(e));
    }
}

}  // namespace

DirectMatrixTensor::DirectMatrixTensor(std::string wid, UpstreamLayoutHost layout)
    : wid_(std::move(wid)), host_(std::move(layout)) {}

size_t DirectMatrixTensor::disk_bytes(int chunk_idx) const {
    if (chunk_idx != 0) throw std::out_of_range("direct_matrix: chunk index");
    return host_.disk_bytes_per_chunk;
}

size_t DirectMatrixTensor::kernel_bytes(int chunk_idx) const {
    if (chunk_idx != 0) throw std::out_of_range("direct_matrix: chunk index");
    return host_.bytes_per_chunk;
}

void DirectMatrixTensor::disk_to_kernel(
    int chunk_idx, const uint8_t * disk_in, uint8_t * kernel_out) const {
    if (chunk_idx != 0) throw std::out_of_range("direct_matrix: chunk index");
    std::memcpy(kernel_out, disk_in, host_.bytes_per_chunk);
}

void DirectMatrixTensor::after_load(
    int chunk_idx, const void * device_ptr, StreamHandle stream) {
    if (chunk_idx != 0 || d_weight_slot_ == nullptr) return;
    check_cuda(cudaMemcpyAsync(d_weight_slot_, &device_ptr, sizeof(void *),
                               cudaMemcpyHostToDevice,
                               (cudaStream_t) stream),
               "after_load slot update");
}

void DirectMatrixTensor::after_evict(int chunk_idx, StreamHandle stream) {
    if (chunk_idx != 0 || d_weight_slot_ == nullptr) return;
    const void * nullp = nullptr;
    check_cuda(cudaMemcpyAsync(d_weight_slot_, &nullp, sizeof(void *),
                               cudaMemcpyHostToDevice,
                               (cudaStream_t) stream),
               "after_evict slot clear");
}

void DirectMatrixTensor::set_device_state(
    void ** d_qw_ptrs, void **, void **) {
    d_weight_slot_ = d_qw_ptrs;
}

UpstreamLayoutHost build_upstream_layout_direct_matrix(
    const TensorLayout & layout,
    const uint8_t * tensor_data) {
    if (tensor_data == nullptr) {
        throw std::runtime_error("direct_matrix: null tensor_data");
    }
    if (layout.fixed_bytes != kHeaderSize || layout.chunk_bytes.size() != 1) {
        throw std::runtime_error("direct_matrix: expected one chunk and 32-byte header");
    }

    Header h{};
    std::memcpy(&h, tensor_data, sizeof(h));
    if (h.magic != kMagic) {
        throw std::runtime_error("direct_matrix: bad magic");
    }
    if (h.version != kVersion) {
        throw std::runtime_error("direct_matrix: unsupported version");
    }
    if (h.n_dims < 1 || h.n_dims > 4) {
        throw std::runtime_error("direct_matrix: bad n_dims");
    }
    if (h.payload_nbytes != layout.chunk_bytes[0]) {
        throw std::runtime_error("direct_matrix: payload size mismatch");
    }

    UpstreamLayoutHost out;
    out.direct_matrix = true;
    out.direct_ggml_type = h.ggml_type;
    out.n_chunks = 1;
    out.n = (int32_t) h.ne[0];
    out.padded_m = h.n_dims > 1 ? (int32_t) h.ne[1] : 1;
    out.K_groups = 1;
    out.group_size = 1;
    out.bytes_per_chunk = (size_t) h.payload_nbytes;
    out.disk_bytes_per_chunk = (size_t) h.payload_nbytes;
    out.chunks.assign(1, std::vector<uint8_t>(
        tensor_data + layout.fixed_bytes,
        tensor_data + layout.fixed_bytes + h.payload_nbytes));
    return out;
}

UpstreamLayoutHost build_upstream_layout_direct_expert_block(
    const TensorLayout & layout,
    const uint8_t * tensor_data) {
    if (tensor_data == nullptr) {
        throw std::runtime_error("direct_expert_block: null tensor_data");
    }
    if (layout.fixed_bytes != kExpertHeaderSize ||
        layout.chunk_bytes.size() != 1) {
        throw std::runtime_error(
            "direct_expert_block: expected one chunk and fixed expert header");
    }

    ExpertHeader h{};
    std::memcpy(&h, tensor_data, sizeof(h));
    if (h.magic != kExpertMagic) {
        throw std::runtime_error("direct_expert_block: bad magic");
    }
    if (h.version != kExpertVersion || h.n_records != kExpertRecords) {
        throw std::runtime_error("direct_expert_block: unsupported version/record count");
    }

    const size_t payload_nbytes = (size_t) layout.chunk_bytes[0];
    UpstreamLayoutHost out;
    out.direct_matrix = true;
    out.direct_expert_block = true;
    out.direct_layer = (int32_t) h.layer;
    out.direct_expert = (int32_t) h.expert;
    out.n_chunks = 1;
    out.n = 1;
    out.padded_m = 1;
    out.K_groups = 1;
    out.group_size = 1;
    out.bytes_per_chunk = payload_nbytes;
    out.disk_bytes_per_chunk = payload_nbytes;

    for (int i = 0; i < 3; ++i) {
        const ExpertRecord & r = h.records[i];
        if (r.kind >= 3) {
            throw std::runtime_error("direct_expert_block: bad record kind");
        }
        if ((size_t) r.offset + (size_t) r.nbytes > payload_nbytes) {
            throw std::runtime_error("direct_expert_block: record out of range");
        }
        auto & dst = out.expert_records[r.kind];
        dst.kind = (int32_t) r.kind;
        dst.ggml_type = (int32_t) r.ggml_type;
        for (int d = 0; d < 4; ++d) dst.ne[d] = (int64_t) r.ne[d];
        dst.offset = (size_t) r.offset;
        dst.nbytes = (size_t) r.nbytes;
    }

    out.chunks.assign(1, std::vector<uint8_t>(
        tensor_data + layout.fixed_bytes,
        tensor_data + layout.fixed_bytes + payload_nbytes));
    return out;
}

}}  // namespace dp_moe_ext::direct_matrix
