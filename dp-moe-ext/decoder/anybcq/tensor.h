// DPMoE / decoder / anybcq — concrete ChunkedTensor wrapper for
// AnyBCQ-family tensors (any-prec + ss_anybcq variants).
//
// Owns one tensor's per-tensor encoder state:
//   - the gguf-parsed host layout (byte sizes, chunk_planes, base_p
//     metadata, β buffer, etc.),
//   - the device-side per-plane pointer tables the kernel reads
//     (d_chunk_qw_ptrs / d_chunk_alpha_ptrs / d_qbias_slot).
//
// Implements core/chunked_tensor.h's ChunkedTensor ABC. The framework
// drives loads/transforms/cleanup via the virtual methods; encoder
// specifics stay routed through the function-pointer callbacks the
// upstream-layout parser installed on ``host``.
//
// Both any-prec and ss_anybcq share this single class — the byte-format
// differences are absorbed by the parser's chosen function pointers.
// Distinct C++ types (AnyBCQTensor, SsAnybcqTensor) are exposed as
// thin aliases so call sites can name the encoder family they expect.

#pragma once

#include "chunked_tensor.h"
#include "upstream_layout.h"

#include <string>

namespace dp_moe_ext { namespace anybcq {

class AnyBCQFamilyTensor : public ChunkedTensor {
public:
    AnyBCQFamilyTensor(std::string wid, UpstreamLayoutHost layout);

    // ─── ChunkedTensor virtuals ────────────────────────────────
    const std::string & id()        const override { return wid_; }
    int                 n_chunks()  const override { return host_.n_chunks; }
    size_t              disk_bytes  (int chunk_idx) const override;
    size_t              kernel_bytes(int chunk_idx) const override;

    void disk_to_kernel(int chunk_idx,
                         const uint8_t * disk_in,
                         uint8_t       * kernel_out) const override;
    void after_load    (int chunk_idx,
                         const void * device_ptr,
                         StreamHandle stream) override;
    void after_evict   (int chunk_idx,
                         StreamHandle stream) override;
    bool append_after_load_patch(int chunk_idx,
                                 const void * device_ptr,
                                 std::vector<PointerPatch> & out) override;
    bool append_after_evict_patch(int chunk_idx,
                                  std::vector<PointerPatch> & out) override;

    // ─── Encoder-private accessors ─────────────────────────────
    UpstreamLayoutHost &       host() override       { return host_; }
    const UpstreamLayoutHost & host() const override { return host_; }

    // Bind the per-plane device pointer tables. Called once at install
    // by the runtime after slab-allocating the three buffers.  The
    // tensor's after_load / after_evict virtuals subsequently mutate
    // these in place; the kernel reads them by pointer at launch.
    //   - d_qw_ptrs    : void** [kMaxChunksPerTensor], plane → signs base
    //   - d_alpha_ptrs : void** [kMaxChunksPerTensor], plane → α base
    //   - d_qbias_slot : void**  [1] for any-prec (highest-active β
    //                    pointer); ignored for ss_anybcq.
    void set_device_state(void ** d_qw_ptrs,
                          void ** d_alpha_ptrs,
                          void ** d_qbias_slot) override;
    void ** d_qw_ptrs()    const { return d_qw_ptrs_; }
    void ** d_alpha_ptrs() const { return d_alpha_ptrs_; }
    void ** d_qbias_slot() const { return d_qbias_slot_; }

private:
    std::string         wid_;
    UpstreamLayoutHost  host_;
    void **             d_qw_ptrs_     = nullptr;
    void **             d_alpha_ptrs_  = nullptr;
    void **             d_qbias_slot_  = nullptr;
};

// Encoder-family tag. Identical behavior to AnyBCQFamilyTensor; the
// distinct type lets call sites assert "this came from the any-prec
// parser" when they need the precision-dial semantics.
class AnyBCQTensor : public AnyBCQFamilyTensor {
public:
    using AnyBCQFamilyTensor::AnyBCQFamilyTensor;
};

}}  // namespace dp_moe_ext::anybcq
