// streamllm-ext — algorithm-specific on-disk → chunk-layout builder.
//
// This file implements the AnyBCQ chunk layout. Other algorithms would
// supply their own upstream_layout_*.{h,cu} with a different carving.
// The framework (VramChunkPool + Scheduler) is oblivious to what's
// inside a chunk — this header defines the per-algorithm contract.
//
// On-disk format (from the Python AnyBCQ encoder):
//
//   [32-byte stream header] [d1 × fp32 beta]
//   [n_chunks × (signs_bytes + alpha_fp32)]
//
// Per managed tensor, the framework sees two chunk kinds:
//   - cid = kCidQBias       the shared meta blob (from beta).
//   - cid = kCidChunkBase+i data chunk i (signs + α, packed together).
//
// The NAVER kernel wants separate q_weight and α base pointers per
// chunk; the runtime splits one chunk allocation into the two kernel
// pointers on-the-fly via a fixed byte offset (qw_bytes_per_chunk).

#pragma once

#include "stream_reader.h"

#include <cstdint>
#include <cstddef>
#include <vector>

namespace streamllm_ext {

// Host-side, algorithm-specific layout for one managed tensor.
// ``chunks[i]`` is chunk i's packed byte blob:
//   [signs (qw_bytes_per_chunk bytes)][alpha (alpha_bytes_per_chunk bytes)]
struct UpstreamLayoutHost {
    std::vector<std::vector<uint8_t>> chunks;    // length = n_chunks
    std::vector<uint16_t>              q_bias;   // [K_groups * n] fp16

    // Per-chunk byte layout (valid after build).
    size_t qw_bytes_per_chunk    = 0;  // K/32 * n * 4
    size_t alpha_bytes_per_chunk = 0;  // K_groups * n * 2
    size_t bytes_per_chunk       = 0;  // sum of the two
    size_t q_bias_n_elem         = 0;  // K_groups * n

    int32_t n           = 0;  // = M, output features
    int32_t padded_m    = 0;  // = K
    int32_t n_chunks    = 0;  // number of data chunks per tensor
    int32_t K_groups    = 0;
    int32_t group_size  = 0;
};

// Build the host-side chunk layout from a managed tensor's on-disk
// byte region. Throws std::runtime_error on size / alignment mismatch.
UpstreamLayoutHost build_upstream_layout_host(
    const TensorLayout & layout,
    const uint8_t * tensor_data,
    uint32_t group_size);

} // namespace streamllm_ext
