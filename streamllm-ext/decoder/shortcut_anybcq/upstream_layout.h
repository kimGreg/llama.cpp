// streamllm-ext / decoder / shortcut_anybcq — host layout builder.
//
// SHORTCUT byte format on disk:
//   [32 B header] [d1 × β (fp16/fp32)]                    ← fixed_meta
//   [P chunks × (plane_sign_bytes + d1 × α_dtype)]        ← one plane per chunk
//
// β lives in the install-pinned ``kCidQBias`` blob; each data chunk owns
// one plane of signs + that plane's α scalar, so the kernel sums over
// the resident chunks (precision = chunk count).

#pragma once

// Reuses the shared UpstreamLayoutHost struct. The struct has both
// shortcut and any-prec fields; the shortcut builder fills only the
// shortcut-relevant ones.
#include "../anybcq/upstream_layout.h"

#include <cstdint>

namespace streamllm_ext { namespace shortcut_anybcq {

// Build the host-side chunk layout for a SHORTCUT-encoded tensor.
// Caller has already validated the magic + version + flag bits.
UpstreamLayoutHost build_upstream_layout_shortcut(
    const TensorLayout & layout,
    const uint8_t * tensor_data,
    uint32_t group_size,
    uint8_t flags);

// Transform one plane's disk-format bytes (signs MSB-packed bytes +
// alpha [fp16 v2, fp32 legacy], both row-major) into kernel-format
// bytes (signs LSB-packed uint32 transposed to [kt, row] + alpha fp16
// transposed to [kg, row]).
//
// ``disk_in`` size = plane_sign_bytes + d1 × disk_alpha_size.
// ``kernel_out`` size = qw_bytes_per_chunk + alpha_bytes_per_chunk.
//
// Used by both install-time host-buffer construction and the runtime's
// SSD-stream HOT-promotion path so a chunk fetched at runtime is byte-
// identical to one pinned at install.
void plane_disk_to_kernel(
    const uint8_t * disk_in,
    uint8_t       * kernel_out,
    int32_t         n,
    int32_t         padded_m,
    int32_t         ng,
    int32_t         disk_alpha_size = 2);

}}  // namespace streamllm_ext::shortcut_anybcq
