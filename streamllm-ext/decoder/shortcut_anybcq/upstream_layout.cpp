// streamllm-ext / decoder / shortcut_anybcq — host layout builder impl.

#include "upstream_layout.h"
#include "anybcq_format.h"
// Per-plane pointer-table updates the shortcut after-load callback
// drives. Lives in decoder/anybcq/core/ (encoder-neutral kernel layer).
#include "anybcq_gemv.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#if defined(__F16C__)
#include <immintrin.h>
#endif

namespace streamllm_ext { namespace shortcut_anybcq {
namespace {

// Local copy of the IEEE 754 fp32 → fp16 RNE helper. The any-prec side
// has its own; duplication here keeps the per-encoder .cpp files
// independently translatable.
uint16_t fp32_to_fp16_rne(float f) {
#if defined(__F16C__)
    __m128 v = _mm_set_ss(f);
    __m128i h = _mm_cvtps_ph(v, _MM_FROUND_CUR_DIRECTION);
    return (uint16_t)_mm_extract_epi16(h, 0);
#else
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    uint32_t sign = (u >> 16) & 0x8000u;
    uint32_t x    = u & 0x7fffffffu;
    if (x >= 0x47800000u) {
        uint16_t h = (uint16_t)(sign | 0x7c00u);
        if ((u & 0x7fffffffu) > 0x7f800000u) {
            h |= 0x200u | (uint16_t)((u >> 13) & 0x3ffu);
            if ((h & 0x3ffu) == 0) h |= 1;
        }
        return h;
    }
    if (x < 0x38800000u) {
        float fa;  uint32_t xa = x;  std::memcpy(&fa, &xa, 4);
        float offset;  uint32_t o_bits = 0x33000000u;
        std::memcpy(&offset, &o_bits, 4);
        fa += offset;
        uint32_t ua;  std::memcpy(&ua, &fa, 4);
        return (uint16_t)(sign | (ua - 0x33000000u));
    }
    uint32_t xr = x + ((uint32_t)(13 + ((x >> 23) & 1)) << 22);
    return (uint16_t)(sign | ((xr - 0x38000000u) >> 13));
#endif
}

constexpr size_t kStreamHeaderSize = anybcq_format::kHeaderSize;
constexpr uint8_t kFlagAlphaFp16   = anybcq_format::flag::kAlphaFp16;
constexpr uint8_t kFlagBetaFp16    = anybcq_format::flag::kBetaFp16;

}  // anon namespace

UpstreamLayoutHost build_upstream_layout_shortcut(
    const TensorLayout & layout,
    const uint8_t * tensor_data,
    uint32_t group_size,
    uint8_t flags)
{
    const int32_t n        = (int32_t)layout.shape[0];
    const int32_t ng       = (int32_t)layout.n_groups_per_row;
    const int32_t padded_m = (int32_t)layout.padded_m;
    const int32_t K_over_32 = padded_m / 32;
    const int32_t P        = (int32_t)layout.chunk_bytes.size();
    const int32_t d1       = (int32_t)layout.d1;
    const int32_t plane_sign_bytes = (d1 * (int32_t)group_size + 7) / 8;

    const bool alpha_fp16 = (flags & kFlagAlphaFp16) != 0;
    const bool beta_fp16  = (flags & kFlagBetaFp16)  != 0;
    const int32_t a_size  = alpha_fp16 ? 2 : 4;
    const int32_t b_size  = beta_fp16  ? 2 : 4;

    if ((size_t)layout.fixed_bytes < kStreamHeaderSize + (size_t)d1 * b_size) {
        throw std::runtime_error(
            "build_upstream_layout_shortcut: fixed_bytes too small for "
            "32 B stream header + d1 × " + std::to_string(b_size) +
            " B beta");
    }
    for (int32_t p = 0; p < P; ++p) {
        int32_t expected = plane_sign_bytes + d1 * a_size;
        if ((int32_t)layout.chunk_bytes[p] != expected) {
            throw std::runtime_error(
                "build_upstream_layout_shortcut: plane " + std::to_string(p) +
                " byte size " + std::to_string(layout.chunk_bytes[p]) +
                " != signs(" + std::to_string(plane_sign_bytes) +
                ") + alpha(" + std::to_string(d1 * a_size) + ")");
        }
    }

    UpstreamLayoutHost out;
    out.n          = n;
    out.padded_m   = padded_m;
    out.n_chunks   = P;
    out.K_groups   = ng;
    out.group_size = (int32_t)group_size;
    out.qw_bytes_per_chunk    = (size_t)K_over_32 * n * sizeof(uint32_t);
    out.alpha_bytes_per_chunk = (size_t)ng       * n * sizeof(uint16_t);
    out.q_bias_n_elem         = (size_t)ng * n;
    out.q_bias_bytes_per_chunk = out.q_bias_n_elem * sizeof(uint16_t);
    out.disk_alpha_size = a_size;
    out.disk_beta_size  = b_size;
    out.any_precision   = false;
    out.bytes_per_chunk = out.qw_bytes_per_chunk + out.alpha_bytes_per_chunk;
    out.disk_bytes_per_chunk =
        (size_t)plane_sign_bytes + (size_t)d1 * (size_t)a_size;

    out.chunks.assign(P, std::vector<uint8_t>(out.bytes_per_chunk));
    out.q_bias.resize(out.q_bias_n_elem);

    // β → q_bias[kg, row] fp16, row innermost.
    const uint8_t * beta_bytes = tensor_data + kStreamHeaderSize;
    if (beta_fp16) {
        const uint16_t * beta_h = reinterpret_cast<const uint16_t *>(beta_bytes);
        for (int32_t row = 0; row < n; ++row) {
            for (int32_t kg = 0; kg < ng; ++kg) {
                out.q_bias[(size_t)kg * n + row] = beta_h[row * ng + kg];
            }
        }
    } else {
        const float * beta_f = reinterpret_cast<const float *>(beta_bytes);
        for (int32_t row = 0; row < n; ++row) {
            for (int32_t kg = 0; kg < ng; ++kg) {
                out.q_bias[(size_t)kg * n + row] =
                    fp32_to_fp16_rne(beta_f[row * ng + kg]);
            }
        }
    }

    size_t plane_off = layout.fixed_bytes;
    for (int32_t p = 0; p < P; ++p) {
        plane_disk_to_kernel(
            tensor_data + plane_off, out.chunks[p].data(),
            n, padded_m, ng, a_size);
        plane_off += layout.chunk_bytes[p];
    }

    // Encoder-side callbacks core's move_chunk invokes (no decoder
    // include needed in core).
    out.disk_to_kernel_fn = +[](
        const UpstreamLayoutHost & h,
        int /*chunk_idx*/,
        const uint8_t * disk_in,
        uint8_t       * kernel_out)
    {
        plane_disk_to_kernel(disk_in, kernel_out,
                              h.n, h.padded_m, h.K_groups,
                              h.disk_alpha_size);
    };
    out.after_load_fn = +[](
        const UpstreamLayoutHost & h,
        int chunk_idx,
        const void * chunk_device_ptr,
        void ** d_qw_ptrs,
        void ** d_alpha_ptrs,
        void ** /*d_qbias_slot*/,
        StreamHandle stream)
    {
        if (stream != nullptr) {
            anybcq::update_per_plane_after_load_async(
                d_qw_ptrs, d_alpha_ptrs, chunk_idx,
                chunk_device_ptr, h.qw_bytes_per_chunk, stream);
        } else {
            anybcq::update_per_plane_after_load(
                d_qw_ptrs, d_alpha_ptrs, chunk_idx,
                chunk_device_ptr, h.qw_bytes_per_chunk);
        }
    };
    out.after_evict_fn = +[](
        void ** d_qw_ptrs, void ** d_alpha_ptrs, int plane_idx)
    {
        anybcq::clear_per_plane_after_evict(d_qw_ptrs, d_alpha_ptrs, plane_idx);
    };

    return out;
}

void plane_disk_to_kernel(
    const uint8_t * disk_in,
    uint8_t       * kernel_out,
    int32_t         n,
    int32_t         padded_m,
    int32_t         ng,
    int32_t         disk_alpha_size)
{
    const int32_t K_over_32      = padded_m / 32;
    const int32_t bytes_per_row  = padded_m / 8;
    const int32_t plane_sign_bytes = (int32_t)((size_t)n * (size_t)bytes_per_row);
    const size_t  qw_bytes        = (size_t)K_over_32 * (size_t)n * sizeof(uint32_t);

    static const uint32_t kPow2[32] = {
        1u<<0,  1u<<1,  1u<<2,  1u<<3,  1u<<4,  1u<<5,  1u<<6,  1u<<7,
        1u<<8,  1u<<9,  1u<<10, 1u<<11, 1u<<12, 1u<<13, 1u<<14, 1u<<15,
        1u<<16, 1u<<17, 1u<<18, 1u<<19, 1u<<20, 1u<<21, 1u<<22, 1u<<23,
        1u<<24, 1u<<25, 1u<<26, 1u<<27, 1u<<28, 1u<<29, 1u<<30, 1u<<31,
    };

    const uint8_t * signs_in   = disk_in;
    const uint8_t * alpha_in_b = disk_in + plane_sign_bytes;
    uint32_t * qw_out = reinterpret_cast<uint32_t *>(kernel_out);
    uint16_t * a_out  = reinterpret_cast<uint16_t *>(kernel_out + qw_bytes);

    if (disk_alpha_size == 2) {
        const uint16_t * alpha_h = reinterpret_cast<const uint16_t *>(alpha_in_b);
        for (int32_t row = 0; row < n; ++row) {
            for (int32_t kg = 0; kg < ng; ++kg) {
                a_out[(size_t)kg * n + row] = alpha_h[row * ng + kg];
            }
        }
    } else if (disk_alpha_size == 4) {
        const float * alpha_f = reinterpret_cast<const float *>(alpha_in_b);
        for (int32_t row = 0; row < n; ++row) {
            for (int32_t kg = 0; kg < ng; ++kg) {
                a_out[(size_t)kg * n + row] =
                    fp32_to_fp16_rne(alpha_f[row * ng + kg]);
            }
        }
    } else {
        throw std::runtime_error(
            "plane_disk_to_kernel: disk_alpha_size must be 2 or 4, got " +
            std::to_string(disk_alpha_size));
    }

    for (int32_t row = 0; row < n; ++row) {
        const uint8_t * row_bytes = signs_in + (size_t)row * bytes_per_row;
        for (int32_t kt = 0; kt < K_over_32; ++kt) {
            uint32_t packed = 0;
            for (int32_t byte_idx = 0; byte_idx < 4; ++byte_idx) {
                uint8_t b = row_bytes[kt * 4 + byte_idx];
                for (int32_t bit_in_byte = 0; bit_in_byte < 8; ++bit_in_byte) {
                    int32_t col_in_tile = byte_idx * 8 + bit_in_byte;
                    int32_t sign_bit = (b >> (7 - bit_in_byte)) & 1;
                    if (sign_bit) packed |= kPow2[col_in_tile];
                }
            }
            qw_out[(size_t)kt * n + row] = packed;
        }
    }
}

}}  // namespace streamllm_ext::shortcut_anybcq
