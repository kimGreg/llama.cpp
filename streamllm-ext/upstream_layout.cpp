// streamllm-ext — host-side builder for NAVER's kernel-ready layout.

#include "upstream_layout.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#if defined(__F16C__)
#include <immintrin.h>
#endif

namespace streamllm_ext {
namespace {

// The stream header in ct.data, mirroring AnyBCQEncoder.HEADER_STRUCT:
//   magic u32, version u8, group_size u16, base_precision u8,
//   target_precision u8, pad 3, n u32, m u32, n_groups_per_row u32,
//   d1 u32, pad 4
// Total = 32 bytes. Sits at the start of each managed tensor's bytes.
constexpr size_t kStreamHeaderSize = 32;

// IEEE 754 fp32 → fp16 with round-to-nearest-even. Matches numpy's
// astype('<f2') and PyTorch's .to(torch.float16).
//
// Use F16C hardware intrinsic when available (Intel Haswell+ 2013,
// AMD Piledriver+ 2012) — gives exact IEEE 754 RNE. Fall back to a
// portable softfloat implementation otherwise (agrees with numpy /
// PyTorch on all values including subnormals and ±Inf/NaN).
uint16_t fp32_to_fp16_rne(float f) {
#if defined(__F16C__)
    __m128 v = _mm_set_ss(f);
    __m128i h = _mm_cvtps_ph(v, _MM_FROUND_CUR_DIRECTION);
    return (uint16_t)_mm_extract_epi16(h, 0);
#else
    // Portable IEEE 754 round-to-nearest-even.
    // Adapted from https://gist.github.com/rygorous/2156668 (public domain).
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    uint32_t sign = (u >> 16) & 0x8000u;
    uint32_t x    = u & 0x7fffffffu;

    // Rebase fp32 exponent (bias 127) to fp16 (bias 15).
    // Then test against special ranges.
    if (x >= 0x47800000u) {  // > max fp16 finite → Inf or NaN
        uint16_t h = (uint16_t)(sign | 0x7c00u);
        if ((u & 0x7fffffffu) > 0x7f800000u) {  // NaN
            h |= 0x200u | (uint16_t)((u >> 13) & 0x3ffu);
            if ((h & 0x3ffu) == 0) h |= 1;  // ensure NaN not Inf
        }
        return h;
    }
    if (x < 0x38800000u) {
        // Subnormal or zero. Add magic value to force RNE into subnormal range.
        float fa;
        uint32_t xa = x;
        std::memcpy(&fa, &xa, 4);
        float offset;
        uint32_t o_bits = 0x33000000u;  // 2^-25
        std::memcpy(&offset, &o_bits, 4);
        fa += offset;
        uint32_t ua;
        std::memcpy(&ua, &fa, 4);
        return (uint16_t)(sign | (ua - 0x33000000u));
    }
    // Normal range: rebase exponent, round mantissa with RNE.
    uint32_t xr = x + ((uint32_t)(13 + ((x >> 23) & 1)) << 22);
    return (uint16_t)(sign | ((xr - 0x38000000u) >> 13));
#endif
}

} // anonymous namespace


UpstreamLayoutHost build_upstream_layout_host(
    const TensorLayout & layout,
    const uint8_t * tensor_data,
    uint32_t group_size)
{
    if (tensor_data == nullptr) {
        throw std::runtime_error("build_upstream_layout_host: null tensor_data");
    }
    if (group_size == 0 || layout.n_groups_per_row == 0 || layout.padded_m == 0) {
        throw std::runtime_error(
            "build_upstream_layout_host: missing layout fields; parse StreamReader first");
    }
    if (layout.padded_m % 32 != 0) {
        throw std::runtime_error(
            "build_upstream_layout_host: padded_m (" +
            std::to_string(layout.padded_m) + ") not divisible by 32");
    }
    const int32_t n        = (int32_t)layout.shape[0];
    const int32_t ng       = (int32_t)layout.n_groups_per_row;
    const int32_t padded_m = (int32_t)layout.padded_m;
    const int32_t K_over_32 = padded_m / 32;
    const int32_t P        = (int32_t)layout.chunk_bytes.size();
    const int32_t d1       = (int32_t)layout.d1;
    const int32_t plane_sign_bytes = (d1 * (int32_t)group_size + 7) / 8;

    // The fixed_meta prefix is [32-byte stream header] + [d1 × fp32 beta].
    // Layout should be consistent with the Python encoder.
    if ((size_t)layout.fixed_bytes < kStreamHeaderSize + (size_t)d1 * 4) {
        throw std::runtime_error(
            "build_upstream_layout_host: fixed_bytes too small for "
            "32 B stream header + d1 fp32 beta");
    }

    // Per-plane size check: signs_bytes + d1 * fp32.
    for (int32_t p = 0; p < P; ++p) {
        int32_t expected = plane_sign_bytes + d1 * 4;
        if ((int32_t)layout.chunk_bytes[p] != expected) {
            throw std::runtime_error(
                "build_upstream_layout_host: plane " + std::to_string(p) +
                " byte size " + std::to_string(layout.chunk_bytes[p]) +
                " != signs(" + std::to_string(plane_sign_bytes) +
                ") + alpha(" + std::to_string(d1 * 4) + ")");
        }
    }
    if (d1 != n * ng) {
        throw std::runtime_error(
            "build_upstream_layout_host: d1 (" + std::to_string(d1) +
            ") != n*ng (" + std::to_string(n * ng) + ")");
    }

    // --- beta → q_bias[K_groups, M]  (M = n, K_groups = ng) ---
    const float * beta = reinterpret_cast<const float *>(tensor_data + kStreamHeaderSize);

    UpstreamLayoutHost out;
    out.n          = n;
    out.padded_m   = padded_m;
    out.n_chunks          = P;
    out.K_groups   = ng;
    out.group_size = (int32_t)group_size;
    out.qw_bytes_per_chunk    = (size_t)K_over_32 * n * sizeof(uint32_t);
    out.alpha_bytes_per_chunk = (size_t)ng       * n * sizeof(uint16_t);
    out.bytes_per_chunk = out.qw_bytes_per_chunk + out.alpha_bytes_per_chunk;
    out.q_bias_n_elem         = (size_t)ng * n;

    out.chunks.assign(P, std::vector<uint8_t>(out.bytes_per_chunk));
    out.q_bias.resize(out.q_bias_n_elem);

    // beta[d1 = row*ng + kg] → q_bias[kg, row] fp16, row innermost.
    for (int32_t row = 0; row < n; ++row) {
        for (int32_t kg = 0; kg < ng; ++kg) {
            out.q_bias[(size_t)kg * n + row] = fp32_to_fp16_rne(beta[row * ng + kg]);
        }
    }

    // --- per-plane signs + α re-pack ---
    //
    // Each plane's chunk is one contiguous byte blob:
    //
    //   [0                        .. qw_bytes)          signs  [K/32, n] uint32
    //   [qw_bytes                 .. qw+alpha_bytes)    alpha  [ng,   n] fp16
    //
    // The runtime later splits these with a fixed byte offset to feed
    // the kernel its q_weight and α base pointers separately.
    size_t plane_off = layout.fixed_bytes;  // into tensor_data
    const int32_t bytes_per_row = padded_m / 8;
    // LSB-first powers-of-two: bit t of uint32 ↔ col (kt*32 + t).
    static const uint32_t kPow2[32] = {
        1u<<0,  1u<<1,  1u<<2,  1u<<3,  1u<<4,  1u<<5,  1u<<6,  1u<<7,
        1u<<8,  1u<<9,  1u<<10, 1u<<11, 1u<<12, 1u<<13, 1u<<14, 1u<<15,
        1u<<16, 1u<<17, 1u<<18, 1u<<19, 1u<<20, 1u<<21, 1u<<22, 1u<<23,
        1u<<24, 1u<<25, 1u<<26, 1u<<27, 1u<<28, 1u<<29, 1u<<30, 1u<<31,
    };

    for (int32_t p = 0; p < P; ++p) {
        const uint8_t * signs_in = tensor_data + plane_off;
        const float *   alpha_in = reinterpret_cast<const float *>(
            signs_in + plane_sign_bytes);

        uint8_t * chunk    = out.chunks[p].data();
        uint32_t * qw_out  = reinterpret_cast<uint32_t *>(chunk);
        uint16_t * a_out   = reinterpret_cast<uint16_t *>(chunk + out.qw_bytes_per_chunk);

        // alpha[d1 = row*ng + kg] → α_out[kg, row] fp16.
        for (int32_t row = 0; row < n; ++row) {
            for (int32_t kg = 0; kg < ng; ++kg) {
                a_out[(size_t)kg * n + row] =
                    fp32_to_fp16_rne(alpha_in[row * ng + kg]);
            }
        }

        // Signs: per-row, pack 32 MSB-first bits of each byte into
        // LSB-first uint32. Result in qw_out[kt, row].
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

        plane_off += layout.chunk_bytes[p];
    }

    return out;
}

} // namespace streamllm_ext
