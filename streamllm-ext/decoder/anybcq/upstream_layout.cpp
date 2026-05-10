// streamllm-ext / decoder / anybcq — host layout builder impl
// (any-prec specifics + top-level dispatcher).

#include "upstream_layout.h"
#include "anybcq_format.h"
// Per-plane pointer-table updates the any-prec after-load callback
// drives. Lives in decoder/anybcq/kernels/.
#include "anybcq_gemv.h"
// Concrete ChunkedTensor wrappers (any-prec + shortcut share the
// AnyBCQFamilyTensor base; the typed subclasses are the public face).
#include "tensor.h"
#include "../shortcut_anybcq/tensor.h"

// Cross-encoder forward into the shortcut parser. The dispatcher reads
// the flag bit and routes; calling cross-encoder is fine because this
// is the encoder-selection layer.
#include "../shortcut_anybcq/upstream_layout.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#if defined(__F16C__)
#include <immintrin.h>
#endif

namespace streamllm_ext {
namespace {

// IEEE 754 fp32 → fp16 with round-to-nearest-even. Local copy; the
// shortcut .cpp has its own to keep the per-encoder TUs independently
// translatable.
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

constexpr size_t   kStreamHeaderSize = anybcq_format::kHeaderSize;
constexpr uint32_t kStreamMagic      = anybcq_format::kStreamMagic;
constexpr uint8_t  kStreamVersion    = anybcq_format::kStreamVersion;
constexpr uint8_t  kFlagAnyPrec      = anybcq_format::flag::kAnyPrec;

// Any-prec parser — only called from the dispatcher when flag bit
// kAnyPrec is set.
UpstreamLayoutHost build_upstream_layout_anyprec(
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

    const bool alpha_fp16 = (flags & anybcq_format::flag::kAlphaFp16) != 0;
    const bool beta_fp16  = (flags & anybcq_format::flag::kBetaFp16)  != 0;
    const int32_t a_size  = alpha_fp16 ? 2 : 4;
    const int32_t b_size  = beta_fp16  ? 2 : 4;

    const uint8_t hdr_base_p   = *(tensor_data + 7);
    const uint8_t hdr_target_p = *(tensor_data + 8);

    if ((size_t)layout.fixed_bytes != kStreamHeaderSize) {
        throw std::runtime_error(
            "build_upstream_layout_anyprec: any-prec fixed_bytes must be 32 "
            "(header-only); got " + std::to_string(layout.fixed_bytes));
    }
    if (P != (int32_t)hdr_target_p - (int32_t)hdr_base_p + 1) {
        throw std::runtime_error(
            "build_upstream_layout_anyprec: any-prec chunk count " +
            std::to_string(P) + " != target − base + 1 (" +
            std::to_string((int)hdr_target_p) + " − " +
            std::to_string((int)hdr_base_p) + " + 1)");
    }
    for (int32_t i = 0; i < P; ++i) {
        const int32_t precision = (int32_t)hdr_base_p + i;
        const int32_t n_planes  = (i == 0) ? (int32_t)hdr_base_p : 1;
        const int32_t expected =
            n_planes * plane_sign_bytes +
            precision * d1 * a_size +
            d1 * b_size;
        if ((int32_t)layout.chunk_bytes[i] != expected) {
            throw std::runtime_error(
                "build_upstream_layout_anyprec: any-prec chunk " +
                std::to_string(i) + " (P=" + std::to_string(precision) +
                ") byte size " + std::to_string(layout.chunk_bytes[i]) +
                " != " + std::to_string(expected));
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
    out.any_precision   = true;
    out.base_precision   = (int32_t)hdr_base_p;
    out.target_precision = (int32_t)hdr_target_p;

    out.q_bias.clear();
    out.chunks.assign(P, std::vector<uint8_t>{});
    out.chunk_planes.assign(P, UpstreamLayoutHost::ChunkPlanes{});

    size_t disk_off = layout.fixed_bytes;
    int32_t plane_idx_running = 0;
    size_t  max_kernel_chunk_bytes = 0;

    for (int32_t i = 0; i < P; ++i) {
        const int32_t precision = (int32_t)hdr_base_p + i;
        const int32_t n_planes  = (i == 0) ? (int32_t)hdr_base_p : 1;
        const size_t  ker_signs_bytes = (size_t)n_planes * out.qw_bytes_per_chunk;
        const size_t  ker_alpha_bytes = (size_t)precision * out.alpha_bytes_per_chunk;
        const size_t  ker_qbias_bytes = out.q_bias_bytes_per_chunk;
        const size_t  ker_chunk_bytes =
            ker_signs_bytes + ker_alpha_bytes + ker_qbias_bytes;
        if (ker_chunk_bytes > max_kernel_chunk_bytes) {
            max_kernel_chunk_bytes = ker_chunk_bytes;
        }

        out.chunks[i].assign(ker_chunk_bytes, 0);
        any_prec_chunk_disk_to_kernel(
            tensor_data + disk_off,
            out.chunks[i].data(),
            n, padded_m, ng,
            n_planes, precision,
            a_size, b_size);

        out.chunk_planes[i].precision_at_chunk  = precision;
        out.chunk_planes[i].n_planes_this_chunk = n_planes;
        out.chunk_planes[i].plane_idx_first     = plane_idx_running;
        out.chunk_planes[i].kernel_chunk_bytes  = ker_chunk_bytes;
        out.chunk_planes[i].disk_chunk_bytes    = (size_t)layout.chunk_bytes[i];
        out.chunk_planes[i].ker_off_signs       = 0;
        out.chunk_planes[i].ker_off_alpha       = ker_signs_bytes;
        out.chunk_planes[i].ker_off_qbias       = ker_signs_bytes + ker_alpha_bytes;
        plane_idx_running += n_planes;

        disk_off += layout.chunk_bytes[i];
    }
    out.bytes_per_chunk      = max_kernel_chunk_bytes;
    out.disk_bytes_per_chunk = 0;  // variable per chunk; SSD-stream path
                                    // reads chunk_planes[i].disk_chunk_bytes.

    // Encoder-side callbacks core's move_chunk invokes (no decoder
    // include needed in core).
    out.disk_to_kernel_fn = +[](
        const UpstreamLayoutHost & h,
        int chunk_idx,
        const uint8_t * disk_in,
        uint8_t       * kernel_out)
    {
        const auto & cp = h.chunk_planes[chunk_idx];
        any_prec_chunk_disk_to_kernel(
            disk_in, kernel_out,
            h.n, h.padded_m, h.K_groups,
            cp.n_planes_this_chunk, cp.precision_at_chunk,
            h.disk_alpha_size, h.disk_beta_size);
    };
    out.after_load_fn = +[](
        const UpstreamLayoutHost & h,
        int chunk_idx,
        const void * chunk_device_ptr,
        void ** d_qw_ptrs,
        void ** d_alpha_ptrs,
        void ** d_qbias_slot,
        StreamHandle stream)
    {
        const auto & cp = h.chunk_planes[chunk_idx];
        if (stream != nullptr) {
            anybcq::update_anyprec_after_load_async(
                d_qw_ptrs, d_alpha_ptrs, d_qbias_slot,
                cp.plane_idx_first, cp.n_planes_this_chunk,
                cp.precision_at_chunk,
                chunk_device_ptr,
                h.qw_bytes_per_chunk, h.alpha_bytes_per_chunk,
                h.q_bias_bytes_per_chunk, stream);
        } else {
            anybcq::update_anyprec_after_load(
                d_qw_ptrs, d_alpha_ptrs, d_qbias_slot,
                cp.plane_idx_first, cp.n_planes_this_chunk,
                cp.precision_at_chunk,
                chunk_device_ptr,
                h.qw_bytes_per_chunk, h.alpha_bytes_per_chunk,
                h.q_bias_bytes_per_chunk);
        }
    };
    out.after_evict_fn = +[](
        void ** d_qw_ptrs, void ** d_alpha_ptrs, int plane_idx)
    {
        anybcq::clear_per_plane_after_evict(d_qw_ptrs, d_alpha_ptrs, plane_idx);
    };

    return out;
}

}  // anonymous namespace

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
    const int32_t n  = (int32_t)layout.shape[0];
    const int32_t ng = (int32_t)layout.n_groups_per_row;
    const int32_t d1 = (int32_t)layout.d1;
    if (d1 != n * ng) {
        throw std::runtime_error(
            "build_upstream_layout_host: d1 (" + std::to_string(d1) +
            ") != n*ng (" + std::to_string(n * ng) + ")");
    }

    // Read + validate stream header (magic / version / flags).
    const uint32_t magic   = *reinterpret_cast<const uint32_t *>(tensor_data + 0);
    const uint8_t  version = *(tensor_data + 4);
    const uint8_t  flags   = *(tensor_data + 9);
    if (magic != kStreamMagic) {
        throw std::runtime_error(
            "build_upstream_layout_host: bad anybcq stream magic 0x" +
            std::to_string(magic));
    }
    if (version != kStreamVersion) {
        throw std::runtime_error(
            "build_upstream_layout_host: unsupported anybcq stream version " +
            std::to_string((int)version) + "; this build expects " +
            std::to_string((int)kStreamVersion) + ". Re-encode the artifact.");
    }

    if ((flags & kFlagAnyPrec) != 0) {
        return build_upstream_layout_anyprec(layout, tensor_data, group_size, flags);
    }
    return shortcut_anybcq::build_upstream_layout_shortcut(
        layout, tensor_data, group_size, flags);
}

std::unique_ptr<anybcq::AnyBCQFamilyTensor> wrap_host_in_tensor(
    std::string         wid,
    UpstreamLayoutHost  host)
{
    if (host.any_precision) {
        return std::unique_ptr<anybcq::AnyBCQFamilyTensor>(
            new anybcq::AnyBCQTensor(std::move(wid), std::move(host)));
    }
    return std::unique_ptr<anybcq::AnyBCQFamilyTensor>(
        new shortcut_anybcq::ShortcutTensor(std::move(wid), std::move(host)));
}

std::unique_ptr<ChunkedTensor> build_chunked_tensor(
    std::string          wid,
    const TensorLayout & layout,
    const uint8_t * tensor_data,
    uint32_t group_size)
{
    return wrap_host_in_tensor(
        std::move(wid),
        build_upstream_layout_host(layout, tensor_data, group_size));
}

void any_prec_chunk_disk_to_kernel(
    const uint8_t * disk_in,
    uint8_t       * kernel_out,
    int32_t         n,
    int32_t         padded_m,
    int32_t         ng,
    int32_t         n_planes_in_chunk,
    int32_t         precision_at_chunk,
    int32_t         disk_alpha_size,
    int32_t         disk_beta_size)
{
    const int32_t K_over_32       = padded_m / 32;
    const int32_t bytes_per_row   = padded_m / 8;
    const int32_t plane_sign_bytes = (int32_t)((size_t)n * (size_t)bytes_per_row);
    const int32_t d1              = n * ng;
    const size_t  qw_bytes        = (size_t)K_over_32 * (size_t)n * sizeof(uint32_t);
    const size_t  alpha_bytes     = (size_t)ng * (size_t)n * sizeof(uint16_t);

    static const uint32_t kPow2[32] = {
        1u<<0,  1u<<1,  1u<<2,  1u<<3,  1u<<4,  1u<<5,  1u<<6,  1u<<7,
        1u<<8,  1u<<9,  1u<<10, 1u<<11, 1u<<12, 1u<<13, 1u<<14, 1u<<15,
        1u<<16, 1u<<17, 1u<<18, 1u<<19, 1u<<20, 1u<<21, 1u<<22, 1u<<23,
        1u<<24, 1u<<25, 1u<<26, 1u<<27, 1u<<28, 1u<<29, 1u<<30, 1u<<31,
    };

    const uint8_t * disk_signs = disk_in;
    const uint8_t * disk_alpha = disk_in +
        (size_t)n_planes_in_chunk * (size_t)plane_sign_bytes;
    const uint8_t * disk_beta  = disk_alpha +
        (size_t)precision_at_chunk * (size_t)d1 * (size_t)disk_alpha_size;

    uint8_t * ker_signs = kernel_out;
    uint8_t * ker_alpha = kernel_out + (size_t)n_planes_in_chunk * qw_bytes;
    uint8_t * ker_beta  = ker_alpha + (size_t)precision_at_chunk * alpha_bytes;

    // Signs.
    for (int32_t plane = 0; plane < n_planes_in_chunk; ++plane) {
        const uint8_t * sin = disk_signs + (size_t)plane * plane_sign_bytes;
        uint32_t * sout = reinterpret_cast<uint32_t *>(
            ker_signs + (size_t)plane * qw_bytes);
        for (int32_t row = 0; row < n; ++row) {
            const uint8_t * row_bytes = sin + (size_t)row * bytes_per_row;
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
                sout[(size_t)kt * n + row] = packed;
            }
        }
    }

    // α: disk α[d1_row × precision + plane_idx] → ker α[plane_idx][kg, row].
    for (int32_t plane = 0; plane < precision_at_chunk; ++plane) {
        uint16_t * aout = reinterpret_cast<uint16_t *>(
            ker_alpha + (size_t)plane * alpha_bytes);
        if (disk_alpha_size == 2) {
            const uint16_t * ain = reinterpret_cast<const uint16_t *>(disk_alpha);
            for (int32_t row = 0; row < n; ++row) {
                for (int32_t kg = 0; kg < ng; ++kg) {
                    const int32_t d1_row = row * ng + kg;
                    aout[(size_t)kg * n + row] =
                        ain[(size_t)d1_row * precision_at_chunk + plane];
                }
            }
        } else if (disk_alpha_size == 4) {
            const float * ain = reinterpret_cast<const float *>(disk_alpha);
            for (int32_t row = 0; row < n; ++row) {
                for (int32_t kg = 0; kg < ng; ++kg) {
                    const int32_t d1_row = row * ng + kg;
                    aout[(size_t)kg * n + row] = fp32_to_fp16_rne(
                        ain[(size_t)d1_row * precision_at_chunk + plane]);
                }
            }
        } else {
            throw std::runtime_error(
                "any_prec_chunk_disk_to_kernel: disk_alpha_size must be 2 or 4");
        }
    }

    // β: disk β[d1_row] → ker q_bias[kg, row].
    uint16_t * bout = reinterpret_cast<uint16_t *>(ker_beta);
    if (disk_beta_size == 2) {
        const uint16_t * bin = reinterpret_cast<const uint16_t *>(disk_beta);
        for (int32_t row = 0; row < n; ++row) {
            for (int32_t kg = 0; kg < ng; ++kg) {
                bout[(size_t)kg * n + row] = bin[row * ng + kg];
            }
        }
    } else if (disk_beta_size == 4) {
        const float * bin = reinterpret_cast<const float *>(disk_beta);
        for (int32_t row = 0; row < n; ++row) {
            for (int32_t kg = 0; kg < ng; ++kg) {
                bout[(size_t)kg * n + row] = fp32_to_fp16_rne(bin[row * ng + kg]);
            }
        }
    } else {
        throw std::runtime_error(
            "any_prec_chunk_disk_to_kernel: disk_beta_size must be 2 or 4");
    }
}

}  // namespace streamllm_ext
