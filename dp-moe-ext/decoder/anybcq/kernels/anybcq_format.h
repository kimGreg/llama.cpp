// DPMoE / decoder / anybcq — single source of truth for the
// AnyBCQ byte stream layout.
//
// **MIRROR THIS FILE IN PYTHON**:
//   src/dp_moe/encoders/format/anybcq_format.py
//
// Every constant / struct here has an identical counterpart in the
// Python file. The cross-language test
//   src/dp_moe/encoders/tests/test_anybcq_format_parity.py
// asserts byte-for-byte equivalence by encoding a tensor in Python and
// parsing the bytes via these C++ structs. A drift in either file fails
// the test.
//
// Layout summary (v2):
//
//   HEADER (32 B)        StreamHeader, see below
//   FIXED_META           ss_anybcq: d1 × β-dtype; any-prec: empty
//   CHUNKS               per-chunk byte recipe — see ShortcutChunkLayout
//                        / AnyPrecChunkLayout
//
// All multi-byte fields are little-endian. Byte stream's outer container
// is the GGUF tensor `dp_moe.bytes.<canonical>` blob.

#pragma once

#include <cstdint>
#include <cstddef>

namespace dp_moe_ext {
namespace anybcq_format {

// ─── Header constants ─────────────────────────────────────────────────

constexpr uint32_t kStreamMagic   = 0x414E5942u;  // "ANYB" LE
constexpr uint8_t  kStreamVersion = 2;
constexpr size_t   kHeaderSize    = 32;

// Header flag-byte bits (offset 9 in the header).
namespace flag {
constexpr uint8_t kAlphaFp16 = 1u << 0;  // 1 = α stored as fp16, 0 = fp32
constexpr uint8_t kBetaFp16  = 1u << 1;  // 1 = β stored as fp16, 0 = fp32
constexpr uint8_t kAnyPrec   = 1u << 2;  // 1 = per-precision-α layout
                                          // 0 = ss_anybcq (one α set)
}

// 32-byte stream header. Mirrors Python ``struct.Struct("<IBHBBB2xIIII4x")``.
//
// Field order matches the encoded bytes exactly (the struct is packed under
// GCC/Clang via `__attribute__((packed))`; static_assert enforces size).
#pragma pack(push, 1)
struct StreamHeader {
    uint32_t magic;             // = kStreamMagic
    uint8_t  version;           // = kStreamVersion
    uint16_t group_size;        // weights per group
    uint8_t  base_precision;    // ∈ [1, target_precision]
    uint8_t  target_precision;  // ∈ [base_precision, 16]
    uint8_t  flags;             // bitfield from `flag::*`
    uint8_t  pad1[2];
    uint32_t n;                 // out features
    uint32_t m;                 // in features
    uint32_t n_groups_per_row;  // ceil(m / group_size)
    uint32_t d1;                // = n × n_groups_per_row, # of groups total
    uint32_t pad2;
};
#pragma pack(pop)
static_assert(sizeof(StreamHeader) == kHeaderSize,
              "StreamHeader must be 32 bytes");

// ─── Per-chunk byte recipes ───────────────────────────────────────────
//
// One struct per layout variant. These describe the SHAPE of a chunk on
// disk; the C++ runtime uses them to compute byte offsets, the Python
// encoder uses them to predict chunk_bytes for GGUF metadata.

// SsAnybcq layout (flag::kAnyPrec = 0): one plane per chunk.
//   [signs (plane_sign_bytes)][alpha (alpha_bytes)]
struct ShortcutChunkLayout {
    int32_t plane_sign_bytes;   // = (d1 × group_size + 7) / 8
    int32_t alpha_bytes;        // = d1 × alpha_dtype_size

    constexpr size_t disk_chunk_bytes() const noexcept {
        return (size_t)plane_sign_bytes + (size_t)alpha_bytes;
    }
};

// Any-precision layout (flag::kAnyPrec = 1): one precision level per chunk.
// chunk[i] (i = 0..target_p-base_p) describes precision p = base_p + i:
//   chunk[0]: signs of base_p planes back-to-back, then α^(p), then β^(p)
//   chunk[i>0]: signs of one new plane, then α^(p), then β^(p)
struct AnyPrecChunkLayout {
    int32_t n_planes;             // base_p for chunk 0, 1 for chunks i > 0
    int32_t precision_at_chunk;   // p = base_p + i
    int32_t plane_sign_bytes;     // per plane = (d1 × group_size + 7) / 8
    int32_t alpha_bytes;          // = d1 × precision × alpha_dtype_size
    int32_t beta_bytes;           // = d1 × beta_dtype_size

    constexpr size_t disk_chunk_bytes() const noexcept {
        return (size_t)n_planes * (size_t)plane_sign_bytes
             + (size_t)alpha_bytes
             + (size_t)beta_bytes;
    }
};

// ─── dtype size helpers ──────────────────────────────────────────────

constexpr int32_t alpha_dtype_size(uint8_t flags) noexcept {
    return (flags & flag::kAlphaFp16) ? 2 : 4;
}
constexpr int32_t beta_dtype_size(uint8_t flags) noexcept {
    return (flags & flag::kBetaFp16) ? 2 : 4;
}

constexpr int32_t plane_sign_bytes_for(int32_t d1, int32_t group_size) noexcept {
    return (d1 * group_size + 7) / 8;
}

}  // namespace anybcq_format
}  // namespace dp_moe_ext
