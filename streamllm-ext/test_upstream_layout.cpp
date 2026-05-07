// streamllm-ext — parity test for the host-side upstream-layout builder.
//
// Usage:
//   streamllm-upstream-test <path.gguf> <tensor_name> <ref_dump.bin>
//
// Reads the GGUF, picks the managed tensor, runs
// build_upstream_layout_host, and compares the resulting q_weight /
// alpha / q_bias byte-for-byte against a Python-produced reference
// dump (see streamllm/scripts/dump_upstream_layout.py).
//
// Exits 0 on exact match, 1 on any mismatch or error.

#include "stream_reader.h"
#include "upstream_layout.h"

#include <gguf.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using streamllm_ext::StreamReader;
using streamllm_ext::TensorLayout;
using streamllm_ext::build_upstream_layout_host;
using streamllm_ext::UpstreamLayoutHost;

namespace {

// Match streamllm/scripts/dump_upstream_layout.py REF_HEADER.
#pragma pack(push, 1)
struct RefHeader {
    uint32_t magic;        // 'SUPL' = 0x4C505553
    uint32_t version;      // 1
    int32_t  n;
    int32_t  m;
    int32_t  P;
    int32_t  group_size;
    int32_t  n_groups_per_row;
    int32_t  padded_m;
};
#pragma pack(pop)
static_assert(sizeof(RefHeader) == 32, "RefHeader must be 32 bytes");

std::vector<uint8_t> read_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open " + path);
    auto sz = f.tellg();
    f.seekg(0);
    std::vector<uint8_t> buf((size_t)sz);
    f.read((char *)buf.data(), sz);
    return buf;
}

// Raw pread from a regular file at an absolute offset.
std::vector<uint8_t> pread_range(const std::string & path,
                                 int64_t offset, int64_t nbytes) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    f.seekg(offset);
    std::vector<uint8_t> buf((size_t)nbytes);
    f.read((char *)buf.data(), nbytes);
    if (f.gcount() != nbytes) {
        throw std::runtime_error(
            "pread short read: wanted " + std::to_string(nbytes) +
            ", got " + std::to_string(f.gcount()));
    }
    return buf;
}

template <typename T>
bool compare_buf(const T * cxx, const T * ref, size_t n_elem,
                 const char * label, size_t max_print = 8) {
    size_t mismatches = 0;
    size_t first = (size_t)-1;
    for (size_t i = 0; i < n_elem; ++i) {
        if (cxx[i] != ref[i]) {
            if (first == (size_t)-1) first = i;
            ++mismatches;
        }
    }
    if (mismatches == 0) {
        std::printf("  %-10s OK (%zu elements matched)\n", label, n_elem);
        return true;
    }
    std::printf("  %-10s MISMATCH: %zu / %zu elements differ\n",
                label, mismatches, n_elem);
    std::printf("    first mismatch at index %zu: cxx=", first);
    if constexpr (std::is_same_v<T, uint32_t>) {
        std::printf("0x%08x ref=0x%08x\n", cxx[first], ref[first]);
    } else {
        std::printf("0x%04x ref=0x%04x\n", (unsigned)cxx[first], (unsigned)ref[first]);
    }
    size_t shown = 0;
    for (size_t i = first; i < n_elem && shown < max_print; ++i) {
        if (cxx[i] != ref[i]) {
            std::printf("    [%zu] cxx=", i);
            if constexpr (std::is_same_v<T, uint32_t>) {
                std::printf("0x%08x ref=0x%08x\n", cxx[i], ref[i]);
            } else {
                std::printf("0x%04x ref=0x%04x\n", (unsigned)cxx[i], (unsigned)ref[i]);
            }
            ++shown;
        }
    }
    return false;
}

} // anonymous


int main(int argc, char ** argv) {
    if (argc != 4) {
        std::fprintf(stderr,
            "usage: %s <path.gguf> <tensor_name> <ref_dump.bin>\n", argv[0]);
        return 1;
    }
    const std::string gguf_path = argv[1];
    const std::string tensor    = argv[2];
    const std::string ref_path  = argv[3];

    // --- Load GGUF metadata via our reader ---
    gguf_init_params p{ /*no_alloc=*/true, /*ctx=*/nullptr };
    gguf_context * ctx = gguf_init_from_file(gguf_path.c_str(), p);
    if (ctx == nullptr) {
        std::fprintf(stderr, "failed to open %s\n", gguf_path.c_str());
        return 1;
    }
    auto reader_opt = StreamReader::from_gguf(ctx, gguf_path.c_str());
    if (!reader_opt) {
        std::fprintf(stderr, "no streamllm metadata in %s\n", gguf_path.c_str());
        gguf_free(ctx);
        return 1;
    }
    const TensorLayout * layout = reader_opt->layout(tensor);
    if (layout == nullptr) {
        std::fprintf(stderr, "tensor '%s' not in managed set\n", tensor.c_str());
        gguf_free(ctx);
        return 1;
    }
    const uint32_t group_size = reader_opt->global().group_size;

    // --- Read tensor bytes from disk (pread at absolute offset) ---
    auto tensor_bytes = pread_range(gguf_path,
                                    layout->tensor_offset,
                                    layout->tensor_nbytes);

    // --- Run the C++ layout builder ---
    UpstreamLayoutHost cxx;
    try {
        cxx = build_upstream_layout_host(*layout, tensor_bytes.data(), group_size);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "build failed: %s\n", e.what());
        gguf_free(ctx);
        return 1;
    }

    std::printf("C++ layout:\n");
    std::printf("  n=%d  padded_m=%d  P=%d  K_groups=%d  group_size=%d\n",
                cxx.n, cxx.padded_m, cxx.n_chunks, cxx.K_groups, cxx.group_size);
    std::printf("  chunks  %d × %zu bytes  (signs %zu + α %zu)\n",
                cxx.n_chunks, cxx.bytes_per_chunk,
                cxx.qw_bytes_per_chunk, cxx.alpha_bytes_per_chunk);
    std::printf("  q_bias        %zu × fp16 (%zu bytes)\n",
                cxx.q_bias_n_elem, cxx.q_bias.size() * sizeof(uint16_t));

    // --- Read the Python reference dump ---
    auto ref = read_file(ref_path);
    if (ref.size() < sizeof(RefHeader)) {
        std::fprintf(stderr, "ref file too small\n");
        gguf_free(ctx);
        return 1;
    }
    RefHeader rh{};
    std::memcpy(&rh, ref.data(), sizeof(rh));
    if (rh.magic != 0x4C505553u || rh.version != 1u) {
        std::fprintf(stderr, "ref file bad magic/version\n");
        gguf_free(ctx);
        return 1;
    }
    if (rh.n != cxx.n || rh.n_chunks != cxx.n_chunks || rh.padded_m != cxx.padded_m ||
        rh.n_groups_per_row != cxx.K_groups || rh.group_size != cxx.group_size) {
        std::fprintf(stderr,
            "ref header mismatch: n=%d/%d P=%d/%d padded_m=%d/%d Kg=%d/%d gs=%d/%d\n",
            rh.n, cxx.n, rh.n_chunks, cxx.n_chunks, rh.padded_m, cxx.padded_m,
            rh.n_groups_per_row, cxx.K_groups, rh.group_size, cxx.group_size);
        gguf_free(ctx);
        return 1;
    }

    size_t off = sizeof(RefHeader);
    // Python reference emits the monolithic [K/32, P, n] /
    // [K_groups, P, n] ordering. Our host layout packs each plane's
    // signs + α into one chunk; rebuild a monolith view here for the
    // byte-for-byte comparison.
    const int K_over_32 = cxx.padded_m / 32;
    std::vector<uint32_t> qw_mono((size_t)K_over_32 * cxx.n_chunks * cxx.n);
    std::vector<uint16_t> a_mono((size_t)cxx.K_groups * cxx.n_chunks * cxx.n);
    for (int p = 0; p < cxx.n_chunks; ++p) {
        const uint8_t * chunk     = cxx.chunks[p].data();
        const uint32_t * qw_plane = reinterpret_cast<const uint32_t *>(chunk);
        const uint16_t * a_plane  = reinterpret_cast<const uint16_t *>(
            chunk + cxx.qw_bytes_per_chunk);
        for (int kt = 0; kt < K_over_32; ++kt) {
            std::memcpy(
                &qw_mono[((size_t)kt * cxx.n_chunks + p) * cxx.n],
                &qw_plane[(size_t)kt * cxx.n],
                cxx.n * sizeof(uint32_t));
        }
        for (int kg = 0; kg < cxx.K_groups; ++kg) {
            std::memcpy(
                &a_mono[((size_t)kg * cxx.n_chunks + p) * cxx.n],
                &a_plane[(size_t)kg * cxx.n],
                cxx.n * sizeof(uint16_t));
        }
    }

    const size_t q_weight_bytes = qw_mono.size() * sizeof(uint32_t);
    const size_t alpha_bytes    = a_mono.size()  * sizeof(uint16_t);
    const size_t q_bias_bytes   = cxx.q_bias.size() * sizeof(uint16_t);
    const size_t want = sizeof(RefHeader) + q_weight_bytes + alpha_bytes + q_bias_bytes;
    if (ref.size() != want) {
        std::fprintf(stderr,
            "ref file size mismatch: got %zu, want %zu\n", ref.size(), want);
        gguf_free(ctx);
        return 1;
    }

    const uint32_t * ref_q_weight = reinterpret_cast<const uint32_t *>(ref.data() + off);
    off += q_weight_bytes;
    const uint16_t * ref_alpha = reinterpret_cast<const uint16_t *>(ref.data() + off);
    off += alpha_bytes;
    const uint16_t * ref_q_bias = reinterpret_cast<const uint16_t *>(ref.data() + off);

    std::printf("\nByte-for-byte parity check:\n");
    bool ok = true;
    ok &= compare_buf(qw_mono.data(), ref_q_weight,
                      qw_mono.size(), "q_weight");
    ok &= compare_buf(a_mono.data(),  ref_alpha,
                      a_mono.size(),  "alpha");
    ok &= compare_buf(cxx.q_bias.data(), ref_q_bias,
                      cxx.q_bias_n_elem, "q_bias");

    gguf_free(ctx);
    if (ok) {
        std::printf("\nPASS.\n");
        return 0;
    }
    std::printf("\nFAIL.\n");
    return 1;
}
