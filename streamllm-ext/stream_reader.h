// streamllm-ext — GGUF metadata reader for the streamllm.* KV block.
//
// The parent StreamLLM encoder writes a standard GGUF whose managed
// weights appear twice:
//
//   <canonical>                 F16 placeholder, correct 2-D shape
//                               (zeroed; satisfies stock llama.cpp's
//                                shape/type checks)
//   streamllm.bytes.<canonical> I8 byte blob holding the real
//                               CompressedTensor (fixed_meta + chunks)
//
// This reader treats the canonical name as the key (that's what the
// graph hook sees on ggml_tensor structs), but resolves byte offsets
// against the sibling ``streamllm.bytes.<name>`` tensor.
//
// No ownership of the gguf_context is taken — callers free it.
//
// See the parent repo's streamllm/runtime/format.py for the exact KV
// schema this reads.

#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

struct gguf_context;

namespace streamllm_ext {

// Per-managed-tensor layout as stored under
// ``streamllm.tensor.<name>.*`` keys.
struct TensorLayout {
    std::string name;                  // GGUF tensor name (e.g. "blk.0.attn_q.weight")
    std::array<int64_t, 2> shape;      // (n_out, m_in) — original W shape
    int64_t tensor_offset;             // byte offset of this tensor's data in the GGUF
    int64_t tensor_nbytes;             // total bytes of this tensor's data
    uint32_t fixed_bytes;              // prefix that is fixed_meta (β, header)
    std::vector<uint32_t> chunk_bytes; // per-plane chunk sizes, in order
    uint32_t d1;                       // n * n_groups_per_row
    uint32_t n_groups_per_row;         // m / group_size (ceil)
    uint32_t padded_m;                 // n_groups_per_row * group_size

    // Derived helper: absolute file offset of chunk ``cid``.
    int64_t chunk_offset(size_t cid) const {
        int64_t off = tensor_offset + fixed_bytes;
        for (size_t i = 0; i < cid && i < chunk_bytes.size(); ++i) off += chunk_bytes[i];
        return off;
    }

    // Sanity: fixed_bytes + sum(chunk_bytes) must equal tensor_nbytes.
    bool byte_accounting_ok() const {
        uint64_t total = fixed_bytes;
        for (auto b : chunk_bytes) total += b;
        return static_cast<int64_t>(total) == tensor_nbytes;
    }
};

// Global ``streamllm.*`` knobs (one value per GGUF file).
struct GlobalMeta {
    uint32_t version;
    std::string encoder;
    uint32_t group_size;
    uint32_t base_precision;
    uint32_t target_precision;
};

// Parse streamllm.* metadata from an already-loaded gguf_context. The
// context is typically obtained via llama.cpp's model loader or a
// direct ``gguf_init_from_file`` call — ownership stays with the caller.
// Returns std::nullopt if the file has no streamllm metadata (i.e.
// the file is a stock upstream GGUF, not a StreamLLM artifact).
class StreamReader {
public:
    // Parse all streamllm.* metadata from ``ctx``. Returns a valid reader
    // iff the file has ``streamllm.version`` set.
    static std::optional<StreamReader> from_gguf(const gguf_context * ctx,
                                                 const char * gguf_path);

    const GlobalMeta & global() const { return global_; }

    // Ordered list of GGUF tensor names that are StreamLLM-managed.
    const std::vector<std::string> & managed_tensor_names() const { return managed_; }

    // Layout for a managed tensor, or nullptr if the name isn't managed.
    const TensorLayout * layout(const std::string & name) const {
        auto it = layouts_.find(name);
        return it == layouts_.end() ? nullptr : &it->second;
    }

    // The GGUF file path — useful for a runtime pread / mmap of chunk
    // bytes by their absolute offsets without going through the already-
    // loaded gguf_context.
    const std::string & gguf_path() const { return gguf_path_; }

private:
    StreamReader() = default;

    GlobalMeta global_{};
    std::vector<std::string> managed_;
    std::unordered_map<std::string, TensorLayout> layouts_;
    std::string gguf_path_;
};

// Pretty-print the parsed metadata to stdout — convenient for the
// M2 CLI tool ``streamllm-gguf-dump``.
void dump(const StreamReader & r);

} // namespace streamllm_ext
