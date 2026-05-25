// DPMoE — GGUF metadata reader for the dp_moe.* KV block.
//
// The parent DPMoE encoder writes a standard GGUF whose managed
// weights appear twice:
//
//   <canonical>                 F16 placeholder, correct 2-D shape
//                               (zeroed; satisfies stock llama.cpp's
//                                shape/type checks)
//   dp_moe.bytes.<canonical> I8 byte blob holding the real
//                               CompressedTensor (fixed_meta + chunks)
//
// This reader treats the canonical name as the key (that's what the
// graph hook sees on ggml_tensor structs), but resolves byte offsets
// against the sibling ``dp_moe.bytes.<name>`` tensor.
//
// No ownership of the gguf_context is taken — callers free it.
//
// See the parent repo's dp_moe/runtime/format.py for the exact KV
// schema this reads.

#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct gguf_context;

namespace dp_moe_ext {

// Per-managed-tensor layout as stored under
// ``dp_moe.tensor.<name>.*`` keys.
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
        return static_cast<int64_t>(total) == tensor_nbytes ||
               static_cast<int64_t>(fixed_bytes) == tensor_nbytes;
    }

    bool fixed_only_payload() const {
        return static_cast<int64_t>(fixed_bytes) == tensor_nbytes;
    }
};

// Global ``dp_moe.*`` knobs (one value per GGUF file).
struct GlobalMeta {
    uint32_t version;
    std::string encoder;
    uint32_t group_size;
    uint32_t base_precision;
    uint32_t target_precision;

    // Mode A loader gate (Milestone 1, Step 1).  When ``required_runtime``
    // is true, ``install_for_gguf`` MUST resolve ``executor`` against the
    // ModelExecutor registry and MUST refuse-to-load if the named executor
    // is not present — stock GGUFs and pre-gate dp_moe artifacts (no
    // such keys) default to ``false`` / ``""`` and keep the legacy
    // behaviour (hardcoded executor name with a soft fallback).
    bool        required_runtime = false;
    std::string executor;            // versioned, e.g. "qwen3_ss_anybcq_v1"
};

struct BundleChunkSlice {
    std::string wid;
    uint32_t cid = 0;
    int64_t offset = 0;
    int64_t size = 0;
    std::string bundle;
    bool kernel_ready = false;
};

// Parse dp_moe.* metadata from an already-loaded gguf_context. The
// context is typically obtained via llama.cpp's model loader or a
// direct ``gguf_init_from_file`` call — ownership stays with the caller.
// Returns std::nullopt if the file has no dp_moe metadata (i.e.
// the file is a stock upstream GGUF, not a DPMoE artifact).
class StreamReader {
public:
    // Parse all dp_moe.* metadata from ``ctx``. Returns a valid reader
    // iff the file has ``dp_moe.version`` set.
    static std::optional<StreamReader> from_gguf(const gguf_context * ctx,
                                                 const char * gguf_path);

    const GlobalMeta & global() const { return global_; }

    // Ordered list of GGUF tensor names that are DPMoE-managed.
    // Includes both *chunked* tensors (regular dense managed tensors,
    // and per-expert MoE chunks under synthetic wids `<canonical>:e<X>`)
    // and *placeholder-only* tensors (canonical MoE stacked names whose
    // disk-copy must be skipped but which carry no chunk metadata at
    // this name level). Use ``is_placeholder_only(name)`` to distinguish
    // — schedulers / install paths typically want to skip those.
    const std::vector<std::string> & managed_tensor_names() const { return managed_; }

    // Subset of ``managed_tensor_names()`` excluding placeholder-only
    // entries. Schedulers iterate this for install-time chunk uploads
    // so they don't trip on canonical MoE stacked names.
    std::vector<std::string> chunked_tensor_names() const {
        std::vector<std::string> out;
        out.reserve(managed_.size());
        for (const auto & n : managed_) {
            if (!is_placeholder_only(n)) out.push_back(n);
        }
        return out;
    }

    // Layout for a managed tensor, or nullptr if the name isn't managed.
    const TensorLayout * layout(const std::string & name) const {
        auto it = layouts_.find(name);
        return it == layouts_.end() ? nullptr : &it->second;
    }

    const BundleChunkSlice * bundle_slice(const std::string & wid,
                                          uint32_t cid) const {
        auto it = bundle_slices_.find(wid + "#" + std::to_string(cid));
        return it == bundle_slices_.end() ? nullptr : &it->second;
    }

    size_t bundle_slice_count() const { return bundle_slices_.size(); }

    // True for managed tensors that exist only as placeholders in the
    // GGUF (loader-skip + seed pointer). These have no chunk metadata
    // because their bytes are accessed via per-expert sub-wids — the
    // canonical MoE stacked tensors (`blk.<N>.ffn_<kind>_exps.weight`)
    // fit this pattern. Schedulers / install paths should skip these.
    bool is_placeholder_only(const std::string & name) const {
        return placeholder_only_.count(name) != 0;
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
    std::unordered_map<std::string, BundleChunkSlice> bundle_slices_;
    std::unordered_set<std::string> placeholder_only_;
    std::string gguf_path_;
};

// Pretty-print the parsed metadata to stdout — used by the
// dp_moe-gguf-dump CLI.
void dump(const StreamReader & r);

} // namespace dp_moe_ext
