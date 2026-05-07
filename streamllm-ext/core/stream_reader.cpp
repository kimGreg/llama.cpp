// streamllm-ext — implementation of the GGUF metadata reader.
//
// Reads the streamllm.* KV block + managed-tensor byte layout from a
// gguf_context. No CUDA dependencies here; this can run CPU-only and is
// safe to call at model-load time before the device is ready.

#include "stream_reader.h"

#include <gguf.h>

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

namespace streamllm_ext {
namespace {

// Helpers around the C gguf API. Each of these aborts via throw on
// missing keys / type mismatches so callers can catch a clean failure.

int64_t require_key(const gguf_context * ctx, const char * key) {
    int64_t id = gguf_find_key(ctx, key);
    if (id < 0) {
        throw std::runtime_error(std::string("streamllm-ext: missing required GGUF key: ") + key);
    }
    return id;
}

int64_t optional_key(const gguf_context * ctx, const char * key) {
    return gguf_find_key(ctx, key);
}

uint32_t get_u32(const gguf_context * ctx, const char * key) {
    int64_t id = require_key(ctx, key);
    return gguf_get_val_u32(ctx, id);
}

std::string get_str(const gguf_context * ctx, const char * key) {
    int64_t id = require_key(ctx, key);
    const char * s = gguf_get_val_str(ctx, id);
    return s ? std::string(s) : std::string();
}

std::vector<std::string> get_arr_str(const gguf_context * ctx, const char * key) {
    int64_t id = require_key(ctx, key);
    size_t n = gguf_get_arr_n(ctx, id);
    std::vector<std::string> out;
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        const char * s = gguf_get_arr_str(ctx, id, i);
        out.emplace_back(s ? s : "");
    }
    return out;
}

// Uniform-array helpers. The gguf-py writer emits streamllm.tensor.*.shape
// as an INT64 array (from `add_array([n, m])`) and chunk_bytes as an INT32
// array; probe the element type at runtime and convert.
std::vector<int64_t> get_arr_i64(const gguf_context * ctx, const char * key) {
    int64_t id = require_key(ctx, key);
    size_t n = gguf_get_arr_n(ctx, id);
    gguf_type t = gguf_get_arr_type(ctx, id);
    const void * data = gguf_get_arr_data(ctx, id);
    std::vector<int64_t> out(n);
    switch (t) {
        case GGUF_TYPE_INT64:
            std::memcpy(out.data(), data, n * sizeof(int64_t));
            break;
        case GGUF_TYPE_UINT64:
            for (size_t i = 0; i < n; ++i) out[i] = (int64_t)((const uint64_t *)data)[i];
            break;
        case GGUF_TYPE_INT32:
            for (size_t i = 0; i < n; ++i) out[i] = (int64_t)((const int32_t *)data)[i];
            break;
        case GGUF_TYPE_UINT32:
            for (size_t i = 0; i < n; ++i) out[i] = (int64_t)((const uint32_t *)data)[i];
            break;
        default:
            throw std::runtime_error(
                std::string("streamllm-ext: array '") + key +
                "' has unsupported element type " + std::to_string((int)t));
    }
    return out;
}

std::vector<uint32_t> get_arr_u32(const gguf_context * ctx, const char * key) {
    auto v64 = get_arr_i64(ctx, key);
    std::vector<uint32_t> out;
    out.reserve(v64.size());
    for (auto x : v64) out.push_back((uint32_t)x);
    return out;
}

} // anonymous namespace


std::optional<StreamReader> StreamReader::from_gguf(const gguf_context * ctx,
                                                    const char * gguf_path) {
    if (ctx == nullptr) return std::nullopt;
    // Presence of streamllm.version is our sentinel. If it's missing,
    // this is a stock GGUF and we bow out cleanly.
    if (gguf_find_key(ctx, "streamllm.version") < 0) {
        return std::nullopt;
    }

    StreamReader r;
    r.gguf_path_ = gguf_path ? std::string(gguf_path) : std::string();
    r.global_.version          = get_u32(ctx, "streamllm.version");
    r.global_.encoder          = get_str(ctx, "streamllm.encoder");
    r.global_.group_size       = get_u32(ctx, "streamllm.group_size");
    r.global_.base_precision   = get_u32(ctx, "streamllm.base_precision");
    r.global_.target_precision = get_u32(ctx, "streamllm.target_precision");
    r.managed_                 = get_arr_str(ctx, "streamllm.managed_tensors");

    r.layouts_.reserve(r.managed_.size());
    for (const auto & name : r.managed_) {
        TensorLayout layout{};
        layout.name = name;

        const std::string prefix = "streamllm.tensor." + name + ".";
        // MoE canonical stacked tensors (`blk.<N>.ffn_<kind>_exps.weight`)
        // are listed in managed_tensors so the model-loader skips their
        // disk-copy and seeds their data pointer, but they don't have
        // chunk metadata at the canonical level — the chunks are stored
        // per-expert under synthetic wids `<canonical>:e<X>`. When the
        // shape key is missing, this is such a placeholder; record it
        // and skip chunk-layout parsing.
        if (gguf_find_key(ctx, (prefix + "shape").c_str()) < 0) {
            r.placeholder_only_.insert(name);
            continue;
        }
        auto shape_vec        = get_arr_i64(ctx, (prefix + "shape").c_str());
        auto chunk_bytes_vec  = get_arr_u32(ctx, (prefix + "chunk_bytes").c_str());
        layout.fixed_bytes      = get_u32(ctx, (prefix + "fixed_bytes").c_str());
        layout.d1               = get_u32(ctx, (prefix + "d1").c_str());
        layout.n_groups_per_row = get_u32(ctx, (prefix + "n_groups_per_row").c_str());
        layout.padded_m         = get_u32(ctx, (prefix + "padded_m").c_str());

        if (shape_vec.size() != 2) {
            throw std::runtime_error(
                "streamllm-ext: tensor " + name +
                " has non-2D shape (" + std::to_string(shape_vec.size()) + ")");
        }
        layout.shape = { shape_vec[0], shape_vec[1] };
        layout.chunk_bytes = std::move(chunk_bytes_vec);

        // Byte payload lives at ``streamllm.bytes.<canonical_name>`` — a
        // sibling I8 tensor that stock llama.cpp ignores. The canonical
        // name itself points at an F16 placeholder used only to satisfy
        // llama.cpp's model-load shape/type checks.
        const std::string bytes_name = "streamllm.bytes." + name;
        int64_t tid = gguf_find_tensor(ctx, bytes_name.c_str());
        if (tid < 0) {
            throw std::runtime_error(
                "streamllm-ext: byte-blob tensor '" + bytes_name +
                "' not found in GGUF (did the encoder emit the split layout?)");
        }
        layout.tensor_offset = (int64_t)(gguf_get_data_offset(ctx) + gguf_get_tensor_offset(ctx, tid));
        layout.tensor_nbytes = (int64_t)gguf_get_tensor_size(ctx, tid);

        if (!layout.byte_accounting_ok()) {
            uint64_t sum_chunks = 0;
            for (auto b : layout.chunk_bytes) sum_chunks += b;
            throw std::runtime_error(
                "streamllm-ext: tensor '" + name + "' byte accounting mismatch: "
                "fixed(" + std::to_string(layout.fixed_bytes) + ") + "
                "sum(chunks)(" + std::to_string(sum_chunks) + ") != "
                "tensor_nbytes(" + std::to_string(layout.tensor_nbytes) + ")");
        }

        r.layouts_.emplace(name, std::move(layout));
    }

    return r;
}


void dump(const StreamReader & r) {
    const auto & g = r.global();
    std::printf("== streamllm global metadata ==\n");
    std::printf("  version           = %u\n", g.version);
    std::printf("  encoder           = %s\n", g.encoder.c_str());
    std::printf("  group_size        = %u\n", g.group_size);
    std::printf("  base_precision    = %u\n", g.base_precision);
    std::printf("  target_precision  = %u\n", g.target_precision);
    std::printf("  managed_tensors   = %zu names\n", r.managed_tensor_names().size());
    if (!r.gguf_path().empty()) {
        std::printf("  gguf_path         = %s\n", r.gguf_path().c_str());
    }
    std::printf("\n== first 3 managed tensors ==\n");
    const auto & names = r.managed_tensor_names();
    size_t limit = names.size() < 3 ? names.size() : 3;
    for (size_t i = 0; i < limit; ++i) {
        const auto * L = r.layout(names[i]);
        if (L == nullptr) continue;
        std::printf("  %s\n", L->name.c_str());
        std::printf("    shape       = [%lld, %lld]\n",
                    (long long)L->shape[0], (long long)L->shape[1]);
        std::printf("    fixed_bytes = %u\n", L->fixed_bytes);
        std::printf("    chunk_bytes = [");
        for (size_t j = 0; j < L->chunk_bytes.size(); ++j) {
            std::printf("%u%s", L->chunk_bytes[j],
                        j + 1 == L->chunk_bytes.size() ? "" : ", ");
        }
        std::printf("]\n");
        std::printf("    d1          = %u\n", L->d1);
        std::printf("    ng/row      = %u\n", L->n_groups_per_row);
        std::printf("    padded_m    = %u\n", L->padded_m);
        std::printf("    tensor_offset = %lld\n", (long long)L->tensor_offset);
        std::printf("    tensor_nbytes = %lld\n", (long long)L->tensor_nbytes);
    }
}

} // namespace streamllm_ext
