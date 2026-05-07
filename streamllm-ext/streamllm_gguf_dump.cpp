// streamllm-gguf-dump — read a GGUF and print its streamllm.* metadata.
//
// Usage:
//   streamllm-gguf-dump path/to/qwen3-1.7b-anybcq-P8.gguf
//
// Exits 0 if streamllm metadata is present and byte accounting checks
// pass; 2 if the file has no streamllm metadata (i.e. it's a stock
// GGUF); 1 on any other error.

#include "stream_reader.h"

#include <gguf.h>

#include <cstdio>
#include <exception>

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <path-to.gguf>\n", argv[0]);
        return 1;
    }
    const char * path = argv[1];
    gguf_init_params p{ /*no_alloc=*/true, /*ctx=*/nullptr };
    gguf_context * ctx = gguf_init_from_file(path, p);
    if (ctx == nullptr) {
        std::fprintf(stderr, "failed to open GGUF: %s\n", path);
        return 1;
    }

    int rc = 0;
    try {
        auto reader = streamllm_ext::StreamReader::from_gguf(ctx, path);
        if (!reader.has_value()) {
            std::fprintf(stderr,
                "no streamllm metadata in %s (stock GGUF?)\n", path);
            rc = 2;
        } else {
            streamllm_ext::dump(*reader);
            // byte_accounting_ok was validated inside from_gguf; reassert
            // across all managed tensors for an explicit signal here.
            // Skip placeholder-only managed names (MoE canonical
            // stacked tensors) — their disk-skip is handled by the
            // loader, no chunk metadata to validate here.
            const auto chunked = reader->chunked_tensor_names();
            for (const auto & n : chunked) {
                const auto * L = reader->layout(n);
                if (!L || !L->byte_accounting_ok()) {
                    std::fprintf(stderr,
                        "byte accounting failed for tensor '%s'\n", n.c_str());
                    rc = 1;
                    break;
                }
            }
            if (rc == 0) {
                std::printf("\nall %zu chunked managed tensors have consistent byte accounting "
                            "(plus %zu placeholder-only canonical names).\n",
                            chunked.size(),
                            reader->managed_tensor_names().size() - chunked.size());
            }
        }
    } catch (const std::exception & e) {
        std::fprintf(stderr, "streamllm-ext error: %s\n", e.what());
        rc = 1;
    }

    gguf_free(ctx);
    return rc;
}
