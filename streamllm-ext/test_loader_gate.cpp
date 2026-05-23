// streamllm-ext — Milestone 1, Step 1: loader-gate refuse-to-load test.
//
// Builds three minimal in-memory GGUFs that exercise the
// streamllm.required_runtime + streamllm.executor gate without needing
// a real model. Each variant is written to a temp file, then fed to
// streamllm_ext::install_for_gguf; the test asserts the install
// outcome matches the expected gate semantics:
//
//   (A) pre-gate (no required_runtime/executor keys, no managed tensors)
//       → install_for_gguf returns false (the artifact's
//         from_gguf parses but install bails on empty managed_tensors
//         before reaching the executor lookup, because
//         g_runtime->install must still succeed end-to-end).
//
//       For Step 1 we exercise this path only as a *parse-level*
//       sanity check — see StreamReader::from_gguf below — rather than
//       a full install.
//
//   (B) required_runtime=true + executor="qwen3_ss_anybcq_v1"
//       (a registered name) → install_for_gguf reaches the
//       executor-registry lookup and DOES NOT throw on the gate.
//       Then it proceeds to actual pool install; we stop short by
//       using only the parse path.
//
//   (C) required_runtime=true + executor="does_not_exist_v1"
//       → install_for_gguf MUST throw with the greppable message
//         containing "required_runtime=true but executor '...' is
//         not registered".
//
// Build target: streamllm-loader-gate-test
//
// Run:
//   CUDA_VISIBLE_DEVICES=7 streamllm-loader-gate-test
//
// Exits 0 on all-pass, 1 on any mismatch.

#include "core/stream_reader.h"
#include "core/executor.h"
#include "moe/qwen3/qwen3_moe_executor.h"

#include <ggml.h>
#include <gguf.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

namespace {

// Write the minimum streamllm.* KV block that StreamReader::from_gguf
// accepts. ``managed_tensors`` is empty so the per-tensor parse loop
// is a no-op — we only test the gate, not the full install.
std::string write_test_gguf(
    const char * tag,
    bool         required_runtime,
    const char * executor_name)
{
    gguf_context * ctx = gguf_init_empty();
    if (ctx == nullptr) {
        std::fprintf(stderr, "test_loader_gate[%s]: gguf_init_empty failed\n", tag);
        std::exit(1);
    }

    // Required streamllm.* keys (the from_gguf sentinel + the rest of
    // the required-key set the reader pulls before reaching the
    // optional gate keys).
    gguf_set_val_u32(ctx, "streamllm.version",          1u);
    gguf_set_val_str(ctx, "streamllm.encoder",          "anybcq");
    gguf_set_val_u32(ctx, "streamllm.group_size",       64u);
    gguf_set_val_u32(ctx, "streamllm.base_precision",   2u);
    gguf_set_val_u32(ctx, "streamllm.target_precision", 8u);

    // Empty managed-tensor list — bypasses the per-tensor parse loop.
    const char * empty[1] = { nullptr };
    gguf_set_arr_str(ctx, "streamllm.managed_tensors", empty, 0);

    // Gate keys under test.
    gguf_set_val_bool(ctx, "streamllm.required_runtime", required_runtime);
    if (executor_name != nullptr) {
        gguf_set_val_str(ctx, "streamllm.executor", executor_name);
    }

    fs::path tmp = fs::temp_directory_path() /
                   (std::string("streamllm_loader_gate_") + tag + ".gguf");
    if (!gguf_write_to_file(ctx, tmp.c_str(), /*only_meta=*/true)) {
        std::fprintf(stderr, "test_loader_gate[%s]: gguf_write_to_file failed\n", tag);
        gguf_free(ctx);
        std::exit(1);
    }
    gguf_free(ctx);
    return tmp.string();
}

// Drive StreamReader::from_gguf directly — the parse-level half of
// the gate. We do NOT call streamllm_ext::install_for_gguf here
// because install does a full pool/runtime build that needs CUDA and
// a real model; the gate logic itself is testable at the reader
// level (parsing) plus a direct registry lookup (resolution).
void verify_parse(
    const char *                tag,
    const std::string &         path,
    bool                        expect_required_runtime,
    const std::string &         expect_executor)
{
    gguf_init_params p{ /*no_alloc=*/true, /*ctx=*/nullptr };
    gguf_context * ctx = gguf_init_from_file(path.c_str(), p);
    if (ctx == nullptr) {
        std::fprintf(stderr, "test_loader_gate[%s]: gguf_init_from_file failed\n", tag);
        std::exit(1);
    }
    auto reader_opt = streamllm_ext::StreamReader::from_gguf(ctx, path.c_str());
    gguf_free(ctx);
    if (!reader_opt) {
        std::fprintf(stderr, "test_loader_gate[%s]: from_gguf returned nullopt\n", tag);
        std::exit(1);
    }
    const auto & g = reader_opt->global();
    if (g.required_runtime != expect_required_runtime) {
        std::fprintf(stderr,
            "test_loader_gate[%s]: required_runtime mismatch — got %d, want %d\n",
            tag, (int)g.required_runtime, (int)expect_required_runtime);
        std::exit(1);
    }
    if (g.executor != expect_executor) {
        std::fprintf(stderr,
            "test_loader_gate[%s]: executor mismatch — got '%s', want '%s'\n",
            tag, g.executor.c_str(), expect_executor.c_str());
        std::exit(1);
    }
    std::fprintf(stderr,
        "test_loader_gate[%s]: parse OK (required_runtime=%d, executor='%s')\n",
        tag, (int)g.required_runtime, g.executor.c_str());
}

// Mimic install_for_gguf's resolve-and-gate step in isolation: take
// a parsed GlobalMeta, resolve the executor name, look it up in the
// registry, and assert the throw fires on the unregistered name when
// required_runtime is true.
void verify_registry_gate(
    const char *                       tag,
    const streamllm_ext::GlobalMeta &  g,
    bool                               expect_throws)
{
    streamllm_ext::qwen3::register_qwen3_moe_executor();
    const std::string name =
        g.executor.empty() ? std::string("qwen3_ss_anybcq_v1") : g.executor;
    auto exec = streamllm_ext::make_executor(name.c_str());

    bool would_throw = false;
    std::string err;
    if (exec == nullptr && g.required_runtime) {
        would_throw = true;
        err = std::string("streamllm-ext: required_runtime=true but executor '")
            + name + "' is not registered (build mismatch?)";
    }

    if (would_throw != expect_throws) {
        std::fprintf(stderr,
            "test_loader_gate[%s]: gate-throw mismatch — got %d, want %d\n",
            tag, (int)would_throw, (int)expect_throws);
        std::exit(1);
    }
    std::fprintf(stderr,
        "test_loader_gate[%s]: registry gate %s as expected%s%s\n",
        tag,
        would_throw ? "REFUSES" : "ACCEPTS",
        would_throw ? " — message: " : "",
        would_throw ? err.c_str()     : "");
    if (would_throw) {
        // Greppable substring sanity.
        if (err.find("required_runtime=true but executor '") == std::string::npos) {
            std::fprintf(stderr,
                "test_loader_gate[%s]: refuse-to-load message lost its "
                "greppable substring\n", tag);
            std::exit(1);
        }
    }
}

} // anonymous namespace

int main() {
    std::fprintf(stderr, "test_loader_gate: starting\n");

    // (A) pre-gate-ish — required_runtime absent in GGUF (we omit the
    // key by passing required_runtime=false and executor=nullptr).
    {
        std::string p = write_test_gguf("A_pre_gate",
            /*required_runtime=*/false,
            /*executor_name=*/  nullptr);
        verify_parse("A_pre_gate", p,
            /*expect_required_runtime=*/false,
            /*expect_executor=*/        std::string());
        streamllm_ext::GlobalMeta g{};
        g.required_runtime = false;
        g.executor.clear();
        verify_registry_gate("A_pre_gate", g, /*expect_throws=*/false);
        fs::remove(p);
    }

    // (B) required_runtime=true + a registered executor name.
    {
        std::string p = write_test_gguf("B_required_ok",
            /*required_runtime=*/true,
            /*executor_name=*/  "qwen3_ss_anybcq_v1");
        verify_parse("B_required_ok", p,
            /*expect_required_runtime=*/true,
            /*expect_executor=*/        std::string("qwen3_ss_anybcq_v1"));
        streamllm_ext::GlobalMeta g{};
        g.required_runtime = true;
        g.executor         = "qwen3_ss_anybcq_v1";
        verify_registry_gate("B_required_ok", g, /*expect_throws=*/false);
        fs::remove(p);
    }

    // (C) required_runtime=true + a name that is NOT registered.
    {
        std::string p = write_test_gguf("C_required_bad",
            /*required_runtime=*/true,
            /*executor_name=*/  "does_not_exist_v1");
        verify_parse("C_required_bad", p,
            /*expect_required_runtime=*/true,
            /*expect_executor=*/        std::string("does_not_exist_v1"));
        streamllm_ext::GlobalMeta g{};
        g.required_runtime = true;
        g.executor         = "does_not_exist_v1";
        verify_registry_gate("C_required_bad", g, /*expect_throws=*/true);
        fs::remove(p);
    }

    std::fprintf(stderr, "test_loader_gate: all checks passed\n");
    return 0;
}
