// streamllm-ext / qwen3 — runtime glue between llama.cpp's model-load
// path, ggml-cuda's hooks, and the Qwen3-MoE scheduler.  Owns the
// process-wide ``g_runtime`` singleton.  Lives in qwen3/ because every
// non-trivial responsibility — install-time MoE scratch sizing,
// ggml-cuda hook registration, score-policy live-dial entry points,
// per-op extern-C shims — funnels into model-specific behavior.

#include "qwen3_runtime_glue.h"
#include "runtime.h"
#include "runtime_diag.h"
#include "stream_reader.h"
#include "executor.h"
#include "anybcq_gemm.h"

#include "qwen3_moe_dispatch.h"
#include "qwen3_moe_scheduler.h"  // qwen3::scheduler_* free-fn shims
#include "qwen3_moe_executor.h"   // Qwen3MoEAnyBcqExecutor + registration

#include <ggml.h>
#include <gguf.h>
#include <ggml-cuda.h>

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

namespace streamllm_ext {

// Defined here, declared extern in core/runtime_hook_internal.h so the
// MoE dispatch impls in qwen3/ can read the active runtime under this
// mutex.
std::mutex                            g_runtime_mu;
std::unique_ptr<StreamllmRuntime>     g_runtime;
std::unique_ptr<ModelExecutor>        g_executor;

// Mode A loader gate (Milestone 1, Step 1) — remember whether the
// currently-installed runtime came from a GGUF with
// ``streamllm.required_runtime=true``.  A second install while a
// required-runtime model is live MUST hard-fail (see the
// executor-lifetime contract in docs/MODE_A_MILESTONE1.md Step 7
// design).  Defaults to false on cold start and after clear().
bool                                  g_required_runtime_was_true = false;

namespace {

// Default executor name preserved from pre-Step-1 installs.  Used when
// the GGUF does not carry ``streamllm.executor`` (pre-gate artifact)
// and required_runtime is false.  This is the soft-fallback name; the
// hard refuse-to-load path lives below.
constexpr const char * kDefaultExecutorName = "qwen3_moe_anybcq_v1";

// Resolve the executor name to look up in the registry.
//   GGUF carries streamllm.executor → use it.
//   Missing key                     → kDefaultExecutorName (legacy).
std::string resolve_executor_name(const GlobalMeta & g) {
    return g.executor.empty() ? std::string(kDefaultExecutorName)
                              : g.executor;
}

// Conservative pool sizing used by install_for_gguf. Mirrors
// test_runtime_full.cpp::estimate_pool_bytes — summed over every
// managed tensor with a 2× margin for allocator fragmentation.
size_t estimate_pool_bytes(const StreamReader & r) {
    size_t total = 0;
    for (const auto & name : r.managed_tensor_names()) {
        const auto * L = r.layout(name);
        if (L == nullptr) continue;
        int32_t M  = (int32_t)L->shape[0];
        int32_t K  = L->padded_m;
        int32_t P  = (int32_t)L->chunk_bytes.size();
        int32_t Kg = (int32_t)L->n_groups_per_row;
        total += (size_t)(K / 32) * P * M * 4;
        total += (size_t)Kg * P * M * 2;
        total += (size_t)Kg * M * 2;
    }
    return total * 2;
}

} // anonymous namespace


bool install_for_gguf(const char * gguf_path) {
    if (gguf_path == nullptr || gguf_path[0] == '\0') return false;

    gguf_init_params p{ /*no_alloc=*/true, /*ctx=*/nullptr };
    gguf_context * ctx = gguf_init_from_file(gguf_path, p);
    if (ctx == nullptr) return false;

    auto reader_opt = StreamReader::from_gguf(ctx, gguf_path);
    gguf_free(ctx);
    if (!reader_opt) {
        // No streamllm.* block — stock GGUF, leave default dispatch alone.
        return false;
    }

    const auto & reader = *reader_opt;

    std::lock_guard<std::mutex> lk(g_runtime_mu);

    if (g_runtime) {
        // Step 1 (Milestone 1) — refuse to silently replace a runtime
        // whose model declared required_runtime=true.  Any
        // llama_model::streamllm_executor pointers into the prior
        // runtime would become dangling.  See the executor-lifetime
        // contract in docs/MODE_A_MILESTONE1.md (Step 7 design).
        // Pre-gate installs (required_runtime=false) keep the legacy
        // silent-replace behaviour for backward compatibility.
        if (g_required_runtime_was_true) {
            throw std::runtime_error(
                "streamllm-ext: a required_runtime model is already loaded; "
                "free it before loading another StreamLLM model");
        }
        std::fprintf(stderr,
            "streamllm-ext: replacing previously-installed (non-required) "
            "runtime — prior model was not cleared explicitly\n");
        g_runtime.reset();
        g_executor.reset();
    }

    size_t cap = estimate_pool_bytes(reader);
    if (const char * s = getenv("STREAMLLM_VRAM_CAP_MB")) {
        long mb = std::atol(s);
        if (mb > 0) cap = (size_t)mb * 1024UL * 1024UL;
    }
    const char * sched_name = getenv("STREAMLLM_SCHEDULER");
    g_runtime = std::make_unique<StreamllmRuntime>(
        cap, /*device=*/0, /*copy_stream=*/true, sched_name);

    g_runtime->install(reader, std::string(gguf_path));
    cudaDeviceSynchronize();

    diag::install_open_trace();

    if (!moe_dispatch::size_scratch_for(reader)) {
        g_runtime.reset();
        throw std::runtime_error(
            "streamllm-ext: failed to size cast scratch");
    }

    // Mode A executor (SSOT §6.1.1, Milestone 1 Step 1 loader gate).
    // Register the Qwen3-MoE AnyBCQ executor (idempotent), then resolve
    // the executor name against the GGUF's streamllm.executor key (with
    // a soft-fallback default for pre-gate artifacts).  When
    // streamllm.required_runtime=true and the named executor is not in
    // the registry — e.g. binary mismatch, missing build config — refuse
    // to load with a greppable error rather than silently downgrade.
    qwen3::register_qwen3_moe_anybcq_executor();
    const std::string exec_name = resolve_executor_name(reader.global());
    g_executor = make_executor(exec_name.c_str());
    if (g_executor == nullptr) {
        if (reader.global().required_runtime) {
            // Bring-up escape hatch: STREAMLLM_REQUIRED_RUNTIME_IGNORE=1
            // downgrades the throw to a warning so operators can run a
            // mismatched binary against a required_runtime artifact for
            // diagnosis. The default behaviour (no env var) is the safe
            // refuse-to-load.
            if (getenv("STREAMLLM_REQUIRED_RUNTIME_IGNORE")) {
                std::fprintf(stderr,
                    "streamllm-ext: WARNING — required_runtime=true but "
                    "executor '%s' is not registered. Continuing because "
                    "STREAMLLM_REQUIRED_RUNTIME_IGNORE is set; managed "
                    "dispatch will fall back to legacy paths and may "
                    "produce wrong output.\n", exec_name.c_str());
            } else {
                g_runtime.reset();
                throw std::runtime_error(
                    "streamllm-ext: required_runtime=true but executor '"
                    + exec_name + "' is not registered (build mismatch?)");
            }
        } else {
            // Legacy soft-fail: pre-gate artifacts (no required_runtime
            // key) still expect the hardcoded default to resolve. If we
            // somehow got here with the default name unregistered,
            // surface a clear error so the operator knows the build is
            // broken.
            g_runtime.reset();
            throw std::runtime_error(
                "streamllm-ext: executor '" + exec_name + "' not registered");
        }
    }
    g_executor->bind_to_model(*g_runtime, reader, std::string(gguf_path));
    g_required_runtime_was_true = reader.global().required_runtime;

    std::fprintf(stderr,
        "streamllm-ext: runtime ready "
        "(scheduler=%s, executor=%s, pool %.1f / %.1f MB used)\n",
        g_runtime->scheduler().name(),
        g_executor->name(),
        (double)g_runtime->pool().used_bytes()     / 1024.0 / 1024.0,
        (double)g_runtime->pool().capacity_bytes() / 1024.0 / 1024.0);

    if (getenv("STREAMLLM_STATS_RESET_AFTER_INSTALL")) {
        g_runtime->pool().reset_stats();
    }

    ggml_cuda_set_mul_mat_hook((void *) &streamllm_try_cuda_mul_mat);
    ggml_cuda_set_fusion_skip_hook((void *) &streamllm_claims_tensor);
    ggml_cuda_set_mul_mat_id_hook((void *) &streamllm_try_cuda_mul_mat_id);
    ggml_cuda_set_topk_moe_hook((void *) &streamllm_topk_moe_observed);
    ggml_cuda_set_graph_compute_begin_hook(
        (void *) &streamllm_graph_compute_begin);
    ggml_cuda_set_graph_compute_end_hook(
        (void *) &streamllm_graph_compute_end);
    ggml_cuda_set_user_node_claims_hook(
        (void *) &streamllm_user_node_claims);
    return true;
}

extern "C" bool streamllm_user_node_claims(const struct ggml_tensor * node) {
    // Per-node claim predicate consulted by ggml-cuda before
    // deciding whether to capture the cgraph into a cuda-graph.
    // Streamllm is an eager-only framework — its LOAD walks need
    // host-side decision-making and full CUDA API access on every
    // dispatch.  When the scheduler claims any node in the cgraph,
    // ggml-cuda disables cuda-graph capture for that compute and
    // every node runs eager.  The streamllm dispatch is fast enough
    // (load-bound at tight cap, near-eager at full pin) that the
    // missing capture speedup doesn't matter for streamllm's use
    // case.
    if (node == nullptr) return false;
    std::lock_guard<std::mutex> lk(g_runtime_mu);
    if (!g_runtime) return false;
    return g_runtime->scheduler().claims_node(node);
}

extern "C" bool streamllm_claims_tensor(const struct ggml_tensor * w) {
    // Fusion-skip predicate. Delegate to the scheduler — it decides
    // which weight tensors must opt out of ggml-cuda's fused-subgraph
    // paths so the streamllm per-op dispatch can claim the underlying
    // mul_mat. Scheduler-agnostic.
    if (w == nullptr) return false;
    std::lock_guard<std::mutex> lk(g_runtime_mu);
    if (!g_runtime) return false;
    return g_runtime->scheduler().claims_tensor(w);
}

// /streamllm/stats accessors — pool-tier counters always tracked
// regardless of DIAG mode. Single atomic-load per call.
extern "C" {
unsigned long long streamllm_stat_pool_used_bytes(void) {
    StreamllmRuntime * rt = nullptr;
    { std::lock_guard<std::mutex> lk(g_runtime_mu); rt = g_runtime.get(); }
    return rt ? (unsigned long long) rt->pool().used_bytes() : 0ull;
}
unsigned long long streamllm_stat_pool_peak_bytes(void) {
    StreamllmRuntime * rt = nullptr;
    { std::lock_guard<std::mutex> lk(g_runtime_mu); rt = g_runtime.get(); }
    return rt ? (unsigned long long) rt->pool().peak_used_bytes() : 0ull;
}
unsigned long long streamllm_stat_pool_cap_bytes(void) {
    StreamllmRuntime * rt = nullptr;
    { std::lock_guard<std::mutex> lk(g_runtime_mu); rt = g_runtime.get(); }
    return rt ? (unsigned long long) rt->pool().capacity_bytes() : 0ull;
}
unsigned long long streamllm_stat_pool_h2d_bytes(void) {
    StreamllmRuntime * rt = nullptr;
    { std::lock_guard<std::mutex> lk(g_runtime_mu); rt = g_runtime.get(); }
    return rt ? (unsigned long long) rt->pool().total_h2d_bytes() : 0ull;
}
unsigned long long streamllm_stat_pool_h2d_calls(void) {
    StreamllmRuntime * rt = nullptr;
    { std::lock_guard<std::mutex> lk(g_runtime_mu); rt = g_runtime.get(); }
    return rt ? (unsigned long long) rt->pool().total_h2d_calls() : 0ull;
}

// DIAG-off weak fallbacks: the real implementations live in
// core/runtime_diag.cpp and only get linked when STREAMLLM_DIAG=ON.
// When the diag TU isn't in the build, these zeros become the answer.
__attribute__((weak)) unsigned long long streamllm_stat_tier_attempts(void)   { return 0ull; }
__attribute__((weak)) unsigned long long streamllm_stat_tier_vram_hits(void)  { return 0ull; }
__attribute__((weak)) unsigned long long streamllm_stat_tier_dram_hits(void)  { return 0ull; }
__attribute__((weak)) unsigned long long streamllm_stat_tier_ssd_misses(void) { return 0ull; }
__attribute__((weak)) int                streamllm_stat_diag_enabled(void)    { return 0; }
} // extern "C"

extern "C" bool streamllm_set_score_table(
    const float * thresholds, int n_thresh)
{
    if (thresholds == nullptr || n_thresh <= 0) return false;
    std::vector<float> th(thresholds, thresholds + n_thresh);
    StreamllmRuntime * rt = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_runtime_mu);
        rt = g_runtime.get();
    }
    if (rt == nullptr) return false;
    return qwen3::scheduler_set_score_table(rt->scheduler(), th);
}

bool streamllm_get_score_table(
    std::vector<float> & out_thresholds)
{
    out_thresholds.clear();
    StreamllmRuntime * rt = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_runtime_mu);
        rt = g_runtime.get();
    }
    if (rt == nullptr) return false;
    out_thresholds = qwen3::scheduler_score_thresholds_snapshot(rt->scheduler());
    return true;
}


void clear() {
    std::lock_guard<std::mutex> lk(g_runtime_mu);
    if (g_runtime && getenv("STREAMLLM_STATS")) {
        const auto & p = g_runtime->pool();
        std::fprintf(stderr,
            "streamllm-ext stats: scheduler=%s peak_pool=%.1f MB "
            "h2d=%.1f MB in %zu moves (avg %.1f KB/move)\n",
            g_runtime->scheduler().name(),
            (double)p.peak_used_bytes() / 1024.0 / 1024.0,
            (double)p.total_h2d_bytes() / 1024.0 / 1024.0,
            p.total_h2d_calls(),
            p.total_h2d_calls()
                ? (double)p.total_h2d_bytes() / 1024.0 / p.total_h2d_calls()
                : 0.0);
    }
    diag::clear_close_trace_and_dump();
    moe_dispatch::print_profile_if_enabled();

    ggml_cuda_set_mul_mat_hook(nullptr);
    ggml_cuda_set_fusion_skip_hook(nullptr);
    ggml_cuda_set_graph_compute_begin_hook(nullptr);
    ggml_cuda_set_graph_compute_end_hook(nullptr);
    ggml_cuda_set_mul_mat_id_hook(nullptr);
    ggml_cuda_set_topk_moe_hook(nullptr);
    ggml_cuda_set_user_node_claims_hook(nullptr);
    moe_dispatch::clear_topk_weights();
    g_executor.reset();
    g_runtime.reset();
    g_required_runtime_was_true = false;
    moe_dispatch::free_scratch();
    batched_gemm_shutdown();
}


// ---- ggml-cuda extern "C" entry points --------------------------------
//
// Scheduler-agnostic: each thin shim looks up the active runtime and
// delegates to a Scheduler virtual.  No qwen3:: namespace types in this
// path — adding a new Scheduler subclass (KV-cache attention, dense
// chunked matmul) needs no changes here.  Per-op dispatch routes to
// Scheduler::dispatch_node with the destination ggml_tensor (operand
// info is reachable via node->src[]).

extern "C" bool streamllm_try_cuda_mul_mat(
    cudaStream_t stream,
    const struct ggml_tensor * /*src0*/,
    const struct ggml_tensor * /*src1*/,
    struct ggml_tensor * dst)
{
    StreamllmRuntime * rt = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_runtime_mu);
        rt = g_runtime.get();
    }
    if (rt == nullptr) return false;
    return rt->scheduler().dispatch_node((StreamHandle) stream, dst);
}

extern "C" void streamllm_topk_moe_observed(
    cudaStream_t stream,
    const struct ggml_tensor * logits,
    struct ggml_tensor *       weights,
    struct ggml_tensor *       ids)
{
    StreamllmRuntime * rt = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_runtime_mu);
        rt = g_runtime.get();
    }
    if (rt == nullptr) return;
    rt->scheduler().observe_topk_moe(
        (StreamHandle) stream, logits, weights, ids);
}

extern "C" bool streamllm_try_cuda_mul_mat_id(
    cudaStream_t stream,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * ids,
    struct ggml_tensor *       dst)
{
    ModelExecutor *    exec = nullptr;
    StreamllmRuntime * rt   = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_runtime_mu);
        rt   = g_runtime.get();
        exec = g_executor.get();
    }
    if (rt == nullptr) return false;
    // Mode A: managed MoE dispatches route through the model
    // executor (SSOT §6.4.2).  The executor wraps the current
    // scheduler + plan + load + kernel flow; the per-op hook is the
    // integration boundary, the executor is the canonical owner.
    if (exec != nullptr) {
        return exec->forward_moe_block(
            (StreamHandle) stream, src0, src1, ids, dst);
    }
    // Fallback for legacy installs that pre-date the executor.
    return rt->scheduler().dispatch_node((StreamHandle) stream, dst);
}

// Graph-compute pre/post hooks. Fire at the top and bottom of
// ggml_backend_cuda_graph_compute (see ggml/include/ggml-cuda.h).
// Forward to the active scheduler so it can prewalk the cgraph
// (managed-tensor identification, prefetch, marker scan) before any
// node-level dispatch starts.
extern "C" void streamllm_graph_compute_begin(
    cudaStream_t                stream,
    const struct ggml_cgraph *  cgraph)
{
    StreamllmRuntime * rt = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_runtime_mu);
        rt = g_runtime.get();
    }
    if (rt == nullptr) return;
    // The scheduler owns its per-replay state — score-table
    // snapshot, residency reservations, anything else routing-
    // dependent — and seeds it from on_graph_compute_begin.
    rt->scheduler().on_graph_compute_begin((StreamHandle) stream, cgraph);
}

extern "C" void streamllm_graph_compute_end(
    cudaStream_t                stream,
    const struct ggml_cgraph *  cgraph)
{
    StreamllmRuntime * rt = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_runtime_mu);
        rt = g_runtime.get();
    }
    if (rt == nullptr) return;
    rt->scheduler().on_graph_compute_end((StreamHandle) stream, cgraph);
    // Drop replay-scoped chunk reservations.  Scheduler-owned now;
    // the default base-class clear_replay_reservations is a no-op
    // for schedulers that don't keep replay state.
    rt->scheduler().clear_replay_reservations();
}

} // namespace streamllm_ext
