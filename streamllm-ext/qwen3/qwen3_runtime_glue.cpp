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

#include <algorithm>              // std::find / std::remove
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>                // strncmp for sentinel name match
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

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

// Mode A milestone-1 S5+S6+S7 cutover — per-rail call counters.
// The cutover acceptance gates (MODE_A_MILESTONE1.md) read these
// counters to prove managed Qwen3-MoE dispatch flows entirely through
// the per-layer sentinel rail, not the legacy per-op MUL_MAT_ID rail.
//
//   g_sentinel_claims              — streamllm_pre_op recognised a
//                                    sentinel node and routed it to
//                                    forward_moe_layer. Cutover
//                                    expectation: > 0, equal to
//                                    n_managed_moe_layers ×
//                                    n_decode_graph_executions.
//   g_managed_mul_mat_id_claims    — legacy backstop fired (managed
//                                    MUL_MAT_ID claim from the
//                                    pre_op_hook). Cutover
//                                    expectation: == 0. Non-zero
//                                    means S5 missed a layer or
//                                    Mode A wasn't selected.
//   g_forward_moe_block_calls      — legacy executor entry called.
//                                    Cutover expectation: == 0.
//   g_forward_moe_layer_calls      — new per-layer entry called.
//                                    Cutover expectation: equal to
//                                    g_sentinel_claims.
//
// The cutover acceptance gate compares these four counters under
// the verifier in F (task #34).
std::atomic<uint64_t>                 g_sentinel_claims{0};
std::atomic<uint64_t>                 g_managed_mul_mat_id_claims{0};
std::atomic<uint64_t>                 g_forward_moe_block_calls{0};
std::atomic<uint64_t>                 g_forward_moe_layer_calls{0};
std::atomic<uint64_t>                 g_legacy_mul_mat_id_calls{0};

// Mode A milestone 1, S1 — bound model-slot set. Records every
// model's streamllm_executor slot address that has been wired by the
// install path. clear() walks this under g_runtime_mu and writes
// nullptr through each slot BEFORE tearing down g_executor, so a
// model's streamllm_executor view never dangles. Slot owners
// (typically llama_model_free) remove their slot via
// unbind_model_slot.
//
// Type-erased over void** to keep streamllm-ext self-contained — the
// ext does not need llama-model.h.
//
// Capacity: M1 supports exactly one model (one-StreamLLM-model-per-
// process); using vector here just to mirror the multi-model
// post-M1 shape without committing to it now.
std::vector<void **>                  g_bound_model_slots_;

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

    // Mode A milestone-1 Step 7+8: install the generic pre-op claim
    // hook FIRST. Fires before every op's default dispatch — claims
    // managed MUL_MAT_ID nodes and routes them through the executor's
    // forward_moe_block. Returning true short-circuits the per-op
    // switch.
    ggml_cuda_set_pre_op_hook((void *) &streamllm_pre_op);

    ggml_cuda_set_mul_mat_hook((void *) &streamllm_try_cuda_mul_mat);
    ggml_cuda_set_fusion_skip_hook((void *) &streamllm_claims_tensor);

    // Mode A milestone-1 Step 9: the legacy per-op mul_mat_id_hook
    // install is retired. Step 7+8 verification confirmed
    // legacy_mul_mat_id_calls=0 across the full decode — managed
    // dispatches all route through pre_op_hook above. Keep the
    // streamllm_try_cuda_mul_mat_id function alive for the
    // STREAMLLM_LEGACY_PEROP_HOOKS=1 bring-up escape hatch and for
    // any future post-M1 work that needs to A/B test the legacy
    // routing against the pre-op surface.
    if (getenv("STREAMLLM_LEGACY_PEROP_HOOKS")) {
        std::fprintf(stderr,
            "streamllm-ext: STREAMLLM_LEGACY_PEROP_HOOKS=1 — re-installing "
            "legacy mul_mat_id_hook as redundant insurance (pre_op_hook "
            "claims first; legacy never sees managed dispatches under "
            "normal operation)\n");
        ggml_cuda_set_mul_mat_id_hook((void *) &streamllm_try_cuda_mul_mat_id);
    }

    ggml_cuda_set_topk_moe_hook((void *) &streamllm_topk_moe_observed);
    ggml_cuda_set_graph_compute_begin_hook(
        (void *) &streamllm_graph_compute_begin);
    ggml_cuda_set_graph_compute_end_hook(
        (void *) &streamllm_graph_compute_end);
    ggml_cuda_set_user_node_claims_hook(
        (void *) &streamllm_user_node_claims);
    return true;
}

// ---------------------------------------------------------------------
// Mode A milestone 1, S1 — model-scoped executor binding APIs.
// See qwen3_runtime_glue.h for the contract.

ModelExecutor * current_executor() {
    std::lock_guard<std::mutex> lk(g_runtime_mu);
    return g_executor.get();
}

void bind_model_slot(void ** streamllm_executor_slot) {
    if (streamllm_executor_slot == nullptr) return;
    std::lock_guard<std::mutex> lk(g_runtime_mu);
    // Idempotent — no duplicate insertion.
    if (std::find(g_bound_model_slots_.begin(), g_bound_model_slots_.end(),
                  streamllm_executor_slot) == g_bound_model_slots_.end()) {
        g_bound_model_slots_.push_back(streamllm_executor_slot);
    }
}

void unbind_model_slot(void ** streamllm_executor_slot) {
    if (streamllm_executor_slot == nullptr) return;
    std::lock_guard<std::mutex> lk(g_runtime_mu);
    auto it = std::find(g_bound_model_slots_.begin(),
                        g_bound_model_slots_.end(),
                        streamllm_executor_slot);
    if (it != g_bound_model_slots_.end()) {
        g_bound_model_slots_.erase(it);
    }
    // Null the slot. Safe to do while the slot's owning model is still
    // alive (canonical call site is llama_model_free, before the model
    // is destroyed). After this returns, the slot owner is free to
    // destroy itself; clear()'s walk will not find this slot.
    *streamllm_executor_slot = nullptr;
}

// ---------------------------------------------------------------------

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
    // Sentinel nodes (Mode A S5) are unambiguously claimed by name
    // and don't need a scheduler hop. Recognising them here keeps
    // the user-node-claims contract tight even if the scheduler's
    // claim set lags behind the cgraph (e.g. a brand-new layer
    // not yet seen by the scheduler).
    if (std::strncmp(node->name, "streamllm.moe_layer_", 20) == 0) {
        return true;
    }
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

// Mode A Milestone 1, Step 6: cumulative count of chunks that
// were not at POINTER_TABLE_READY when the executor's pre-launch
// validation ran. Process-wide; release builds increment this
// instead of aborting. Debug builds abort on first miss and
// never increment past 0. See
// Qwen3MoEAnyBcqExecutor::validate_required_set_.
unsigned long long streamllm_stat_required_set_misses(void) {
    return (unsigned long long)
        qwen3::Qwen3MoEAnyBcqExecutor::required_set_misses_
            .load(std::memory_order_relaxed);
}

// Mode A milestone-1 S5+S6+S7 cutover: rail-routing counters. The
// acceptance gates compare these to expected counts to prove managed
// Qwen3-MoE flows through the per-layer sentinel rail.
unsigned long long streamllm_stat_sentinel_claims(void) {
    return (unsigned long long) g_sentinel_claims.load(std::memory_order_relaxed);
}
unsigned long long streamllm_stat_managed_mul_mat_id_claims(void) {
    return (unsigned long long) g_managed_mul_mat_id_claims.load(std::memory_order_relaxed);
}
unsigned long long streamllm_stat_forward_moe_block_calls(void) {
    return (unsigned long long) g_forward_moe_block_calls.load(std::memory_order_relaxed);
}
unsigned long long streamllm_stat_forward_moe_layer_calls(void) {
    return (unsigned long long) g_forward_moe_layer_calls.load(std::memory_order_relaxed);
}
unsigned long long streamllm_stat_legacy_mul_mat_id_calls(void) {
    return (unsigned long long) g_legacy_mul_mat_id_calls.load(std::memory_order_relaxed);
}

// Compatibility alias for the pre-cutover counter name.  Returns the
// sum across the two new rails so dashboards expecting a single
// "pre_op_claims" number still see all hook firings.
unsigned long long streamllm_stat_pre_op_claims(void) {
    return (unsigned long long) (
        g_sentinel_claims.load(std::memory_order_relaxed) +
        g_managed_mul_mat_id_claims.load(std::memory_order_relaxed));
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
    // Mode A milestone-1 S5+S6+S7 routing summary — emitted at clear
    // so the cutover acceptance verifier sees the rail-by-rail
    // breakdown. After the cutover, managed_mul_mat_id_claims and
    // forward_moe_block_calls MUST both be zero.
    std::fprintf(stderr,
        "streamllm-ext routing: sentinel_claims=%llu "
        "managed_mul_mat_id_claims=%llu forward_moe_block_calls=%llu "
        "forward_moe_layer_calls=%llu legacy_mul_mat_id_calls=%llu\n",
        (unsigned long long) g_sentinel_claims.load(std::memory_order_relaxed),
        (unsigned long long) g_managed_mul_mat_id_claims.load(std::memory_order_relaxed),
        (unsigned long long) g_forward_moe_block_calls.load(std::memory_order_relaxed),
        (unsigned long long) g_forward_moe_layer_calls.load(std::memory_order_relaxed),
        (unsigned long long) g_legacy_mul_mat_id_calls.load(std::memory_order_relaxed));
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

    ggml_cuda_set_pre_op_hook(nullptr);
    ggml_cuda_set_mul_mat_hook(nullptr);
    ggml_cuda_set_fusion_skip_hook(nullptr);
    ggml_cuda_set_graph_compute_begin_hook(nullptr);
    ggml_cuda_set_graph_compute_end_hook(nullptr);
    ggml_cuda_set_mul_mat_id_hook(nullptr);
    ggml_cuda_set_topk_moe_hook(nullptr);
    ggml_cuda_set_user_node_claims_hook(nullptr);
    moe_dispatch::clear_topk_weights();

    // Mode A milestone 1, S1: null every bound model's streamllm_executor
    // slot BEFORE g_executor.reset() runs. Otherwise a live model could
    // hold a dangling view into the freed executor. The slots are
    // type-erased void** — we write nullptr through each, never
    // dereferencing the owning model. Safe even if a caller misuses the
    // API and calls clear() between bind_model_slot and
    // unbind_model_slot (the slot must still be a writable address per
    // the lifetime contract).
    for (void ** slot : g_bound_model_slots_) {
        if (slot != nullptr) *slot = nullptr;
    }
    g_bound_model_slots_.clear();

    g_executor.reset();
    g_runtime.reset();
    g_required_runtime_was_true = false;
    g_sentinel_claims.store(0, std::memory_order_relaxed);
    g_managed_mul_mat_id_claims.store(0, std::memory_order_relaxed);
    g_forward_moe_block_calls.store(0, std::memory_order_relaxed);
    g_forward_moe_layer_calls.store(0, std::memory_order_relaxed);
    g_legacy_mul_mat_id_calls.store(0, std::memory_order_relaxed);
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
        g_legacy_mul_mat_id_calls.fetch_add(1, std::memory_order_relaxed);
        return exec->forward_moe_block(
            (StreamHandle) stream, src0, src1, ids, dst);
    }
    // Fallback for legacy installs that pre-date the executor.
    return rt->scheduler().dispatch_node((StreamHandle) stream, dst);
}

extern "C" bool streamllm_pre_op(
    cudaStream_t stream,
    struct ggml_tensor * dst)
{
    // Mode A milestone-1 S5+S6+S7 cutover entry.
    //
    // Two rails. The cutover gates require the legacy rail to stay
    // at zero firings — managed Qwen3-MoE must flow entirely through
    // the per-layer sentinel rail emitted by qwen3moe.cpp's Mode A
    // branch (S5).
    //
    //   Rail 1 (sentinel) — dst is an op node named
    //                       "streamllm.moe_layer_<L>". src[0]=cur,
    //                       src[1..3]={ids, probs, weights} from the
    //                       cgraph builder. Route to forward_moe_layer.
    //
    //   Rail 2 (legacy backstop) — dst is GGML_OP_MUL_MAT_ID over a
    //                       managed canonical (src0 in the scheduler's
    //                       claim set). Route to forward_moe_block.
    //                       After S5 this rail SHOULD never fire — the
    //                       counter exists to detect a regression.
    if (dst == nullptr) return false;

    ModelExecutor *    exec = nullptr;
    StreamllmRuntime * rt   = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_runtime_mu);
        rt   = g_runtime.get();
        exec = g_executor.get();
    }
    if (rt == nullptr || exec == nullptr) return false;

    // ── Rail 1: sentinel ────────────────────────────────────────────
    // Name match is the only authoritative predicate. ggml_dup is the
    // op the cgraph builder emits; we don't pre-filter on op so a
    // future shape (e.g. GGML_OP_CUSTOM) doesn't slip past.
    static constexpr const char * kSentinelPrefix = "streamllm.moe_layer_";
    constexpr size_t kSentinelPrefixLen = 20; // strlen("streamllm.moe_layer_")
    if (std::strncmp(dst->name, kSentinelPrefix, kSentinelPrefixLen) == 0) {
        const struct ggml_tensor * cur    = dst->src[0];
        const struct ggml_tensor * ids    = dst->src[1];
        const struct ggml_tensor * probs  = dst->src[2];
        const struct ggml_tensor * weights = dst->src[3];
        if (cur == nullptr || ids == nullptr ||
            probs == nullptr || weights == nullptr) {
            std::fprintf(stderr,
                "streamllm-ext: sentinel '%s' missing src[0..3] "
                "(src=[%p, %p, %p, %p]) — refusing to dispatch\n",
                dst->name, (void*)cur, (void*)ids,
                (void*)probs, (void*)weights);
            return false;
        }
        const int layer_idx = std::atoi(dst->name + kSentinelPrefixLen);
        g_sentinel_claims.fetch_add(1, std::memory_order_relaxed);
        g_forward_moe_layer_calls.fetch_add(1, std::memory_order_relaxed);
        return exec->forward_moe_layer(
            (StreamHandle) stream, cur, ids, probs, weights, dst, layer_idx);
    }

    // ── Rail 2: legacy MUL_MAT_ID backstop ─────────────────────────
    if (dst->op != GGML_OP_MUL_MAT_ID) return false;

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    const struct ggml_tensor * ids  = dst->src[2];
    if (src0 == nullptr || src1 == nullptr || ids == nullptr) return false;
    if (src0->name[0] == '\0') return false;
    if (!rt->scheduler().claims_tensor(src0)) return false;

    g_managed_mul_mat_id_claims.fetch_add(1, std::memory_order_relaxed);
    g_forward_moe_block_calls.fetch_add(1, std::memory_order_relaxed);
    return exec->forward_moe_block(
        (StreamHandle) stream, src0, src1, ids, dst);
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
