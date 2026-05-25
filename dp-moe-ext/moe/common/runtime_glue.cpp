// DPMoE / qwen3 — runtime glue between llama.cpp's model-load
// path, ggml-cuda's hooks, and the Qwen3-MoE scheduler.  Owns the
// process-wide ``g_runtime`` singleton.  Lives in qwen3/ because every
// non-trivial responsibility — install-time MoE scratch sizing,
// ggml-cuda hook registration, score-policy live-dial entry points,
// per-op extern-C shims — funnels into model-specific behavior.

#include "runtime_glue.h"
#include "launch_diag.h"
#include "runtime.h"
#include "runtime_diag.h"
#include "stream_reader.h"
#include "moe_executor.h"
#include "anybcq_gemm.h"

#include "dispatch.h"
#include "moe_scheduler.h"  // qwen3::scheduler_* free-fn shims
#include "decoder/anybcq/chunked_matmul.h"  // anybcq counters (constraint 7)
#include "residency.h"  // MoEResidencyTracker eviction counters

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

namespace dp_moe_ext {

// Defined here, declared extern in core/runtime_hook_internal.h so the
// MoE dispatch impls in qwen3/ can read the active runtime under this
// mutex.
std::mutex                            g_runtime_mu;
std::unique_ptr<DPMoERuntime>     g_runtime;
std::unique_ptr<ModelExecutor>        g_executor;

// Mode A loader gate (Milestone 1, Step 1) — remember whether the
// currently-installed runtime came from a GGUF with
// ``dp_moe.required_runtime=true``.  A second install while a
// required-runtime model is live MUST hard-fail (see the
// executor-lifetime contract in docs/MODE_A_MILESTONE1.md Step 7
// design).  Defaults to false on cold start and after clear().
bool                                  g_required_runtime_was_true = false;

// Mode A M1 routing counters (post-cleanup).
//
//   g_sentinel_claims         — pre_op_hook claimed a sentinel and
//                               routed it to forward_moe_layer.
//   g_forward_moe_layer_calls — forward_moe_layer entered. Equals
//                               g_sentinel_claims by construction.
//
// The legacy-rail counters (managed_mul_mat_id_claims,
// forward_moe_block_calls, legacy_mul_mat_id_calls) were retired in
// the post-M1 cleanup along with the rails themselves. The cgraph
// audit (``DP_MOE_CGRAPH_AUDIT=1``) now carries the "no managed
// MUL_MAT_ID/MUL_MAT leaked" invariant.
std::atomic<uint64_t>                 g_sentinel_claims{0};
std::atomic<uint64_t>                 g_forward_moe_layer_calls{0};

// Benchmark/score-mode capture opt-in.  Mirrors MoEScheduler::
// allow_capture_ so the cgraph-level claim hook can short-circuit
// without locking g_runtime_mu on every node — install_for_gguf
// reads DP_MOE_ALLOW_CAPTURE once and writes here, the scheduler
// reads the same env var in its own on_install.  clear() resets to
// false so a subsequent install (e.g. server reload) re-reads the
// env.
std::atomic<bool>                     g_dp_moe_allow_capture{false};

// Mode A milestone 1, S1 — bound model-slot set. Records every
// model's dp_moe_executor slot address that has been wired by the
// install path. clear() walks this under g_runtime_mu and writes
// nullptr through each slot BEFORE tearing down g_executor, so a
// model's dp_moe_executor view never dangles. Slot owners
// (typically llama_model_free) remove their slot via
// unbind_model_slot.
//
// Type-erased over void** to keep DPMoE self-contained — the
// ext does not need llama-model.h.
//
// Capacity: M1 supports exactly one model (one-DPMoE-model-per-
// process); using vector here just to mirror the multi-model
// post-M1 shape without committing to it now.
std::vector<void **>                  g_bound_model_slots_;

namespace {

std::string resolve_executor_name(const GlobalMeta & g) {
    return g.executor;
}

// Conservative pool sizing used by install_for_gguf. Mirrors
// test_runtime_full.cpp::estimate_pool_bytes — summed over every
// managed tensor with a 2× margin for allocator fragmentation.
size_t estimate_pool_bytes(const StreamReader & r) {
    size_t total = 0;
    for (const auto & name : r.managed_tensor_names()) {
        const auto * L = r.layout(name);
        if (L == nullptr) continue;
        if (r.global().encoder == "direct_expert_block" ||
            r.global().encoder == "direct_matrix") {
            for (uint32_t b : L->chunk_bytes) total += (size_t) b;
            continue;
        }
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
    // Read general.architecture before freeing the gguf context — we
    // use it post-bind to pick the SwiGLU activation that matches the
    // model.  Default empty string means "use executor default" (SiLU).
    std::string gguf_arch;
    {
        const int64_t kid = gguf_find_key(ctx, "general.architecture");
        if (kid >= 0) {
            const char * s = gguf_get_val_str(ctx, kid);
            if (s) gguf_arch = s;
        }
    }
    gguf_free(ctx);
    if (!reader_opt) {
        // No dp_moe.* block — stock GGUF, leave default dispatch alone.
        return false;
    }

    const auto & reader = *reader_opt;

    std::lock_guard<std::mutex> lk(g_runtime_mu);

    if (g_runtime) {
        // Step 1 (Milestone 1) — refuse to silently replace a runtime
        // whose model declared required_runtime=true.  Any
        // llama_model::dp_moe_executor pointers into the prior
        // runtime would become dangling.  See the executor-lifetime
        // contract in docs/MODE_A_MILESTONE1.md (Step 7 design).
        // Pre-gate installs (required_runtime=false) keep the legacy
        // silent-replace behaviour for backward compatibility.
        if (g_required_runtime_was_true) {
            throw std::runtime_error(
                "DPMoE: a required_runtime model is already loaded; "
                "free it before loading another DPMoE model");
        }
        std::fprintf(stderr,
            "DPMoE: replacing previously-installed (non-required) "
            "runtime — prior model was not cleared explicitly\n");
        g_runtime.reset();
        g_executor.reset();
    }

    size_t cap = estimate_pool_bytes(reader);
    if (const char * s = getenv("DP_MOE_VRAM_CAP_MB")) {
        long mb = std::atol(s);
        if (mb > 0) cap = (size_t)mb * 1024UL * 1024UL;
    }

    // Mode A executor (SSOT §6.1.1, Milestone 1 Step 1 loader gate).
    // Gate before runtime install so stale metadata fails without
    // parsing/registering thousands of managed tensors into the pool.
    qwen3::register_qwen3_moe_anybcq_executor();
    const std::string exec_name = resolve_executor_name(reader.global());
    if (exec_name.empty()) {
        throw std::runtime_error(
            "DPMoE: managed artifact is missing dp_moe.executor; "
            "re-encode or re-materialize with current DPMoE metadata");
    }
    std::unique_ptr<ModelExecutor> new_executor =
        make_executor(exec_name.c_str());
    if (new_executor == nullptr) {
        throw std::runtime_error(
            "DPMoE: executor '" + exec_name +
            "' is not registered; re-encode or re-materialize this artifact");
    }

    // Benchmark/score-mode capture opt-in. Read here so the hook
    // short-circuit below sees the right value the first time it
    // fires (sentinel-bearing cgraphs land on the very first
    // graph_compute, before any HTTP handler can touch the runtime).
    if (const char * s = getenv("DP_MOE_ALLOW_CAPTURE")) {
        g_dp_moe_allow_capture.store(
            s[0] && s[0] != '0', std::memory_order_relaxed);
    } else {
        g_dp_moe_allow_capture.store(false, std::memory_order_relaxed);
    }
    const char * sched_name = getenv("DP_MOE_SCHEDULER");
    g_runtime = std::make_unique<DPMoERuntime>(
        cap, /*device=*/0, /*copy_stream=*/true, sched_name);

    g_runtime->install(reader, std::string(gguf_path));
    cudaDeviceSynchronize();

    diag::install_open_trace();

    if (!moe_dispatch::size_scratch_for(reader)) {
        g_runtime.reset();
        throw std::runtime_error(
            "DPMoE: failed to size cast scratch");
    }

    g_executor = std::move(new_executor);
    g_executor->bind_to_model(*g_runtime, reader, std::string(gguf_path));
    if (g_executor->uses_stock_moe_graph()) {
        g_dp_moe_allow_capture.store(false, std::memory_order_relaxed);
    }
    g_required_runtime_was_true = reader.global().required_runtime;

    // Per-arch activation is now set in the executor subclass's
    // constructor (Qwen3MoEExecutor → SiLU, DeepSeekMoEExecutor →
    // SiLU, Gemma4MoEExecutor → GELU).  The class identity is implied
    // by the GGUF's ``dp_moe.executor`` field; the dynamic-cast
    // auto-select path that used to live here is gone.  Log the arch
    // + activation for greppable provenance.
    if (auto * exec_moe =
            dynamic_cast<qwen3::MoEAnyBcqExecutor *>(g_executor.get())) {
        std::fprintf(stderr,
            "DPMoE: gguf_arch='%s' executor='%s' activation=%s\n",
            gguf_arch.c_str(),
            g_executor->name(),
            exec_moe->activation() == qwen3::Activation::GELU ? "GELU" : "SiLU");
    }

    std::fprintf(stderr,
        "DPMoE: runtime ready "
        "(scheduler=%s, executor=%s, pool %.1f / %.1f MB used)\n",
        g_runtime->scheduler().name(),
        g_executor->name(),
        (double)g_runtime->pool().used_bytes()     / 1024.0 / 1024.0,
        (double)g_runtime->pool().capacity_bytes() / 1024.0 / 1024.0);

    if (getenv("DP_MOE_STATS_RESET_AFTER_INSTALL")) {
        g_runtime->pool().reset_stats();
    }

    // Mode A milestone-1 hook surface — sentinel rail only (post-S8).
    //
    //   pre_op_hook           — claims sentinel nodes, routes to
    //                            forward_moe_layer.
    //   user_node_claims_hook — disables CUDA-graph capture for any
    //                            cgraph carrying a sentinel.
    //   mul_mat_hook          — survives only as the criterion-10
    //                            hard-fail point for managed-dense
    //                            tensors (S10). Body aborts loudly
    //                            on a managed-canonical match.
    //   graph_compute_begin   — score-table snapshot for the score-
    //                            dial side channel + (when
    //                            DP_MOE_CGRAPH_AUDIT=1) the
    //                            audit walk that verifies sentinel
    //                            identity survived cb callbacks and
    //                            zero managed MUL_MAT_IDs remain.
    //   graph_compute_end     — replay-reservation cleanup. No
    //                            instrumenter walk.
    //
    // The legacy installs — topk_moe_hook, fusion_skip_hook,
    // mul_mat_id_hook — were retired in S8 along with the legacy
    // backstop rail in pre_op_hook. No managed-MoE runtime path
    // remains reachable through them.
    ggml_cuda_set_pre_op_hook((void *) &dp_moe_pre_op);
    ggml_cuda_set_mul_mat_hook((void *) &dp_moe_try_cuda_mul_mat);
    ggml_cuda_set_graph_compute_begin_hook(
        (void *) &dp_moe_on_graph_audit_and_score_snapshot);
    ggml_cuda_set_graph_compute_end_hook(
        (void *) &dp_moe_on_graph_audit_and_score_snapshot_end);
    ggml_cuda_set_user_node_claims_hook(
        (void *) &dp_moe_user_node_claims);
    ggml_cuda_set_dp_moe_score_version_hook(
        (void *) &dp_moe_replay_dial_version);
    return true;
}

// ---------------------------------------------------------------------
// Mode A milestone 1, S1 — model-scoped executor binding APIs.
// See runtime_glue.h for the contract.

ModelExecutor * current_executor() {
    std::lock_guard<std::mutex> lk(g_runtime_mu);
    return g_executor.get();
}

void bind_model_slot(void ** dp_moe_executor_slot) {
    if (dp_moe_executor_slot == nullptr) return;
    std::lock_guard<std::mutex> lk(g_runtime_mu);
    // Idempotent — no duplicate insertion.
    if (std::find(g_bound_model_slots_.begin(), g_bound_model_slots_.end(),
                  dp_moe_executor_slot) == g_bound_model_slots_.end()) {
        g_bound_model_slots_.push_back(dp_moe_executor_slot);
    }
}

void unbind_model_slot(void ** dp_moe_executor_slot) {
    if (dp_moe_executor_slot == nullptr) return;
    std::lock_guard<std::mutex> lk(g_runtime_mu);
    auto it = std::find(g_bound_model_slots_.begin(),
                        g_bound_model_slots_.end(),
                        dp_moe_executor_slot);
    if (it != g_bound_model_slots_.end()) {
        g_bound_model_slots_.erase(it);
    }
    // Null the slot. Safe to do while the slot's owning model is still
    // alive (canonical call site is llama_model_free, before the model
    // is destroyed). After this returns, the slot owner is free to
    // destroy itself; clear()'s walk will not find this slot.
    *dp_moe_executor_slot = nullptr;
}

bool executor_uses_stock_moe_graph(void * dp_moe_executor) {
    auto * exec = reinterpret_cast<ModelExecutor *>(dp_moe_executor);
    return exec != nullptr && exec->uses_stock_moe_graph();
}

// ---------------------------------------------------------------------

extern "C" bool dp_moe_user_node_claims(const struct ggml_tensor * node) {
    // Per-node claim predicate consulted by ggml-cuda before
    // deciding whether to capture the cgraph into a cuda-graph.
    // DPMoE is an eager-only framework — its LOAD walks need
    // host-side decision-making and full CUDA API access on every
    // dispatch.  When the scheduler claims any node in the cgraph,
    // ggml-cuda disables cuda-graph capture for that compute and
    // every node runs eager.  The dp_moe dispatch is fast enough
    // (load-bound at tight cap, near-eager at full pin) that the
    // missing capture speedup doesn't matter for dp_moe's use
    // case.
    if (node == nullptr) return false;
    // Benchmark/score-mode opt-in: operator has set
    // DP_MOE_ALLOW_CAPTURE=1 AND DP_MOE_VRAM_CAP_MB large
    // enough to fit every chunk (verified by the scheduler's
    // on_install assert). Skip the per-node claim so ggml-cuda can
    // capture the cgraph; dial swaps are picked up by the score-
    // version hook, which invalidates the cached graph.
    const bool allow_capture =
        g_dp_moe_allow_capture.load(std::memory_order_relaxed);
    // Sentinel nodes (Mode A S5) are unambiguously claimed by name
    // and don't need a scheduler hop. Recognising them here keeps
    // the user-node-claims contract tight even if the scheduler's
    // claim set lags behind the cgraph (e.g. a brand-new layer
    // not yet seen by the scheduler).
    if (std::strncmp(node->name, "dp_moe.moe_layer_", 20) == 0) {
        return true;
    }
    std::lock_guard<std::mutex> lk(g_runtime_mu);
    if (!g_runtime) return false;
    if (g_executor && g_executor->uses_stock_moe_graph()) {
        if (node->op == GGML_OP_MUL_MAT_ID && node->src[0] != nullptr) {
            return g_runtime->scheduler().claims_tensor(node->src[0]);
        }
    }
    if (allow_capture) return false;
    return g_runtime->scheduler().claims_node(node);
}

extern "C" uint64_t dp_moe_replay_dial_version(void) {
    // Hot path: called by ggml-cuda's graph-update predicate once
    // per cgraph_compute. The runtime's atomic version field is
    // refreshed in MoEScheduler::on_graph_compute_begin alongside
    // the dial snapshot, so it stays consistent with the kernels'
    // grid dims for the current capture window.
    std::lock_guard<std::mutex> lk(g_runtime_mu);
    if (!g_runtime) return 0;
    return g_runtime->replay_dial_version();
}

// ── Runtime-mutable gradual-schedule state ────────────────────────────
//
// Lives in this TU so the HTTP route in tools/server can wire to the
// extern-C setters and the common-sampler observer in common/sampling
// can call dp_moe_schedule_get on init.  Mutex-guarded; the writers
// are HTTP handlers (very low frequency) and the reader is per-request
// common_sampler_init (also low frequency).
namespace {
std::mutex                            g_schedule_mu;
bool                                  g_schedule_active_v = false;
std::vector<int>                      g_schedule_thresholds_v;
std::vector<float>                    g_schedule_kbars_v;
int                                   g_schedule_allocator_v = 0;
}  // anon

extern "C" bool dp_moe_kbar_schedule_set(
    const int *   thresholds, int n_thresh,
    const float * kbars, int n_kbars,
    int allocator_mode)
{
    if (thresholds == nullptr || kbars == nullptr) {
        return false;
    }
    if (n_thresh < 0 || n_kbars <= 0 || n_kbars != n_thresh + 1) {
        return false;
    }
    // Validate strictly-ascending thresholds.
    for (int i = 1; i < n_thresh; ++i) {
        if (thresholds[i] <= thresholds[i - 1]) return false;
    }
    for (int i = 0; i < n_kbars; ++i) {
        if (!(kbars[i] >= 0.0f)) return false;
    }
    std::vector<int> new_thr(thresholds, thresholds + n_thresh);
    std::vector<float> new_kbars(kbars, kbars + n_kbars);
    std::lock_guard<std::mutex> lk(g_schedule_mu);
    g_schedule_thresholds_v = std::move(new_thr);
    g_schedule_kbars_v      = std::move(new_kbars);
    g_schedule_allocator_v  = allocator_mode;
    g_schedule_active_v     = true;
    return true;
}

extern "C" void dp_moe_schedule_clear(void) {
    std::lock_guard<std::mutex> lk(g_schedule_mu);
    g_schedule_active_v = false;
    g_schedule_thresholds_v.clear();
    g_schedule_kbars_v.clear();
    g_schedule_allocator_v = 0;
}

extern "C" bool dp_moe_schedule_active(void) {
    std::lock_guard<std::mutex> lk(g_schedule_mu);
    return g_schedule_active_v;
}

bool dp_moe_schedule_get(
    std::vector<int> *                out_thresholds,
    std::vector<float> *              out_kbars,
    int *                             out_allocator_mode)
{
    std::lock_guard<std::mutex> lk(g_schedule_mu);
    if (!g_schedule_active_v) return false;
    if (out_thresholds) *out_thresholds = g_schedule_thresholds_v;
    if (out_kbars)      *out_kbars      = g_schedule_kbars_v;
    if (out_allocator_mode) *out_allocator_mode = g_schedule_allocator_v;
    return true;
}

// dp_moe_claims_tensor extern-C was retired with the fusion_skip
// hook surface. The internal Scheduler::claims_tensor predicate
// survives because the S10 dense-managed clear-fail
// (dp_moe_try_cuda_mul_mat) still reads it.

// /dp_moe/stats accessors — pool-tier counters always tracked
// regardless of DIAG mode. Single atomic-load per call.
extern "C" {
unsigned long long dp_moe_stat_pool_used_bytes(void) {
    DPMoERuntime * rt = nullptr;
    { std::lock_guard<std::mutex> lk(g_runtime_mu); rt = g_runtime.get(); }
    return rt ? (unsigned long long) rt->pool().used_bytes() : 0ull;
}
unsigned long long dp_moe_stat_pool_peak_bytes(void) {
    DPMoERuntime * rt = nullptr;
    { std::lock_guard<std::mutex> lk(g_runtime_mu); rt = g_runtime.get(); }
    return rt ? (unsigned long long) rt->pool().peak_used_bytes() : 0ull;
}
unsigned long long dp_moe_stat_pool_cap_bytes(void) {
    DPMoERuntime * rt = nullptr;
    { std::lock_guard<std::mutex> lk(g_runtime_mu); rt = g_runtime.get(); }
    return rt ? (unsigned long long) rt->pool().capacity_bytes() : 0ull;
}
unsigned long long dp_moe_stat_pool_h2d_bytes(void) {
    DPMoERuntime * rt = nullptr;
    { std::lock_guard<std::mutex> lk(g_runtime_mu); rt = g_runtime.get(); }
    return rt ? (unsigned long long) rt->pool().total_h2d_bytes() : 0ull;
}
unsigned long long dp_moe_stat_pool_h2d_calls(void) {
    DPMoERuntime * rt = nullptr;
    { std::lock_guard<std::mutex> lk(g_runtime_mu); rt = g_runtime.get(); }
    return rt ? (unsigned long long) rt->pool().total_h2d_calls() : 0ull;
}

// Mode A Milestone 1, Step 6: cumulative count of chunks that
// were not at POINTER_TABLE_READY when the executor's pre-launch
// validation ran. Process-wide; release builds increment this
// instead of aborting. Debug builds abort on first miss and
// never increment past 0. See
// Qwen3MoEAnyBcqExecutor::validate_required_set_.
unsigned long long dp_moe_stat_required_set_misses(void) {
    return (unsigned long long)
        qwen3::Qwen3MoEAnyBcqExecutor::required_set_misses_
            .load(std::memory_order_relaxed);
}

// Mode A M1 routing counters — only the sentinel rail exists.
unsigned long long dp_moe_stat_sentinel_claims(void) {
    return (unsigned long long) g_sentinel_claims.load(std::memory_order_relaxed);
}
unsigned long long dp_moe_stat_forward_moe_layer_calls(void) {
    return (unsigned long long) g_forward_moe_layer_calls.load(std::memory_order_relaxed);
}

// DIAG-off weak fallbacks: the real implementations live in
// core/runtime_diag.cpp and only get linked when DP_MOE_DIAG=ON.
// When the diag TU isn't in the build, these zeros become the answer.
__attribute__((weak)) unsigned long long dp_moe_stat_tier_attempts(void)   { return 0ull; }
__attribute__((weak)) unsigned long long dp_moe_stat_tier_vram_hits(void)  { return 0ull; }
__attribute__((weak)) unsigned long long dp_moe_stat_tier_dram_hits(void)  { return 0ull; }
__attribute__((weak)) unsigned long long dp_moe_stat_tier_ssd_misses(void) { return 0ull; }
__attribute__((weak)) int                dp_moe_stat_diag_enabled(void)    { return 0; }
} // extern "C"

extern "C" bool dp_moe_set_kbar(float kbar, int allocator_mode)
{
    DPMoERuntime * rt = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_runtime_mu);
        rt = g_runtime.get();
    }
    if (rt == nullptr) return false;
    const auto mode = allocator_mode == 1
        ? qwen3::KBarAllocatorMode::Uniform
        : qwen3::KBarAllocatorMode::Profile;
    return qwen3::scheduler_set_kbar(rt->scheduler(), kbar, mode);
}

extern "C" bool dp_moe_get_kbar(float * out_kbar, int * out_allocator_mode)
{
    DPMoERuntime * rt = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_runtime_mu);
        rt = g_runtime.get();
    }
    if (rt == nullptr) return false;
    if (out_kbar) *out_kbar = qwen3::scheduler_kbar(rt->scheduler());
    if (out_allocator_mode) {
        *out_allocator_mode =
            qwen3::scheduler_kbar_allocator_mode(rt->scheduler()) ==
                    qwen3::KBarAllocatorMode::Uniform ? 1 : 0;
    }
    return true;
}

extern "C" bool dp_moe_set_prefill_decay_end(
    unsigned long long n_tokens, int reset_state)
{
    DPMoERuntime * rt = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_runtime_mu);
        rt = g_runtime.get();
    }
    if (rt == nullptr) return false;
    return qwen3::scheduler_set_prefill_decay_end(
        rt->scheduler(), (uint64_t) n_tokens, reset_state != 0);
}

void clear() {
    std::lock_guard<std::mutex> lk(g_runtime_mu);
    // Mode A M1 routing summary — sentinel-only rail.
    std::fprintf(stderr,
        "DPMoE routing: sentinel_claims=%llu "
        "forward_moe_layer_calls=%llu\n",
        (unsigned long long) g_sentinel_claims.load(std::memory_order_relaxed),
        (unsigned long long) g_forward_moe_layer_calls.load(std::memory_order_relaxed));

    // Correctness counters (constraint 7).  Healthy run has
    // validation_failures=0 and oob=0; comp_execute and kernel_launch
    // should both be > 0 if the MoE layer fired at all.
    {
        const unsigned long long val_misses =
            (unsigned long long)
            qwen3::Qwen3MoEAnyBcqExecutor::required_set_misses_
                .load(std::memory_order_relaxed);
        const unsigned long long comp_calls =
            anybcq::counter_chunk_matmul_calls();
        const unsigned long long kern_calls =
            anybcq::counter_kernel_launch_calls();
        const unsigned long long oob =
            anybcq::counter_n_chunks_oob();
        std::fprintf(stderr,
            "DPMoE correctness: required_set_misses=%llu "
            "comp_execute=%llu kernel_launch=%llu n_chunks_oob=%llu\n",
            val_misses, comp_calls, kern_calls, oob);
        std::fprintf(stderr,
            "DPMoE eviction: total=%llu during_dispatch=%llu\n",
            MoEResidencyTracker::evictions_total(),
            MoEResidencyTracker::evictions_during_dispatch());
    }

    if (g_runtime && getenv("DP_MOE_STATS")) {
        const auto & p = g_runtime->pool();
        std::fprintf(stderr,
            "DPMoE stats: scheduler=%s peak_pool=%.1f MB "
            "used_pool=%.1f MB residents=%zu "
            "h2d=%.1f MB in %zu moves (avg %.1f KB/move) "
            "batch_h2d=%.1f MB in %zu moves fallback=%zu\n",
            g_runtime->scheduler().name(),
            (double)p.peak_used_bytes() / 1024.0 / 1024.0,
            (double)p.used_bytes() / 1024.0 / 1024.0,
            p.n_resident(),
            (double)p.total_h2d_bytes() / 1024.0 / 1024.0,
            p.total_h2d_calls(),
            p.total_h2d_calls()
                ? (double)p.total_h2d_bytes() / 1024.0 / p.total_h2d_calls()
                : 0.0,
            (double)p.batch_h2d_bytes() / 1024.0 / 1024.0,
            p.batch_h2d_calls(),
            p.batch_fallbacks());
        // Residency-invariant counters.  At full cap with no eviction,
        // after warmup: resident_hits should dominate, load_misses
        // should approach zero, h2d_submitted should stop growing, and
        // unexpected_h2d_for_resident MUST be 0.  See
        // docs/BENCH_V2_BOTTLENECK.md.
        const size_t req  = p.required_chunks();
        const size_t hits = p.resident_hits();
        const double hit_pct = req
            ? 100.0 * (double)hits / (double)req : 0.0;
        std::fprintf(stderr,
            "DPMoE residency: required=%zu  resident_hits=%zu (%.1f%%)  "
            "load_misses=%zu  h2d_submitted=%zu  redundant_h2d_skipped=%zu  "
            "unexpected_h2d_for_resident=%zu\n",
            req, hits, hit_pct,
            p.load_misses(),
            p.h2d_submitted_chunks(),
            p.redundant_h2d_skipped(),
            p.unexpected_h2d_for_resident());
        const size_t pre_req = p.prefill_required_chunks();
        const size_t pre_hit = p.prefill_resident_hits();
        const size_t dec_req = p.decode_required_chunks();
        const size_t dec_hit = p.decode_resident_hits();
        std::fprintf(stderr,
            "DPMoE residency phases: prefill=%zu/%zu (%.1f%%)  "
            "decode=%zu/%zu (%.1f%%)\n",
            pre_hit, pre_req,
            pre_req ? 100.0 * (double)pre_hit / (double)pre_req : 0.0,
            dec_hit, dec_req,
            dec_req ? 100.0 * (double)dec_hit / (double)dec_req : 0.0);
        // Launch / sync breakdown for the sentinel eager MoE path.
        launch_diag::dump_to_stderr();
    }
    diag::clear_close_trace_and_dump();
    moe_dispatch::print_profile_if_enabled();

    ggml_cuda_set_pre_op_hook(nullptr);
    ggml_cuda_set_mul_mat_hook(nullptr);
    ggml_cuda_set_graph_compute_begin_hook(nullptr);
    ggml_cuda_set_graph_compute_end_hook(nullptr);
    ggml_cuda_set_user_node_claims_hook(nullptr);
    ggml_cuda_set_dp_moe_score_version_hook(nullptr);
    g_dp_moe_allow_capture.store(false, std::memory_order_relaxed);

    // Mode A milestone 1, S1: null every bound model's dp_moe_executor
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
    g_forward_moe_layer_calls.store(0, std::memory_order_relaxed);
    anybcq::reset_counters();
    MoEResidencyTracker::reset_eviction_counters();
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

extern "C" bool dp_moe_try_cuda_mul_mat(
    cudaStream_t                /*stream*/,
    const struct ggml_tensor *  src0,
    const struct ggml_tensor *  /*src1*/,
    struct ggml_tensor *        /*dst*/)
{
    // S10 (Mode A) — managed-dense clear-fail (M1 criterion 10).
    //
    // Mode A reserves every managed canonical for the MoE sentinel
    // rail. A managed canonical surfacing as a dense GGML_OP_MUL_MAT
    // means either:
    //   - an arch builder change introduced a non-MoE managed path
    //     (regression), or
    //   - a pruned-MoE artifact bypassed the loader's chunked-rail
    //     setup and the dense weights point at the placeholder
    //     storage we deliberately skip on the backend buffer.
    // Either way, stock dense matmul would read uninitialised bytes
    // for the dp_moe-skipped tensor and silently produce wrong
    // output. Refuse loudly instead.
    if (src0 == nullptr || src0->name[0] == '\0') return false;

    DPMoERuntime * rt = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_runtime_mu);
        rt = g_runtime.get();
    }
    if (rt == nullptr) return false;

    if (rt->scheduler().claims_tensor(src0)) {
        GGML_ABORT(
            "DPMoE: criterion 10 violation — managed canonical "
            "'%s' surfaced as GGML_OP_MUL_MAT. Mode A reserves managed "
            "tensors for the per-layer MoE sentinel rail; dense matmul "
            "fallback is forbidden. Most likely cause: a future arch "
            "builder introduced a non-MoE path over managed tensors, "
            "or a pruned-MoE artifact was loaded without sentinels.",
            src0->name);
    }
    return false;
}

// S8 (Mode A): ``dp_moe_try_cuda_mul_mat_id`` and
// ``dp_moe_topk_moe_observed`` were retired together with the
// ``mul_mat_id_hook`` and ``topk_moe_hook`` installs. Managed MoE
// dispatch flows exclusively through ``dp_moe_pre_op`` below.

extern "C" bool dp_moe_pre_op(
    cudaStream_t stream,
    struct ggml_tensor * dst)
{
    // Mode A milestone-1 (S5+S6+S7 + S8) — sentinel-only dispatch.
    //
    // Managed Qwen3-MoE flows entirely through the per-layer sentinel
    // rail emitted by qwen3moe.cpp's Mode A branch. The legacy
    // MUL_MAT_ID backstop was retired in S8 — if a managed MUL_MAT_ID
    // node ever surfaces in a cgraph again, the cgraph audit
    // (DP_MOE_CGRAPH_AUDIT=1) catches it at build time, and the
    // pre_op_hook deliberately does NOT claim it at compute time.
    // No silent legacy path remains reachable.
    if (dst == nullptr) return false;

    if (dst->op == GGML_OP_MUL_MAT_ID) {
        ModelExecutor * exec = nullptr;
        DPMoERuntime * rt = nullptr;
        {
            std::lock_guard<std::mutex> lk(g_runtime_mu);
            rt = g_runtime.get();
            exec = g_executor.get();
        }
        if (rt != nullptr && exec != nullptr && exec->uses_stock_moe_graph()) {
            return exec->prepare_moe_mul_mat_id((StreamHandle) stream, dst);
        }
    }

    static constexpr const char * kSentinelPrefix = "dp_moe.moe_layer_";
    constexpr size_t kSentinelPrefixLen = 20; // strlen("dp_moe.moe_layer_")
    if (std::strncmp(dst->name, kSentinelPrefix, kSentinelPrefixLen) != 0) {
        return false;
    }

    ModelExecutor *    exec = nullptr;
    DPMoERuntime * rt   = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_runtime_mu);
        rt   = g_runtime.get();
        exec = g_executor.get();
    }
    if (rt == nullptr || exec == nullptr) return false;

    const struct ggml_tensor * cur     = dst->src[0];
    const struct ggml_tensor * ids     = dst->src[1];
    const struct ggml_tensor * probs   = dst->src[2];
    const struct ggml_tensor * weights = dst->src[3];
    if (cur == nullptr || ids == nullptr ||
        probs == nullptr || weights == nullptr) {
        std::fprintf(stderr,
            "DPMoE: sentinel '%s' missing src[0..3] "
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

// Mode A M1 graph-compute callback — AUDIT + SCORE-SNAPSHOT ONLY.
//
// Fires at the top of ``ggml_backend_cuda_graph_compute``. The body
// must do only:
//   (a) score-table snapshot for the score-dial side channel, and
//   (b) the optional DP_MOE_CGRAPH_AUDIT=1 cgraph walk that proves
//       Mode A invariants (sentinels well-formed, no managed
//       MUL_MAT_ID/MUL_MAT nodes leaked).
//
// Forbidden in this callback (all scheduling lives at the per-sentinel
// dispatch site instead):
//   - routing planning / prior-token routing
//   - chunk reservations
//   - load submission
//   - managed-node dispatch
//   - fallback-path selection
//
// The legacy name ``dp_moe_graph_compute_begin`` was the entrypoint
// when this callback also drove the per-canonical instrumenter prewalk;
// after S8 retired that prewalk, the callback's role is strictly
// audit+snapshot. Rename reflects the new scope so a future reader
// can't mistake it for MoE scheduling semantics.
extern "C" void dp_moe_on_graph_audit_and_score_snapshot(
    cudaStream_t                stream,
    const struct ggml_cgraph *  cgraph)
{
    DPMoERuntime * rt = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_runtime_mu);
        rt = g_runtime.get();
    }
    if (rt == nullptr) return;
    rt->scheduler().on_graph_compute_begin((StreamHandle) stream, cgraph);
}

// End-of-cgraph callback — drops replay-scoped chunk reservations.
// Same forbidden-behavior contract as the begin callback above.
extern "C" void dp_moe_on_graph_audit_and_score_snapshot_end(
    cudaStream_t                stream,
    const struct ggml_cgraph *  cgraph)
{
    DPMoERuntime * rt = nullptr;
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

} // namespace dp_moe_ext
