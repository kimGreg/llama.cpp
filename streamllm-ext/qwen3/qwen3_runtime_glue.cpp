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
#include "anybcq_gemm.h"

#include "qwen3_moe_dispatch.h"
#include "qwen3_moe_scheduler.h"  // qwen3::scheduler_* free-fn shims

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

namespace {

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
        std::fprintf(stderr,
            "streamllm-ext: replacing previously-installed runtime "
            "(prior model not cleared explicitly)\n");
        g_runtime.reset();
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

    std::fprintf(stderr,
        "streamllm-ext: runtime ready "
        "(scheduler=%s, pool %.1f / %.1f MB used)\n",
        g_runtime->scheduler().name(),
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
    // Per-node claim predicate consulted by ggml-cuda before deciding
    // whether to capture the cgraph into a cuda-graph. Return true for
    // any node whose dispatch needs the streamllm LOAD walk to run on
    // every invocation (managed mul_mat / mul_mat_id at tight cap).
    // Returning true for any node in a cgraph disables cuda-graph
    // capture for that compute call; everything still works because
    // the per-op streamllm hooks above intercept the regular dispatch
    // path in eager mode.
    //
    // At full-pin (every managed chunk fits in pool) we return false
    // so the cgraph captures normally — the captured plane pointers
    // stay valid across replays since the scheduler never evicts.
    // STREAMLLM_FORCE_EAGER=1 overrides to always disable capture
    // (useful for debugging or when residency dynamics defeat the
    // install-time heuristic).
    if (node == nullptr) return false;
    if (node->op != GGML_OP_MUL_MAT && node->op != GGML_OP_MUL_MAT_ID) {
        return false;
    }
    const ggml_tensor * w = node->src[0];
    if (w == nullptr || w->name[0] == '\0') return false;
    std::lock_guard<std::mutex> lk(g_runtime_mu);
    if (!g_runtime) return false;
    if (!g_runtime->is_managed_name(w->name)) return false;
    static const bool force_eager = []() {
        const char * e = std::getenv("STREAMLLM_FORCE_EAGER");
        return e && (e[0] == '1' || e[0] == 't' || e[0] == 'T');
    }();
    if (force_eager) return true;
    return !g_runtime->can_pin_all_managed();
}

extern "C" bool streamllm_claims_tensor(const struct ggml_tensor * w) {
    if (w == nullptr || w->name[0] == '\0') return false;
    std::lock_guard<std::mutex> lk(g_runtime_mu);
    if (!g_runtime) return false;
    // The default rule — managed-by-name — opts every managed tensor
    // out of upstream's fused-subgraph paths so the streamllm hook
    // can claim the underlying mul_mat. Concrete schedulers don't
    // currently override this beyond the name set.
    return g_runtime->is_managed_name(w->name);
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
    g_runtime.reset();
    moe_dispatch::free_scratch();
    batched_gemm_shutdown();
}


// ---- ggml-cuda extern "C" entry points --------------------------------
//
// All four are thin shims: they look up the active runtime, then route
// through Scheduler::handle_* so the scheduler is architecturally in
// the dispatch path. MoEScheduler's overrides land in
// qwen3/qwen3_moe_dispatch.cpp.

extern "C" bool streamllm_try_cuda_mul_mat(
    cudaStream_t stream,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    struct ggml_tensor * dst)
{
    StreamllmRuntime * rt = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_runtime_mu);
        rt = g_runtime.get();
    }
    if (rt == nullptr) return false;
    return qwen3::scheduler_handle_mul_mat(
        rt->scheduler(), (StreamHandle) stream, src0, src1, dst);
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
    qwen3::scheduler_on_topk_moe_observed(
        rt->scheduler(), (StreamHandle) stream, logits, weights, ids);
}

extern "C" bool streamllm_try_cuda_mul_mat_id(
    cudaStream_t stream,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * ids,
    struct ggml_tensor * dst)
{
    StreamllmRuntime * rt = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_runtime_mu);
        rt = g_runtime.get();
    }
    if (rt == nullptr) return false;
    return qwen3::scheduler_handle_mul_mat_id(
        rt->scheduler(), (StreamHandle) stream, src0, src1, ids, dst);
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
    // Snapshot the score table once per replay so every managed
    // dispatch in this token sees the same dial value (the host-fn
    // reads ``rt->current_replay_score_table()`` from inside its
    // plan()). Set/get on the scheduler is mutex-guarded so this
    // grabs a consistent view at this instant.
    rt->set_replay_score_table(
        qwen3::scheduler_score_thresholds_snapshot(rt->scheduler()));
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
    // Drop replay-scoped chunk reservations now that the captured
    // graph for this token has finished executing. The next replay's
    // host-fn callbacks rebuild them from scratch.
    rt->clear_replay_reservations();
}

} // namespace streamllm_ext
