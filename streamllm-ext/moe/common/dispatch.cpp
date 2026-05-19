// streamllm-ext — Qwen3-MoE ggml-cuda hook bodies.
//
// Architecturally these belong with the scheduler: dispatch logic
// (graph introspection, gate-score reading, per-(t, u) plan, chunk
// fan-out, kernel launch) is model-specific. qwen3/runtime_glue.cpp
// keeps lifecycle (install_for_gguf / clear) and the extern-C shims
// that ggml-cuda calls; those shims forward through
// Scheduler::handle_*, which lands here.

#include "dispatch.h"

#include "runtime_glue.h"   // g_runtime + g_runtime_mu (internal externs)
#include "matmul_comp.h" // MoEMatMulComp + scheduler_lookup_moe_comp
#include "runtime.h"
#include "runtime_diag.h"
#include "stream_reader.h"
#include "moe_scheduler.h"
#include "moe_scheduler.h"  // qwen3::scheduler_* typed accessors
#include "anybcq_gemv.h"
#include "anybcq_gemm.h"
#include "fused_kernels.h"        // MoeExpertTable + qwen3::naver_gemv_moe_launch
#include "chunked_matmul.h"   // ss_anybcq::chunk_matmul_*for_wid
#include "streamllm_nvtx.h"

#include <ggml.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace streamllm_ext {

// MoE chunk-movement profiling counters. Enabled by STREAMLLM_PROFILE=1
// at the env-var level; printed at teardown via clear() ->
// moe_dispatch::print_profile_if_enabled.
namespace {

struct MoEProfile {
    std::atomic<uint64_t> hook_calls{0};
    std::atomic<uint64_t> compute_dispatches{0};
    std::atomic<uint64_t> compute_chunks_sum{0};
    std::atomic<uint64_t> async_load_attempts{0};
    std::atomic<uint64_t> async_load_skipped{0};
    std::atomic<uint64_t> async_load_issued{0};
    std::atomic<uint64_t> compute_ns{0};
    std::atomic<uint64_t> async_load_ns{0};
    std::atomic<uint64_t> hook_total_ns{0};
    std::atomic<uint64_t> ph_load_walk_ns{0};
    std::atomic<uint64_t> ph_plan_lookup_ns{0};
    std::atomic<uint64_t> ph_wait_barrier_ns{0};
    std::atomic<uint64_t> ph_compute_ops_ns{0};
    std::atomic<uint64_t> ph_release_ns{0};

    std::atomic<uint64_t> compute_resident_query_ns{0};
    std::atomic<uint64_t> compute_cast_in_ns{0};
    std::atomic<uint64_t> compute_matmul_ns{0};
    std::atomic<uint64_t> compute_cast_out_ns{0};

    std::atomic<uint64_t> async_load_probe_ns{0};
    std::atomic<uint64_t> async_load_move_ns{0};

    std::atomic<uint64_t> mc_pread_ns{0};
    std::atomic<uint64_t> mc_pool_load_ns{0};
    std::atomic<uint64_t> mc_calls{0};

    std::atomic<uint64_t> compute_planes_hist[kMaxChunksPerTensor + 1] = {};
    std::atomic<uint64_t> compute_skipped_zero{0};
    std::atomic<uint64_t> pool_make_room_calls{0};
};
MoEProfile g_moe_profile;

// Side-channel: ggml-cuda's topk_moe fusion captures (ids, weights)
// here so the mul_mat_id hook can later read the post-norm routing
// weights instead of the bypassed softmax buffer. Keyed by the ids
// tensor's data pointer — stable for the lifetime of one forward
// pass.
// S8 (Mode A): the renormalised-weights topk side channel
// (``g_topk_weights`` / ``WeightsHandle`` / ``g_topk_weights_mu``) was
// retired with the legacy MUL_MAT_ID dispatch surface. The sentinel
// carries weights through ``src[3]`` directly.

// Per-stream scratch for F32↔F16 cast bridges + ids/precision device
// buffers used by the fused MoE kernel.  ``StreamScratch`` itself is
// declared in dispatch.h so MoEMatMulComp::execute can read
// its fields without a TU-private re-definition.
std::mutex g_scratch_mu;
std::unordered_map<cudaStream_t, moe_dispatch::StreamScratch> g_scratch;
size_t g_scratch_x_bytes   = 0;
size_t g_scratch_y_bytes   = 0;
size_t g_scratch_xb_bytes  = 0;
size_t g_scratch_yb_bytes  = 0;
size_t g_scratch_w_bytes   = 0;
size_t g_scratch_ids_bytes = 0;
int    g_batch_n_max       = 0;

} // anonymous namespace


// Public: runtime.cpp records per-pread / per-pool-load timings here.
void profile_record_move_chunk_ssd(uint64_t pread_ns, uint64_t pool_load_ns) {
    g_moe_profile.mc_pread_ns.fetch_add(pread_ns, std::memory_order_relaxed);
    g_moe_profile.mc_pool_load_ns.fetch_add(pool_load_ns, std::memory_order_relaxed);
    g_moe_profile.mc_calls.fetch_add(1, std::memory_order_relaxed);
}

// Public: runtime.cpp bumps this every time pool.load returns null and
// the scheduler is asked for a victim.
void profile_record_pool_make_room() {
    g_moe_profile.pool_make_room_calls.fetch_add(1, std::memory_order_relaxed);
}

// Public counter accessors for /streamllm/stats. Each is a single
// atomic load — ~1 ns. Safe to call at high frequency.
extern "C" {
unsigned long long streamllm_stat_hook_calls(void) {
    return (unsigned long long) g_moe_profile.hook_calls.load(std::memory_order_relaxed);
}
unsigned long long streamllm_stat_dispatch_count(void) {
    return (unsigned long long) g_moe_profile.compute_dispatches.load(std::memory_order_relaxed);
}
unsigned long long streamllm_stat_async_load_attempts(void) {
    return (unsigned long long) g_moe_profile.async_load_attempts.load(std::memory_order_relaxed);
}
unsigned long long streamllm_stat_async_load_skipped(void) {
    return (unsigned long long) g_moe_profile.async_load_skipped.load(std::memory_order_relaxed);
}
unsigned long long streamllm_stat_async_load_issued(void) {
    return (unsigned long long) g_moe_profile.async_load_issued.load(std::memory_order_relaxed);
}
unsigned long long streamllm_stat_make_room_calls(void) {
    return (unsigned long long) g_moe_profile.pool_make_room_calls.load(std::memory_order_relaxed);
}
unsigned long long streamllm_stat_mc_calls(void) {
    return (unsigned long long) g_moe_profile.mc_calls.load(std::memory_order_relaxed);
}
unsigned long long streamllm_stat_mc_pread_ns(void) {
    return (unsigned long long) g_moe_profile.mc_pread_ns.load(std::memory_order_relaxed);
}
}


namespace moe_dispatch {

// S8 (Mode A): ``on_topk_moe_observed_impl`` and
// ``topk_weights_lookup`` were retired together with the
// ``topk_moe_hook`` install — the sentinel rail carries probs and
// renormalised weights through ``src[2..3]`` directly.

StreamScratch * scratch_for_stream(cudaStream_t stream) {
    std::lock_guard<std::mutex> lk(g_scratch_mu);
    auto it = g_scratch.find(stream);
    if (it != g_scratch.end()) return &it->second;
    StreamScratch s{};
    auto fail = [&]() {
        if (s.x_f16)  cudaFree(s.x_f16);
        if (s.y_f16)  cudaFree(s.y_f16);
        if (s.xb_f16) cudaFree(s.xb_f16);
        if (s.yb_f16) cudaFree(s.yb_f16);
        if (s.w_f16)  cudaFree(s.w_f16);
        if (s.ids_d)  cudaFree(s.ids_d);
        return nullptr;
    };
    if (cudaMalloc(&s.x_f16,  g_scratch_x_bytes)  != cudaSuccess) return fail();
    if (cudaMalloc(&s.y_f16,  g_scratch_y_bytes)  != cudaSuccess) return fail();
    if (cudaMalloc(&s.xb_f16, g_scratch_xb_bytes) != cudaSuccess) return fail();
    if (cudaMalloc(&s.yb_f16, g_scratch_yb_bytes) != cudaSuccess) return fail();
    if (cudaMalloc(&s.w_f16,  g_scratch_w_bytes)  != cudaSuccess) return fail();
    if (g_scratch_ids_bytes > 0 &&
        cudaMalloc(&s.ids_d, g_scratch_ids_bytes) != cudaSuccess) return fail();
    auto ins = g_scratch.emplace(stream, s);
    return &ins.first->second;
}

size_t scratch_xb_bytes_total()  { return g_scratch_xb_bytes; }
size_t scratch_yb_bytes_total()  { return g_scratch_yb_bytes; }
size_t scratch_ids_bytes_total() { return g_scratch_ids_bytes; }
int    scratch_batch_n_max()     { return g_batch_n_max; }


bool size_scratch_for(const StreamReader & r) {
    size_t max_k = 0;
    size_t max_m = 0;
    size_t max_mk = 0;
    for (const auto & name : r.managed_tensor_names()) {
        const auto * L = r.layout(name);
        if (L == nullptr) continue;
        size_t K  = (size_t)L->padded_m;
        size_t M  = (size_t)L->shape[0];
        max_k  = std::max(max_k,  K);
        max_m  = std::max(max_m,  M);
        max_mk = std::max(max_mk, M * K);
    }
    g_batch_n_max = 2048;
    if (const char * s = getenv("STREAMLLM_BATCH_N_MAX")) {
        int v = std::atoi(s); if (v > 0) g_batch_n_max = v;
    }
    g_scratch_x_bytes  = max_k * sizeof(uint16_t);
    g_scratch_y_bytes  = max_m * sizeof(uint16_t);
    g_scratch_xb_bytes = max_k * (size_t)g_batch_n_max * sizeof(uint16_t);
    g_scratch_yb_bytes = max_m * (size_t)g_batch_n_max * sizeof(uint16_t);
    g_scratch_w_bytes  = max_mk * sizeof(uint16_t);
    g_scratch_ids_bytes = (size_t)g_batch_n_max * 32 * sizeof(int32_t);
    return g_scratch_x_bytes > 0 && g_scratch_y_bytes > 0 &&
           g_scratch_w_bytes > 0;
}

void free_scratch() {
    std::lock_guard<std::mutex> lk(g_scratch_mu);
    for (auto & [s, sc] : g_scratch) {
        if (sc.x_f16)  cudaFree(sc.x_f16);
        if (sc.y_f16)  cudaFree(sc.y_f16);
        if (sc.xb_f16) cudaFree(sc.xb_f16);
        if (sc.yb_f16) cudaFree(sc.yb_f16);
        if (sc.w_f16)  cudaFree(sc.w_f16);
        if (sc.ids_d)  cudaFree(sc.ids_d);
    }
    g_scratch.clear();
    g_scratch_x_bytes = g_scratch_y_bytes = 0;
    g_scratch_xb_bytes = g_scratch_yb_bytes = g_scratch_w_bytes = 0;
    g_scratch_ids_bytes = 0;
}

void print_profile_if_enabled() {
    if (!getenv("STREAMLLM_PROFILE")) return;
    const auto & m = g_moe_profile;
    const uint64_t calls   = m.hook_calls.load();
    const uint64_t disps   = m.compute_dispatches.load();
    const uint64_t chunks  = m.compute_chunks_sum.load();
    const uint64_t att     = m.async_load_attempts.load();
    const uint64_t skip    = m.async_load_skipped.load();
    const uint64_t iss     = m.async_load_issued.load();
    const uint64_t cn_ns   = m.compute_ns.load();
    const uint64_t pn_ns   = m.async_load_ns.load();
    const double pf_hit_pct = att ? 100.0 * (double)skip / (double)att : 0.0;
    const uint64_t crq = m.compute_resident_query_ns.load();
    const uint64_t cci = m.compute_cast_in_ns.load();
    const uint64_t cmm = m.compute_matmul_ns.load();
    const uint64_t cco = m.compute_cast_out_ns.load();
    const uint64_t pp_probe = m.async_load_probe_ns.load();
    const uint64_t pp_move  = m.async_load_move_ns.load();
    const uint64_t mc_pread = m.mc_pread_ns.load();
    const uint64_t mc_pload = m.mc_pool_load_ns.load();
    const uint64_t mc_n     = m.mc_calls.load();
    const double per_disp_us = disps ? (double)cn_ns / (double)disps / 1000.0 : 0.0;
    const double per_iss_us  = iss   ? (double)pp_move / (double)iss / 1000.0 : 0.0;
    const double mc_pread_us = mc_n  ? (double)mc_pread / (double)mc_n / 1000.0 : 0.0;
    const double mc_pload_us = mc_n  ? (double)mc_pload / (double)mc_n / 1000.0 : 0.0;
    std::fprintf(stderr,
        "streamllm-ext moe-profile:\n"
        "  hook_calls         = %lu  (per mul_mat_id firing)\n"
        "  compute_dispatches = %lu  chunk_matmul calls\n"
        "  compute_avg_chunks = %.2f  (BASE+cached_HOT per call)\n"
        "  async_load_attempts  = %lu  plan->moves entries seen\n"
        "  async_load_skipped   = %lu  (%.1f%% cache hit)\n"
        "  async_load_issued    = %lu  actual move_chunk calls\n"
        "  compute_loop_total = %.3f s     (%.2f us/dispatch)\n"
        "    breakdown: resident_query=%.3f s  cast_in=%.3f s  matmul=%.3f s  cast_out=%.3f s\n"
        "  prefetch_loop_total= %.3f s     (%.2f us/issue)\n"
        "    breakdown: probe=%.3f s  move_chunk=%.3f s\n"
        "  move_chunk(SSD-stream) per-call:  pread=%.2f us  pool_load=%.2f us  (n=%lu)\n"
        "  hook_total_wallclock = %.3f s     (%.2f ms/hook_call) — host-side hook overhead\n"
        "    phase breakdown:  load_walk=%.3f s  plan_lookup=%.3f s  wait_barrier=%.3f s  compute_ops=%.3f s  release=%.3f s\n",
        (unsigned long)calls, (unsigned long)disps,
        disps ? (double)chunks / (double)disps : 0.0,
        (unsigned long)att, (unsigned long)skip, pf_hit_pct,
        (unsigned long)iss,
        (double)cn_ns / 1e9, per_disp_us,
        (double)crq / 1e9, (double)cci / 1e9,
        (double)cmm / 1e9, (double)cco / 1e9,
        (double)pn_ns / 1e9, per_iss_us,
        (double)pp_probe / 1e9, (double)pp_move / 1e9,
        mc_pread_us, mc_pload_us, (unsigned long)mc_n,
        (double)m.hook_total_ns.load() / 1e9,
        calls ? (double)m.hook_total_ns.load() / (double)calls / 1e6 : 0.0,
        (double)m.ph_load_walk_ns.load() / 1e9,
        (double)m.ph_plan_lookup_ns.load() / 1e9,
        (double)m.ph_wait_barrier_ns.load() / 1e9,
        (double)m.ph_compute_ops_ns.load() / 1e9,
        (double)m.ph_release_ns.load() / 1e9);

    const uint64_t mr  = m.pool_make_room_calls.load();
    const uint64_t skz = m.compute_skipped_zero.load();
    std::fprintf(stderr,
        "  compute planes hist (N planes used = count, %% of dispatches):\n");
    for (int i = 0; i <= kMaxChunksPerTensor; ++i) {
        const uint64_t v = m.compute_planes_hist[i].load();
        if (v == 0) continue;
        const double pct = disps ? 100.0 * (double)v / (double)disps : 0.0;
        std::fprintf(stderr,
            "    %2d planes : %12lu  (%5.1f %%)\n",
            i, (unsigned long)v, pct);
    }
    std::fprintf(stderr,
        "  compute_skipped_zero = %lu  (dispatch zeroed because plane 0 missing)\n"
        "  pool_make_room_calls = %lu  (= make_room_for invocations)\n",
        (unsigned long)skz, (unsigned long)mr);
}


// Mode A M1 dense-managed clear-fail (see runtime_glue.cpp's
// streamllm_try_cuda_mul_mat). The previous dense streaming body
// ``handle_mul_mat_impl`` was retired in the post-M1 cleanup pass —
// any future dense streaming lands as a separate DenseExecutor on
// the same Runtime / Scheduler / ChunkedTensor framework, not by
// re-enabling this body.


// S8 (Mode A): ``handle_mul_mat_id_impl`` was retired together with
// the per-canonical ``forward_moe_block`` virtual it called and the
// scheduler ``dispatch_node`` rail that called it. Managed MoE
// dispatch lives entirely on the sentinel rail (runtime_glue.cpp
// ``streamllm_pre_op`` → ``forward_moe_layer``).

// Step 3 (Milestone 1): counter shim — the executor's forward_moe_block
// calls this on every managed dispatch. The counter lives in
// g_moe_profile alongside the rest of the dispatch-time counters so
// print_profile_if_enabled prints a coherent report.
void profile_inc_hook_calls() {
    g_moe_profile.hook_calls.fetch_add(1, std::memory_order_relaxed);
}

}  // namespace moe_dispatch
}  // namespace streamllm_ext
