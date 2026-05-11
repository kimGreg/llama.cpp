// streamllm-ext — Qwen3-MoE ggml-cuda hook bodies.
//
// Architecturally these belong with the scheduler: dispatch logic
// (graph introspection, gate-score reading, per-(t, u) plan, chunk
// fan-out, kernel launch) is model-specific. qwen3/qwen3_runtime_glue.cpp
// keeps lifecycle (install_for_gguf / clear) and the extern-C shims
// that ggml-cuda calls; those shims forward through
// Scheduler::handle_*, which lands here.

#include "qwen3_moe_dispatch.h"

#include "qwen3_runtime_glue.h"   // g_runtime + g_runtime_mu (internal externs)
#include "qwen3_moe_matmul_comp.h" // MoEMatMulComp + scheduler_lookup_moe_comp
#include "runtime.h"
#include "runtime_diag.h"
#include "stream_reader.h"
#include "scheduler.h"
#include "qwen3_moe_scheduler.h"  // qwen3::scheduler_* typed accessors
#include "anybcq_gemv.h"
#include "anybcq_gemm.h"
#include "qwen3_moe_fused.h"        // MoeExpertTable + qwen3::naver_gemv_moe_launch
#include "chunked_matmul.h"   // shortcut_anybcq::chunk_matmul_*for_wid
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
struct WeightsHandle {
    const ggml_tensor * weights = nullptr;
    int                 n_used  = 0;
};
std::mutex                                       g_topk_weights_mu;
std::unordered_map<const void *, WeightsHandle>  g_topk_weights;

// Per-stream scratch for F32↔F16 cast bridges + ids/precision device
// buffers used by the fused MoE kernel.  ``StreamScratch`` itself is
// declared in qwen3_moe_dispatch.h so MoEMatMulComp::execute can read
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

void on_topk_moe_observed_impl(
    cudaStream_t /*stream*/,
    const ggml_tensor * /*logits*/,
    ggml_tensor *       weights,
    ggml_tensor *       ids)
{
    if (ids == nullptr || weights == nullptr || ids->data == nullptr) {
        return;
    }
    int n_used = (int)weights->ne[0];
    if (n_used == 1) n_used = (int)weights->ne[1];
    std::lock_guard<std::mutex> lk(g_topk_weights_mu);
    g_topk_weights[ids->data] = WeightsHandle{weights, n_used};
}

bool topk_weights_lookup(
    const void *               ids_data,
    const struct ggml_tensor ** out_weights,
    int *                       out_n_used)
{
    if (ids_data == nullptr) return false;
    std::lock_guard<std::mutex> lk(g_topk_weights_mu);
    auto it = g_topk_weights.find(ids_data);
    if (it == g_topk_weights.end()) return false;
    if (it->second.weights == nullptr) return false;
    if (out_weights) *out_weights = it->second.weights;
    if (out_n_used)  *out_n_used  = it->second.n_used;
    return true;
}

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

void clear_topk_weights() {
    std::lock_guard<std::mutex> lk(g_topk_weights_mu);
    g_topk_weights.clear();
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


// ---- dense managed mul_mat dispatch -----------------------------------

bool handle_mul_mat_impl(
    cudaStream_t stream,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    struct ggml_tensor * dst)
{
    static std::atomic<int> n_called{0};
    static std::atomic<int> n_hit{0};
    static std::atomic<int> n_miss_shape{0};
    static std::atomic<int> n_miss_dtype{0};
    static std::atomic<int> n_miss_stride{0};
    static std::atomic<int> n_miss_runtime{0};
    static std::atomic<int> n_miss_name{0};

    auto diag = [&](const char * reason, const char * extra = nullptr) {
        if (getenv("STREAMLLM_TRACE")) {
            std::fprintf(stderr,
                "streamllm-hook: %s | name=%s src0.type=%d src1.type=%d "
                "dst.type=%d ne=[%lld,%lld,%lld,%lld]x[%lld,%lld,%lld,%lld] "
                "extra=%s\n",
                reason,
                src0 && src0->name[0] ? src0->name : "?",
                src0 ? (int)src0->type : -1,
                src1 ? (int)src1->type : -1,
                dst  ? (int)dst->type  : -1,
                src0?(long long)src0->ne[0]:-1, src0?(long long)src0->ne[1]:-1,
                src0?(long long)src0->ne[2]:-1, src0?(long long)src0->ne[3]:-1,
                src1?(long long)src1->ne[0]:-1, src1?(long long)src1->ne[1]:-1,
                src1?(long long)src1->ne[2]:-1, src1?(long long)src1->ne[3]:-1,
                extra ? extra : "");
        }
    };

    if (src0 == nullptr || dst == nullptr || src1 == nullptr) return false;
    n_called.fetch_add(1);

    if (getenv("STREAMLLM_BYPASS")) return false;

    StreamllmRuntime * rt = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_runtime_mu);
        rt = g_runtime.get();
    }
    if (rt == nullptr) { n_miss_runtime.fetch_add(1); return false; }

    const char * name = src0->name;
    if (name == nullptr || name[0] == '\0') {
        n_miss_name.fetch_add(1); diag("miss-noname"); return false;
    }

    {
        static std::mutex dbg_mu;
        static std::set<std::string> dbg_seen;
        if (getenv("STREAMLLM_DUMP_DENSE_NAMES")) {
            std::lock_guard<std::mutex> lk(dbg_mu);
            if (dbg_seen.insert(name).second) {
                std::fprintf(stderr,
                    "[dense-hook] name=%s ne=[%lld,%lld]x[%lld,%lld]\n",
                    name,
                    (long long)src0->ne[0], (long long)src0->ne[1],
                    (long long)src1->ne[0], (long long)src1->ne[1]);
            }
        }
    }

    const std::string wid(name);
    const Plan * plan = qwen3::scheduler_plan_dense(rt->scheduler(), wid, stream);
    if (plan == nullptr) {
        n_miss_name.fetch_add(1); diag("miss-name"); return false;
    }
    // Step-4b instrumenter: fire LayerBegin/LayerEnd markers when
    // crossing a layer boundary. dst is the unique node identity in
    // the prewalk-built node→layer map.
    qwen3::scheduler_on_managed_node_visit(
        rt->scheduler(), dst, (StreamHandle) stream);
    cudaStreamCaptureStatus cap_status = cudaStreamCaptureStatusNone;
    const bool in_capture =
        cudaStreamIsCapturing(stream, &cap_status) == cudaSuccess &&
        cap_status == cudaStreamCaptureStatusActive;
    for (const auto & mv : plan->moves) {
        if (in_capture && mv.wid != wid) continue;
        rt->move_chunk(mv.wid, mv.cid, mv.src, mv.dst, (StreamHandle)stream);
    }
    const UpstreamLayoutDevice * L = rt->layout(wid);
    if (L == nullptr || L->q_bias_fp16 == nullptr ||
        L->chunk_ptrs[0] == nullptr) {
        n_miss_name.fetch_add(1); diag("miss-layout"); return false;
    }

    const bool src1_f16 = (src1->type == GGML_TYPE_F16);
    const bool src1_f32 = (src1->type == GGML_TYPE_F32);
    const bool dst_f16  = (dst->type  == GGML_TYPE_F16);
    const bool dst_f32  = (dst->type  == GGML_TYPE_F32);
    if (!(src1_f16 || src1_f32) || !(dst_f16 || dst_f32)) {
        n_miss_dtype.fetch_add(1); diag("miss-dtype");
        return false;
    }

    if ((int)src0->ne[0] != L->K || (int)src0->ne[1] != L->M) {
        n_miss_shape.fetch_add(1); diag("miss-shape-src0"); return false;
    }
    if ((int)src1->ne[0] != L->K) {
        n_miss_shape.fetch_add(1); diag("miss-shape-src1"); return false;
    }
    if ((int)dst->ne[0]  != L->M) {
        n_miss_shape.fetch_add(1); diag("miss-shape-dst"); return false;
    }

    const int64_t n_tokens = src1->ne[1];
    if (n_tokens != dst->ne[1]) { n_miss_shape.fetch_add(1); return false; }
    if (src1->ne[2] > 1 || src1->ne[3] > 1) {
        n_miss_shape.fetch_add(1); diag("miss-shape-batch"); return false;
    }

    const size_t x_elt = src1_f16 ? sizeof(uint16_t) : sizeof(float);
    const size_t y_elt = dst_f16  ? sizeof(uint16_t) : sizeof(float);
    if (src1->nb[1] != (size_t)src1->ne[0] * x_elt) {
        n_miss_stride.fetch_add(1); diag("miss-stride-src1"); return false;
    }
    if (dst->nb[1]  != (size_t)dst->ne[0]  * y_elt) {
        n_miss_stride.fetch_add(1); diag("miss-stride-dst"); return false;
    }

    const size_t need_x = (size_t)L->K * sizeof(uint16_t);
    const size_t need_y = (size_t)L->M * sizeof(uint16_t);
    if (need_x > g_scratch_x_bytes || need_y > g_scratch_y_bytes) {
        diag("miss-scratch-too-small"); return false;
    }
    StreamScratch * sc = scratch_for_stream(stream);
    if (sc == nullptr) {
        diag("miss-scratch-alloc"); return false;
    }
    void * scratch_x_f16 = sc->x_f16;
    void * scratch_y_f16 = sc->y_f16;

    const uint8_t * x_base = (const uint8_t *) src1->data;
    uint8_t * y_base       = (uint8_t *)       dst->data;
    const size_t x_stride  = src1->nb[1];
    const size_t y_stride  = dst->nb[1];

    const bool stamp_constant = getenv("STREAMLLM_STAMP_DST") != nullptr;

    const bool use_batched = (n_tokens > 1) && !stamp_constant &&
                             (getenv("STREAMLLM_NO_BATCHED") == nullptr) &&
                             ((size_t)L->M * (size_t)L->K * sizeof(uint16_t)
                              <= g_scratch_w_bytes) &&
                             ((size_t)L->K * (size_t)n_tokens * sizeof(uint16_t)
                              <= g_scratch_xb_bytes) &&
                             ((size_t)L->M * (size_t)n_tokens * sizeof(uint16_t)
                              <= g_scratch_yb_bytes);

    bool did_batched = false;
    if (use_batched) {
        const void * x_block_f16 = src1->data;
        if (src1_f32) {
            launch_f32_to_f16(src1->data, sc->xb_f16,
                              (int)((size_t)L->K * n_tokens), stream);
            x_block_f16 = sc->xb_f16;
        }
        void * y_block_f16 = dst_f16 ? dst->data : sc->yb_f16;

        bool ok = shortcut_anybcq::chunk_matmul_batched_for_wid(
            *rt, wid, plan->chunks,
            x_block_f16, y_block_f16, (int)n_tokens,
            sc->w_f16, stream);
        if (ok) {
            if (dst_f32) {
                launch_f16_to_f32(y_block_f16, dst->data,
                                  (int)((size_t)L->M * n_tokens), stream);
            }
            did_batched = true;
        } else {
            diag("batched-fallback");
        }
    }

    if (!did_batched) {
        for (int64_t t = 0; t < n_tokens; ++t) {
            const void * x_t = x_base + (size_t)t * x_stride;
            void       * y_t = y_base + (size_t)t * y_stride;

            if (stamp_constant) {
                cudaMemsetAsync(y_t, 0x3c, (size_t)L->M * y_elt, stream);
                continue;
            }

            const void * x_f16 = x_t;
            if (src1_f32) {
                launch_f32_to_f16(x_t, scratch_x_f16, L->K, stream);
                x_f16 = scratch_x_f16;
            }

            void * y_f16 = dst_f16 ? y_t : scratch_y_f16;

            shortcut_anybcq::chunk_matmul_for_wid(
                *rt, wid, plan->chunks,
                x_f16, y_f16,
                /*n_tokens=*/1,
                /*x_stride=*/0, /*y_stride=*/0,
                stream);

            if (dst_f32) {
                launch_f16_to_f32(y_f16, y_t, L->M, stream);
            }
        }
    }
    rt->pool().record_compute_event(stream);
    qwen3::scheduler_after_compute(rt->scheduler(), wid, stream);
    if (getenv("STREAMLLM_DEVICE_SYNC")) {
        cudaDeviceSynchronize();
    }
    if (const char * wanted = getenv("STREAMLLM_DUMP_XY")) {
        if (std::strcmp(wanted, name) == 0) {
            static std::atomic<int> xy_dumped{0};
            int target_hit = 0;
            if (const char * hs = getenv("STREAMLLM_DUMP_HIT")) {
                target_hit = std::atoi(hs);
            }
            int this_hit = xy_dumped.fetch_add(1);
            if (this_hit == target_hit) {
                cudaStreamSynchronize(stream);
                const char * dir = getenv("STREAMLLM_DUMP_DIR");
                if (dir == nullptr) dir = "/tmp/sllm_hook";
                std::string mk = std::string("mkdir -p ") + dir;
                (void) system(mk.c_str());

                const size_t x_total = (size_t) L->K * n_tokens * x_elt;
                const size_t y_total = (size_t) L->M * n_tokens * y_elt;
                std::vector<uint8_t> x_buf(x_total), y_buf(y_total);
                cudaMemcpy(x_buf.data(), src1->data, x_total, cudaMemcpyDeviceToHost);
                cudaMemcpy(y_buf.data(), dst->data,  y_total, cudaMemcpyDeviceToHost);

                auto write_file = [&](const std::string & p, const void * data, size_t n) {
                    FILE * f = fopen(p.c_str(), "wb");
                    if (!f) return;
                    fwrite(data, 1, n, f);
                    fclose(f);
                };
                std::string x_path = std::string(dir) + "/" + name + ".x.bin";
                std::string y_path = std::string(dir) + "/" + name + ".y.bin";
                std::string m_path = std::string(dir) + "/" + name + ".meta.txt";
                write_file(x_path, x_buf.data(), x_total);
                write_file(y_path, y_buf.data(), y_total);

                FILE * mf = fopen(m_path.c_str(), "w");
                if (mf) {
                    std::fprintf(mf,
                        "name=%s\nM=%d\nK=%d\nP=%d\ngroup_size=%d\n"
                        "n_tokens=%lld\nsrc1_type=%s\ndst_type=%s\n",
                        name, L->M, L->K, L->n_chunks, L->group_size,
                        (long long)n_tokens,
                        src1_f16 ? "fp16" : "fp32",
                        dst_f16  ? "fp16" : "fp32");
                    fclose(mf);
                }
                std::fprintf(stderr,
                    "streamllm-dump-xy: wrote %s (X %zuB, Y %zuB, nt=%lld)\n",
                    name, x_total, y_total, (long long)n_tokens);
            }
        }
    }

    n_hit.fetch_add(1);
    diag("hit");
    return true;
}


// ---- MoE mul_mat_id dispatch ------------------------------------------
//
// Step 3 (Milestone 1) — the dispatch body lives on the executor now
// (Qwen3MoEAnyBcqExecutor::forward_moe_block). This symbol is a thin
// shim that survives so the legacy scheduler ``dispatch_node`` path
// (qwen3_moe_scheduler.cpp:797) keeps working without being rewired.
// Step 9 retires the shim entirely once the per-op hook installs are
// gone for managed installs.
//
// **Lock discipline**: the previous implementation held g_runtime_mu
// across the *entire* dispatch, which serialised every concurrent
// invocation against any unrelated install/clear and risked deadlock
// against the worker pool. The new shim copies the executor pointer
// under the lock, releases the lock immediately, and only then calls
// into the forward — matching the pattern at
// qwen3_runtime_glue.cpp::streamllm_try_cuda_mul_mat_id.
bool handle_mul_mat_id_impl(
    cudaStream_t stream,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * ids,
    struct ggml_tensor * dst)
{
    ModelExecutor * exec = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_runtime_mu);
        exec = g_executor.get();
    }
    if (exec == nullptr) return false;
    return exec->forward_moe_block(
        (StreamHandle) stream, src0, src1, ids, dst);
}

// Step 3 (Milestone 1): counter shim — the executor's forward_moe_block
// calls this on every managed dispatch. The counter lives in
// g_moe_profile alongside the rest of the dispatch-time counters so
// print_profile_if_enabled prints a coherent report.
void profile_inc_hook_calls() {
    g_moe_profile.hook_calls.fetch_add(1, std::memory_order_relaxed);
}

}  // namespace moe_dispatch
}  // namespace streamllm_ext
