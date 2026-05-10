// streamllm-ext — Qwen3-MoE ggml-cuda hook bodies.
//
// Architecturally these belong with the scheduler: dispatch logic
// (graph introspection, gate-score reading, per-(t, u) plan, chunk
// fan-out, kernel launch) is model-specific. core/runtime_hook.cpp
// keeps lifecycle (install_for_gguf / clear) and the extern-C shims
// that ggml-cuda calls; those shims forward through
// Scheduler::handle_*, which lands here.

#include "qwen3_moe_dispatch.h"

#include "qwen3_runtime_glue.h"   // g_runtime + g_runtime_mu (internal externs)
#include "runtime.h"
#include "runtime_hook_diag.h"
#include "stream_reader.h"
#include "scheduler.h"
#include "qwen3_moe_scheduler.h"  // qwen3::scheduler_* typed accessors
#include "anybcq_gemv.h"
#include "anybcq_gemm.h"
#include "moe_fused.h"        // MoeExpertTable + qwen3::naver_gemv_moe_launch
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
    std::atomic<uint64_t> prefetch_attempts{0};
    std::atomic<uint64_t> prefetch_skipped{0};
    std::atomic<uint64_t> prefetch_issued{0};
    std::atomic<uint64_t> compute_ns{0};
    std::atomic<uint64_t> prefetch_ns{0};
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

    std::atomic<uint64_t> prefetch_probe_ns{0};
    std::atomic<uint64_t> prefetch_move_ns{0};

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
// buffers used by the fused MoE kernel.
struct StreamScratch {
    void *  x_f16          = nullptr;
    void *  y_f16          = nullptr;
    void *  xb_f16         = nullptr;
    void *  yb_f16         = nullptr;
    void *  w_f16          = nullptr;
    void *  ids_d          = nullptr;
    void *  prec_per_tu_d  = nullptr;
};
std::mutex g_scratch_mu;
std::unordered_map<cudaStream_t, StreamScratch> g_scratch;
size_t g_scratch_x_bytes   = 0;
size_t g_scratch_y_bytes   = 0;
size_t g_scratch_xb_bytes  = 0;
size_t g_scratch_yb_bytes  = 0;
size_t g_scratch_w_bytes   = 0;
size_t g_scratch_ids_bytes = 0;
int    g_batch_n_max       = 0;

StreamScratch * scratch_for_stream(cudaStream_t stream) {
    std::lock_guard<std::mutex> lk(g_scratch_mu);
    auto it = g_scratch.find(stream);
    if (it != g_scratch.end()) return &it->second;
    StreamScratch s{};
    auto fail = [&]() {
        if (s.x_f16)         cudaFree(s.x_f16);
        if (s.y_f16)         cudaFree(s.y_f16);
        if (s.xb_f16)        cudaFree(s.xb_f16);
        if (s.yb_f16)        cudaFree(s.yb_f16);
        if (s.w_f16)         cudaFree(s.w_f16);
        if (s.ids_d)         cudaFree(s.ids_d);
        if (s.prec_per_tu_d) cudaFree(s.prec_per_tu_d);
        return nullptr;
    };
    if (cudaMalloc(&s.x_f16,  g_scratch_x_bytes)  != cudaSuccess) return fail();
    if (cudaMalloc(&s.y_f16,  g_scratch_y_bytes)  != cudaSuccess) return fail();
    if (cudaMalloc(&s.xb_f16, g_scratch_xb_bytes) != cudaSuccess) return fail();
    if (cudaMalloc(&s.yb_f16, g_scratch_yb_bytes) != cudaSuccess) return fail();
    if (cudaMalloc(&s.w_f16,  g_scratch_w_bytes)  != cudaSuccess) return fail();
    if (g_scratch_ids_bytes > 0 &&
        cudaMalloc(&s.ids_d, g_scratch_ids_bytes) != cudaSuccess) return fail();
    if (g_scratch_ids_bytes > 0 &&
        cudaMalloc(&s.prec_per_tu_d, g_scratch_ids_bytes) != cudaSuccess) return fail();
    auto ins = g_scratch.emplace(stream, s);
    return &ins.first->second;
}

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
unsigned long long streamllm_stat_prefetch_attempts(void) {
    return (unsigned long long) g_moe_profile.prefetch_attempts.load(std::memory_order_relaxed);
}
unsigned long long streamllm_stat_prefetch_skipped(void) {
    return (unsigned long long) g_moe_profile.prefetch_skipped.load(std::memory_order_relaxed);
}
unsigned long long streamllm_stat_prefetch_issued(void) {
    return (unsigned long long) g_moe_profile.prefetch_issued.load(std::memory_order_relaxed);
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
    const uint64_t att     = m.prefetch_attempts.load();
    const uint64_t skip    = m.prefetch_skipped.load();
    const uint64_t iss     = m.prefetch_issued.load();
    const uint64_t cn_ns   = m.compute_ns.load();
    const uint64_t pn_ns   = m.prefetch_ns.load();
    const double pf_hit_pct = att ? 100.0 * (double)skip / (double)att : 0.0;
    const uint64_t crq = m.compute_resident_query_ns.load();
    const uint64_t cci = m.compute_cast_in_ns.load();
    const uint64_t cmm = m.compute_matmul_ns.load();
    const uint64_t cco = m.compute_cast_out_ns.load();
    const uint64_t pp_probe = m.prefetch_probe_ns.load();
    const uint64_t pp_move  = m.prefetch_move_ns.load();
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
        "  prefetch_attempts  = %lu  plan->moves entries seen\n"
        "  prefetch_skipped   = %lu  (%.1f%% cache hit)\n"
        "  prefetch_issued    = %lu  actual move_chunk calls\n"
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

bool handle_mul_mat_id_impl(
    cudaStream_t stream,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * ids,
    struct ggml_tensor * dst) {
    if (src0 == nullptr || src0->name[0] == '\0') {
        return false;
    }
    if (src1 == nullptr || ids == nullptr || dst == nullptr) {
        return false;
    }

    std::lock_guard<std::mutex> lk(g_runtime_mu);
    if (!g_runtime) {
        return false;
    }

    const std::string canonical(src0->name);
    if (!g_runtime->is_managed_name(canonical)) {
        return false;
    }
    const std::string synth_zero = canonical + ":e0";
    if (g_runtime->layout(synth_zero) == nullptr) {
        return false;
    }
    // Step-4b instrumenter: fire LayerBegin/LayerEnd markers when
    // crossing a layer boundary. dst is the unique node identity in
    // the prewalk-built node→layer map.
    qwen3::scheduler_on_managed_node_visit(
        g_runtime->scheduler(), dst, (StreamHandle) stream);

    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
        ids->type != GGML_TYPE_I32) {
        std::fprintf(stderr,
            "streamllm-ext: mul_mat_id hook bailing on wid=%s "
            "(unexpected dtypes: src1=%d ids=%d dst=%d)\n",
            canonical.c_str(),
            (int)src1->type, (int)ids->type, (int)dst->type);
        return false;
    }

    const int K              = (int)src1->ne[0];
    const int n_tokens       = (int)src1->ne[2];
    const int n_used_per_tok = (int)ids->ne[0];
    const int M              = (int)dst->ne[0];

    const bool shared_x = (src1->ne[1] == 1);
    const bool per_tu_x = (src1->ne[1] == n_used_per_tok);
    if (!shared_x && !per_tu_x) {
        std::fprintf(stderr,
            "streamllm-ext: mul_mat_id bail %s (src1->ne[1]=%lld, expected 1 or %d)\n",
            canonical.c_str(), (long long)src1->ne[1], n_used_per_tok);
        return false;
    }
    if ((int)ids->ne[1] != n_tokens ||
        (int)dst->ne[1] != n_used_per_tok ||
        (int)dst->ne[2] != n_tokens) {
        std::fprintf(stderr,
            "streamllm-ext: mul_mat_id bail %s (ids/dst shape mismatch)\n",
            canonical.c_str());
        return false;
    }

    StreamScratch * sc = scratch_for_stream(stream);
    if (sc == nullptr) {
        std::fprintf(stderr,
            "streamllm-ext: mul_mat_id hook bailing — scratch alloc failed\n");
        return false;
    }

    cudaStreamCaptureStatus _cap_status = cudaStreamCaptureStatusNone;
    const bool _hook_in_capture =
        cudaStreamIsCapturing(stream, &_cap_status) == cudaSuccess &&
        _cap_status == cudaStreamCaptureStatusActive;

    std::vector<int32_t> ids_host;
    std::vector<float>   probs_host;
    int64_t              n_expert_in_probs = 0;
    bool                 have_real_scores  = false;
    std::vector<float>   weights_host;
    bool                 have_renorm_weights = false;
    if (!_hook_in_capture) {
        ids_host.resize((size_t)n_used_per_tok * (size_t)n_tokens);
        {
            const size_t row_bytes = (size_t)n_used_per_tok * sizeof(int32_t);
            const size_t src_pitch = (size_t)ids->nb[1];
            cudaError_t err = cudaMemcpy2DAsync(
                ids_host.data(), /*dpitch=*/row_bytes,
                ids->data,       /*spitch=*/src_pitch,
                /*width=*/row_bytes,
                /*height=*/(size_t)n_tokens,
                cudaMemcpyDeviceToHost, stream);
            if (err != cudaSuccess) {
                std::fprintf(stderr,
                    "streamllm-ext: mul_mat_id hook ids host-readback failed: %s\n",
                    cudaGetErrorString(err));
                return false;
            }
        }

        {
            std::lock_guard<std::mutex> tlk(g_topk_weights_mu);
            auto it = g_topk_weights.find(ids->data);
            if (it != g_topk_weights.end() &&
                it->second.weights != nullptr &&
                it->second.weights->data != nullptr) {
                const ggml_tensor * w = it->second.weights;
                const size_t bytes = (size_t)n_used_per_tok *
                                     (size_t)n_tokens * sizeof(float);
                weights_host.resize((size_t)n_used_per_tok * (size_t)n_tokens);
                if (cudaMemcpyAsync(weights_host.data(), w->data, bytes,
                                    cudaMemcpyDeviceToHost,
                                    stream) == cudaSuccess) {
                    have_renorm_weights = true;
                }
                g_topk_weights.erase(it);
            }
        }

        const ggml_tensor * probs = nullptr;
        if (ids->src[0] != nullptr) {
            if (ids->src[0]->type == GGML_TYPE_F32 &&
                ids->src[0]->data != nullptr &&
                (int64_t)ids->src[0]->ne[1] == n_tokens) {
                probs = ids->src[0];
            }
            if (probs == nullptr && ids->src[0]->src[0] != nullptr &&
                ids->src[0]->src[0]->type == GGML_TYPE_F32 &&
                ids->src[0]->src[0]->data != nullptr &&
                (int64_t)ids->src[0]->src[0]->ne[1] == n_tokens) {
                probs = ids->src[0]->src[0];
            }
        }
        if (probs != nullptr) {
            n_expert_in_probs = probs->ne[0];
            probs_host.resize((size_t)n_expert_in_probs * (size_t)n_tokens);
            if (cudaMemcpyAsync(probs_host.data(), probs->data,
                                probs_host.size() * sizeof(float),
                                cudaMemcpyDeviceToHost, stream) == cudaSuccess) {
                have_real_scores = true;
            }
        }

        if (cudaStreamSynchronize(stream) != cudaSuccess) {
            std::fprintf(stderr,
                "streamllm-ext: mul_mat_id hook stream-sync after ids/probs copy failed\n");
            return false;
        }
        static std::atomic<int> probs_logged{0};
        if (probs_logged.exchange(1) == 0) {
            const char * pname = (probs && probs->name[0])
                ? probs->name : "(unnamed)";
            double sum0 = 0.0;
            float  min0 = 0.0f, max0 = 0.0f;
            int    n_neg = 0;
            const int n_view = std::min((int)n_expert_in_probs,
                                        (int)probs_host.size());
            if (n_view > 0) {
                min0 = max0 = probs_host[0];
                for (int i = 0; i < n_view; ++i) {
                    const float v = probs_host[i];
                    sum0 += v;
                    if (v < min0) min0 = v;
                    if (v > max0) max0 = v;
                    if (v < 0.0f) ++n_neg;
                }
            }
            std::fprintf(stderr,
                "streamllm-ext: gate scores %s — tensor='%s' "
                "n_expert=%lld  token0_sum=%.4f min=%.4f max=%.4f "
                "n_negative=%d/%d\n",
                have_real_scores ? "captured"
                                  : "FALLBACK (rank-based)",
                pname,
                (long long)n_expert_in_probs,
                sum0, min0, max0, n_neg, n_view);
        }
    }

    if (getenv("STREAMLLM_TRACE")) {
        static std::unordered_set<std::string> seen;
        static std::mutex seen_mu;
        std::lock_guard<std::mutex> sk(seen_mu);
        if (seen.insert(canonical).second) {
            std::fprintf(stderr,
                "streamllm-ext: mul_mat_id hook fired wid=%s "
                "K=%d M=%d n_used=%d n_tokens=%d\n",
                canonical.c_str(), K, M, n_used_per_tok, n_tokens);
        }
    }

    Scheduler & sched = g_runtime->scheduler();
    g_moe_profile.hook_calls.fetch_add(1, std::memory_order_relaxed);

    const auto _hook_t0 = std::chrono::steady_clock::now();
    struct HookExit {
        std::chrono::steady_clock::time_point t0;
        ~HookExit() {
            const auto dt = std::chrono::steady_clock::now() - t0;
            g_moe_profile.hook_total_ns.fetch_add(
                (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(dt).count(),
                std::memory_order_relaxed);
        }
    } _hook_exit{_hook_t0};

    const bool prefetch_on =
        std::strcmp(getenv("STREAMLLM_MOE_PREFETCH") ?: "1", "0") != 0;
    const bool async_on = g_runtime->prefetch_worker_count() > 0;

    struct PerSlot {
        std::string synth;
        int         n_chunks_desired = 0;
        std::shared_ptr<std::atomic<uint32_t>> batch;
    };
    std::vector<PerSlot> slots((size_t)n_tokens * (size_t)n_used_per_tok);

    struct Reservation { std::string wid; int cid; };
    std::vector<Reservation> reservations;
    reservations.reserve(slots.size() * 4);

    const MoeExpertTable * fuse_table =
        qwen3::scheduler_moe_expert_table(sched, canonical);
    if (fuse_table == nullptr) {
        return false;
    }
    const auto * any_layout = g_runtime->layout(canonical + ":e0");
    const int group_size = any_layout ? any_layout->group_size : 0;
    if (group_size <= 0) {
        return false;
    }

    const uint8_t * x_base = (const uint8_t *) src1->data;
    uint8_t       * y_base = (uint8_t *)       dst->data;
    const size_t x_used_stride  = src1->nb[1];
    const size_t x_tok_stride   = src1->nb[2];
    const size_t y_tok_stride   = dst->nb[2];

    if (sc->ids_d == nullptr) {
        std::fprintf(stderr, "streamllm-ext: ids_d scratch missing\n");
        return false;
    }
    int32_t * ids_d = (int32_t *) sc->ids_d;
    const size_t x_tile_bytes = (size_t)K * sizeof(__half);

    const char * pol_env = getenv("STREAMLLM_MOE_POLICY");
    const bool dynamic_policy =
        pol_env != nullptr &&
        (std::strcmp(pol_env, "dynamic") == 0 ||
         std::strcmp(pol_env, "DYNAMIC") == 0);
    int minibatch = n_tokens;
    if (dynamic_policy && !_hook_in_capture && n_tokens > 1) {
        minibatch = 64;
        if (const char * s = getenv("STREAMLLM_MOE_PREFILL_MINIBATCH")) {
            int v = std::atoi(s);
            if (v > 0) minibatch = v;
        }
        if (minibatch <= 0 || minibatch > n_tokens) minibatch = n_tokens;
    }

    const auto t_load_start = std::chrono::steady_clock::now();
    STLM_NVTX_RANGE("moe_hook");

    for (int t_base = 0; t_base < n_tokens; t_base += minibatch) {
        const int sub_n = std::min(minibatch, n_tokens - t_base);

        int sub_max_precision = 0;
        std::vector<int> host_prec_per_tu;
        const bool score_outer = qwen3::scheduler_is_score_policy(sched);
        if (score_outer) {
            host_prec_per_tu.assign((size_t)sub_n * n_used_per_tok, 0);
        }
        std::shared_ptr<std::atomic<uint32_t>> batch;
        if (prefetch_on && !_hook_in_capture) {
            STLM_NVTX_RANGE("LOAD");

            const bool score = score_outer;
            // Per-expert max gate score across the minibatch. Under
            // batched prefill, one expert may serve many tokens with
            // different scores; we load enough chunks for the highest
            // and re-use them across the lower-scoring tokens (free,
            // since the chunks are already resident).
            std::unordered_map<int, float> max_gate_by_expert;
            std::unordered_map<int, int>   max_prec_by_expert;
            std::vector<int> unique_experts;
            const size_t reserve_n = (size_t)n_used_per_tok * 4;
            max_gate_by_expert.reserve(reserve_n);
            if (score) max_prec_by_expert.reserve(reserve_n);
            unique_experts.reserve(reserve_n);

            const auto _ph_load_t0 = std::chrono::steady_clock::now();

            // Snapshot the score table once per minibatch. set_score_table()
            // can swap the active table at any time (per-request live
            // dial); the snapshot guarantees an in-flight LOAD walk
            // doesn't tear if a swap lands mid-iteration.
            const std::vector<float> sc_thresh =
                qwen3::scheduler_score_thresholds_snapshot(sched);
            const std::vector<int>   sc_chunks =
                qwen3::scheduler_score_chunks_snapshot(sched);
            const int                sc_max    = kMaxChunksPerTensor;

            // Walk descending thresholds; first match wins. Below the
            // smallest threshold falls back to chunks.back() (the
            // tail-tier precision the user opted into).
            auto score_lookup = [&](float g) -> int {
                int desired = sc_chunks.empty() ? sc_max : sc_chunks.back();
                for (size_t k = 0; k < sc_thresh.size() &&
                                   k < sc_chunks.size(); ++k) {
                    if (g >= sc_thresh[k]) {
                        desired = sc_chunks[k];
                        break;
                    }
                }
                if (desired < 1)      desired = 1;
                if (desired > sc_max) desired = sc_max;
                return desired;
            };

            // Pass 1: collect per-expert max gate score + register every
            // (t, u)'s eid so unique_experts is dedup'd in encounter order.
            // Per-(t, u) precision is filled in Pass 2 once we know the
            // per-expert max — under batched prefill this gives every
            // (t, u) routed to expert e the SAME precision, derived from
            // e's highest-scoring token in the minibatch.
            std::vector<float> g_per_tu;
            if (score) {
                g_per_tu.assign((size_t)sub_n * n_used_per_tok, 0.0f);
            }
            for (int t = t_base; t < t_base + sub_n; ++t) {
                for (int u = 0; u < n_used_per_tok; ++u) {
                    const int eid =
                        ids_host[(size_t)t * n_used_per_tok + u];
                    float g;
                    if (have_renorm_weights) {
                        g = weights_host[(size_t)t * n_used_per_tok +
                                          (size_t)u];
                    } else if (have_real_scores &&
                        eid >= 0 && eid < n_expert_in_probs) {
                        g = probs_host[
                            (size_t)t * n_expert_in_probs + (size_t)eid];
                    } else {
                        g = 0.45f - 0.05f * (float)u;
                        if (g < 0.05f) g = 0.05f;
                    }
                    auto it_g = max_gate_by_expert.find(eid);
                    if (it_g == max_gate_by_expert.end()) {
                        max_gate_by_expert.emplace(eid, g);
                        unique_experts.push_back(eid);
                    } else if (g > it_g->second) {
                        it_g->second = g;
                    }
                    if (score) {
                        g_per_tu[(size_t)(t - t_base) *
                                  n_used_per_tok + u] = g;
                    }
                }
            }

            // Pass 2 (score policy only): per-expert max precision
            // from per-expert max g; broadcast back to per-(t, u).
            // For any-prec wids, translate the requested target precision
            // (in PLANES) to the precision actually served — chunks are
            // packed in tiers of [base_p planes][+1 plane]×(Pa−1), so
            // serving target P_t needs n_chunks = max(1, P_t−base_p+1)
            // clamped to Pa, yielding planes_served = base_p+n_chunks−1.
            // The kernel's prec_per_tu_d must equal planes_served to
            // avoid reading past the last-loaded plane pointer.
            auto translate_planes = [&](int target) -> int {
                if (any_layout && any_layout->any_precision) {
                    const int base_p = (int)any_layout->base_precision;
                    const int Pa     = (int)any_layout->n_chunks;
                    int n_chunks = target - base_p + 1;
                    if (n_chunks < 1)  n_chunks = 1;
                    if (n_chunks > Pa) n_chunks = Pa;
                    return base_p + n_chunks - 1;
                }
                return target;
            };
            if (score) {
                for (int eid : unique_experts) {
                    max_prec_by_expert[eid] =
                        translate_planes(score_lookup(max_gate_by_expert[eid]));
                }
                const int layer =
                    std::strncmp(canonical.c_str(), "blk.", 4) == 0
                        ? std::atoi(canonical.c_str() + 4) : -1;
                for (int t = t_base; t < t_base + sub_n; ++t) {
                    for (int u = 0; u < n_used_per_tok; ++u) {
                        const int eid =
                            ids_host[(size_t)t * n_used_per_tok + u];
                        const int desired = max_prec_by_expert[eid];
                        host_prec_per_tu[(size_t)(t - t_base) *
                                         n_used_per_tok + u] = desired;
                        const float g = g_per_tu[(size_t)(t - t_base) *
                                                  n_used_per_tok + u];
                        diag::record_gate_event(layer, t, u, eid,
                                                g, /*cum_before=*/0.0f,
                                                desired);
                        const std::string synthetic =
                            canonical + ":e" + std::to_string(eid);
                        for (int pp = 0; pp < desired; ++pp) {
                            diag::record_chunk(*g_runtime, synthetic,
                                               u, pp, cid_chunk(pp));
                        }
                    }
                }
            }

            {
                const auto dt = std::chrono::steady_clock::now() - _ph_load_t0;
                g_moe_profile.ph_load_walk_ns.fetch_add(
                    (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(dt).count(),
                    std::memory_order_relaxed);
            }
            const auto _ph_plan_t0 = std::chrono::steady_clock::now();
            std::unordered_set<std::string> issued_keys;
            issued_keys.reserve(unique_experts.size() * 8);
            if (async_on && !batch) {
                batch = std::make_shared<std::atomic<uint32_t>>(0);
            }
            for (int eid : unique_experts) {
                const Plan * plan;
                if (score) {
                    const int desired = max_prec_by_expert[eid];
                    plan = qwen3::scheduler_plan_for_expert_with_precision(
                        sched, canonical, eid, desired, stream);
                } else {
                    const float gate_score = max_gate_by_expert[eid];
                    plan = qwen3::scheduler_plan_for_expert(
                        sched, canonical, eid, gate_score, /*rank=*/-1, stream);
                }
                if (plan == nullptr) continue;
                const int n_chunks_desired =
                    std::max(0, (int)plan->chunks.size() - 1);
                // n_chunks_desired counts data chunks emitted by the plan;
                // for shortcut that equals plane count, but for any-prec
                // we have planes = base_p + n_chunks − 1.
                const int planes_for_kernel =
                    (any_layout && any_layout->any_precision &&
                     n_chunks_desired > 0)
                        ? (int)any_layout->base_precision + n_chunks_desired - 1
                        : n_chunks_desired;
                if (planes_for_kernel > sub_max_precision)
                    sub_max_precision = planes_for_kernel;

                for (const auto & mv : plan->moves) {
                    g_moe_profile.prefetch_attempts.fetch_add(
                        1, std::memory_order_relaxed);
                    std::string key = mv.wid + "#" + std::to_string(mv.cid);
                    if (!issued_keys.insert(std::move(key)).second) {
                        g_moe_profile.prefetch_skipped.fetch_add(
                            1, std::memory_order_relaxed);
                        continue;
                    }
                    qwen3::scheduler_reserve_for_dispatch(sched, mv.wid, mv.cid);
                    reservations.push_back({mv.wid, mv.cid});

                    if (g_runtime->pool().is_resident(mv.wid, mv.cid)) {
                        g_moe_profile.prefetch_skipped.fetch_add(
                            1, std::memory_order_relaxed);
                        continue;
                    }
                    g_moe_profile.prefetch_issued.fetch_add(
                        1, std::memory_order_relaxed);

                    if (async_on) {
                        g_runtime->submit_prefetch(mv.wid, mv.cid, batch);
                    } else {
                        g_runtime->move_chunk(mv.wid, mv.cid, mv.src, mv.dst,
                                              /*compute_stream=*/nullptr);
                    }
                }
            }

            {
                const auto dt = std::chrono::steady_clock::now() - _ph_plan_t0;
                g_moe_profile.ph_plan_lookup_ns.fetch_add(
                    (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(dt).count(),
                    std::memory_order_relaxed);
            }
        }

        int uniform_precision = sub_max_precision;
        if (uniform_precision <= 0) {
            const int env_base = (getenv("STREAMLLM_MOE_BASE_CHUNKS")
                                   ? std::atoi(getenv("STREAMLLM_MOE_BASE_CHUNKS"))
                                   : 0);
            const int env_hot  = (getenv("STREAMLLM_MOE_HOT_CHUNKS")
                                   ? std::atoi(getenv("STREAMLLM_MOE_HOT_CHUNKS"))
                                   : 0);
            uniform_precision = env_base + env_hot;
            if (uniform_precision <= 0)
                uniform_precision = kMaxChunksPerTensor;
        }
        if (!_hook_in_capture) {
            const auto _ph_wait_t0 = std::chrono::steady_clock::now();
            if (batch) {
                g_runtime->wait_prefetch_batch(batch);
            } else {
                g_runtime->wait_prefetch_idle();
            }
            const auto dt = std::chrono::steady_clock::now() - _ph_wait_t0;
            g_moe_profile.ph_wait_barrier_ns.fetch_add(
                (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(dt).count(),
                std::memory_order_relaxed);
        }

        // Cross-stream sync: ``wait_prefetch_*`` only drains the host
        // worker queue — it does NOT order compute_stream after the
        // copy_stream events that completed the H2D + per-plane
        // pointer-table updates.  Without an explicit wait_on_stream,
        // the refresh kernel below (and the GEMV kernel after it) can
        // start before the per-plane update kernel has run on
        // copy_stream → reads still-zero ``d_qbias_slot[e][0]`` →
        // d_q_bias_per_expert[e] = nullptr → kernel illegal-access.
        //
        // For shortcut canonicals this race exists too but doesn't
        // bite because d_q_bias_per_expert was captured at MoE-table
        // build with a valid pointer (install-time q_bias upload).
        // For any-prec, q_bias is per-chunk so the pointer is updated
        // by ``update_anyprec_after_load_async`` on every chunk land —
        // the compute side MUST wait on those events.
        for (const auto & r : reservations) {
            g_runtime->pool().wait_on_stream(r.wid, r.cid,
                                              (StreamHandle) stream);
        }
        const auto _ph_compute_t0 = std::chrono::steady_clock::now();

        const int n_x_slots = shared_x ? sub_n : sub_n * n_used_per_tok;

        const size_t k_bytes_f32 = (size_t)K * sizeof(float);
        const bool x_contig =
            shared_x ? (x_tok_stride == k_bytes_f32)
                     : (x_used_stride == k_bytes_f32 &&
                        x_tok_stride  == (size_t)n_used_per_tok * k_bytes_f32);
        if (x_contig) {
            const uint8_t * x_src0 =
                x_base + (size_t)t_base * x_tok_stride;
            launch_f32_to_f16(
                x_src0, sc->xb_f16, n_x_slots * K, stream);
        } else {
            for (int i = 0; i < n_x_slots; ++i) {
                const uint8_t * x_src = shared_x
                    ? x_base + (size_t)(t_base + i) * x_tok_stride
                    : x_base + (size_t)(t_base + i / n_used_per_tok) * x_tok_stride
                             + (size_t)(i % n_used_per_tok) * x_used_stride;
                launch_f32_to_f16(
                    x_src,
                    (uint8_t *)sc->xb_f16 + (size_t)i * x_tile_bytes,
                    K, stream);
            }
        }

        {
            const size_t row_bytes =
                (size_t)n_used_per_tok * sizeof(int32_t);
            const size_t sub_ids_bytes =
                (size_t)sub_n * row_bytes;
            if (sub_ids_bytes > g_scratch_ids_bytes) {
                std::fprintf(stderr,
                    "streamllm-ext: ids_d scratch too small for %zu B (have %zu B)\n",
                    sub_ids_bytes, g_scratch_ids_bytes);
                for (auto & r : reservations) {
                    qwen3::scheduler_release_from_dispatch(sched, r.wid, r.cid);
                }
                return false;
            }
            const uint8_t * ids_src =
                (const uint8_t *)ids->data + (size_t)t_base * ids->nb[1];
            cudaMemcpy2DAsync(
                ids_d, /*dpitch=*/row_bytes,
                ids_src, /*spitch=*/(size_t)ids->nb[1],
                /*width=*/row_bytes,
                /*height=*/(size_t)sub_n,
                cudaMemcpyDeviceToDevice, stream);
        }

        void * dst_sub = y_base + (size_t)t_base * y_tok_stride;
        const size_t dst_sub_bytes =
            (size_t)sub_n * n_used_per_tok * M * sizeof(float);
        cudaMemsetAsync(dst_sub, 0, dst_sub_bytes, stream);

        const int * prec_per_tu_d = nullptr;
        if (score_outer && !host_prec_per_tu.empty()) {
            const size_t bytes =
                host_prec_per_tu.size() * sizeof(int);
            if (bytes <= g_scratch_ids_bytes && sc->prec_per_tu_d) {
                cudaMemcpyAsync(sc->prec_per_tu_d,
                                host_prec_per_tu.data(),
                                bytes, cudaMemcpyHostToDevice, stream);
                prec_per_tu_d = (const int *) sc->prec_per_tu_d;
            }
        }

        // Any-prec wids store β per-chunk; the table's d_q_bias_per_expert
        // entries were captured at install (then null) and need to be
        // refreshed from each expert's current d_qbias_slot[0] before
        // the kernel reads them. No-op for shortcut canonicals.
        qwen3::refresh_q_bias_for_anyprec_launch(*fuse_table,
                                                   (StreamHandle) stream);

        qwen3::naver_gemv_moe_launch(
            sc->xb_f16, dst_sub, ids_d, *fuse_table,
            M, K, sub_n, n_used_per_tok,
            uniform_precision, prec_per_tu_d,
            group_size,
            shared_x ? 1 : 0,
            (StreamHandle) stream);

        const int n_disp = sub_n * n_used_per_tok;
        g_moe_profile.compute_dispatches.fetch_add(
            n_disp, std::memory_order_relaxed);
        if (prec_per_tu_d) {
            size_t total_chunks = 0;
            for (int p : host_prec_per_tu) {
                int hist_idx = std::min(p, (int)kMaxChunksPerTensor);
                if (hist_idx >= 0) {
                    g_moe_profile.compute_planes_hist[hist_idx].fetch_add(
                        1, std::memory_order_relaxed);
                }
                total_chunks += (size_t)(p + 1);  // +1 for q_bias
            }
            g_moe_profile.compute_chunks_sum.fetch_add(
                total_chunks, std::memory_order_relaxed);
        } else {
            g_moe_profile.compute_chunks_sum.fetch_add(
                (size_t)n_disp * (uniform_precision + 1),
                std::memory_order_relaxed);
            const int hist_idx = std::min(
                uniform_precision, (int)kMaxChunksPerTensor);
            g_moe_profile.compute_planes_hist[hist_idx].fetch_add(
                n_disp, std::memory_order_relaxed);
        }
        {
            const auto dt = std::chrono::steady_clock::now() - _ph_compute_t0;
            g_moe_profile.ph_compute_ops_ns.fetch_add(
                (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(dt).count(),
                std::memory_order_relaxed);
        }
    }

    g_runtime->pool().record_compute_event(stream);
    const auto _ph_release_t0 = std::chrono::steady_clock::now();

    {
        auto dt = std::chrono::steady_clock::now() - t_load_start;
        g_moe_profile.compute_ns.fetch_add(
            std::chrono::duration_cast<std::chrono::nanoseconds>(dt).count(),
            std::memory_order_relaxed);
    }

    for (auto & r : reservations) {
        qwen3::scheduler_release_from_dispatch(sched, r.wid, r.cid);
    }
    {
        const auto dt = std::chrono::steady_clock::now() - _ph_release_t0;
        g_moe_profile.ph_release_ns.fetch_add(
            (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(dt).count(),
            std::memory_order_relaxed);
    }

    return true;
}

} // namespace moe_dispatch
} // namespace streamllm_ext
