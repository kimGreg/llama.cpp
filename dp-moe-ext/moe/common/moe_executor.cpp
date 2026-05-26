// DPMoE / qwen3 — Qwen3 MoE AnyBCQ executor implementation.
//
// Milestone 1 Step 3 (2026-05-12): the dispatch body moved here from
// moe_dispatch::handle_mul_mat_id_impl. The executor now owns the
// canonical Mode A flow (current-batch routing → plan → reserve → load
// → wait → kernel → release). The per-op hook entry point in
// runtime_glue.cpp::dp_moe_try_cuda_mul_mat_id reaches us
// directly; the legacy scheduler.dispatch_node path (used when a
// managed MUL_MAT_ID surfaces through the dense-matmul hook) reaches
// us through the thin shim that survives in dispatch.cpp.
//
// `rt_` is set once at bind_to_model time and is read-only from then
// on — the body does NOT take g_runtime_mu. The runtime's internal
// state (pool, scheduler, layouts) carries its own thread-safety.

#include "moe_executor.h"
#include "dispatch.h"        // scratch_for_stream, topk_weights_lookup, profile_inc_hook_calls
#include "fused_kernels.h"           // launch_swiglu_mul, launch_weighted_reduce_slots
#include "matmul_comp.h"     // MoEMatMulComp, MoEInput/Output
#include "moe_scheduler.h"       // qwen3::scheduler_* helpers
#include "computation.h"               // ChunkPlan, ChunkKey
#include "launch_diag.h"               // launch_diag counters
#include "vram_pool.h"                 // ChunkState
#include "runtime.h"
#include "moe_scheduler.h"

#include <chrono>
#include <vector>

#include <ggml.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>   // getenv (DP_MOE_VIOLATION_DUMP)
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace dp_moe_ext { namespace qwen3 {

namespace {

bool planner_profile_enabled()
{
    static const bool enabled = [] {
        const char * s = std::getenv("DP_MOE_PROFILE_PLANNER");
        if (s == nullptr || s[0] == '\0') return false;
        return std::strcmp(s, "0") != 0 &&
               std::strcmp(s, "false") != 0 &&
               std::strcmp(s, "FALSE") != 0 &&
               std::strcmp(s, "off") != 0 &&
               std::strcmp(s, "OFF") != 0;
    }();
    return enabled;
}

uint64_t elapsed_ns(std::chrono::steady_clock::time_point t0,
                    std::chrono::steady_clock::time_point t1)
{
    return (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
        t1 - t0).count();
}

} // namespace

std::atomic<std::uint64_t> MoEAnyBcqExecutor::required_set_misses_{0};

MoEAnyBcqExecutor::~MoEAnyBcqExecutor()
{
    free_dynamic_plan_();
    std::lock_guard<std::mutex> lk(slot_scratch_mu_);
    for (auto & [_, ss] : slot_scratch_) {
        if (ss.slot_a) cudaFree(ss.slot_a);
        if (ss.slot_b) cudaFree(ss.slot_b);
    }
    slot_scratch_.clear();
}

void MoEAnyBcqExecutor::bind_to_model(
    DPMoERuntime &  rt,
    const StreamReader & /*reader*/,
    const std::string &  /*gguf_path*/)
{
    rt_ = &rt;
}

void MoEAnyBcqExecutor::free_dynamic_plan_()
{
    if (dynamic_plan_.chunks_d)  cudaFree(dynamic_plan_.chunks_d);
    if (dynamic_plan_.chunks_h)  cudaFreeHost(dynamic_plan_.chunks_h);
    if (dynamic_plan_.expert_order_d) cudaFree(dynamic_plan_.expert_order_d);
    if (dynamic_plan_.expert_order_h) cudaFreeHost(dynamic_plan_.expert_order_h);
    if (dynamic_plan_.n_active_d) cudaFree(dynamic_plan_.n_active_d);
    if (dynamic_plan_.n_active_h) cudaFreeHost(dynamic_plan_.n_active_h);
    if (dynamic_plan_.dp_prev_d) cudaFree(dynamic_plan_.dp_prev_d);
    if (dynamic_plan_.dp_cur_d)  cudaFree(dynamic_plan_.dp_cur_d);
    if (dynamic_plan_.trace_d)   cudaFree(dynamic_plan_.trace_d);
    dynamic_plan_ = DynamicGpuPlanScratch{};
}

bool MoEAnyBcqExecutor::ensure_dynamic_plan_(int n_experts, int B1)
{
    if (n_experts <= 0 || B1 <= 0) return false;
    if (dynamic_plan_.chunks_d != nullptr &&
        dynamic_plan_.n_experts >= n_experts &&
        dynamic_plan_.B1 >= B1) {
        return true;
    }
    free_dynamic_plan_();

    dynamic_plan_.n_experts = n_experts;
    dynamic_plan_.B1 = B1;
    const size_t chunks_bytes = (size_t) n_experts * sizeof(int);
    const size_t n_active_bytes = sizeof(int);
    const size_t dp_bytes = (size_t) B1 * sizeof(float);
    const size_t trace_bytes = (size_t) n_experts * (size_t) B1 * sizeof(uint8_t);

    if (cudaMalloc((void **) &dynamic_plan_.chunks_d, chunks_bytes) != cudaSuccess) {
        free_dynamic_plan_();
        return false;
    }
    if (cudaMallocHost((void **) &dynamic_plan_.chunks_h, chunks_bytes) != cudaSuccess) {
        free_dynamic_plan_();
        return false;
    }
    if (cudaMalloc((void **) &dynamic_plan_.expert_order_d, chunks_bytes) != cudaSuccess ||
        cudaMallocHost((void **) &dynamic_plan_.expert_order_h, chunks_bytes) != cudaSuccess ||
        cudaMalloc((void **) &dynamic_plan_.n_active_d, n_active_bytes) != cudaSuccess ||
        cudaMallocHost((void **) &dynamic_plan_.n_active_h, n_active_bytes) != cudaSuccess) {
        free_dynamic_plan_();
        return false;
    }
    if (cudaMalloc((void **) &dynamic_plan_.dp_prev_d, dp_bytes) != cudaSuccess ||
        cudaMalloc((void **) &dynamic_plan_.dp_cur_d,  dp_bytes) != cudaSuccess ||
        cudaMalloc((void **) &dynamic_plan_.trace_d,   trace_bytes) != cudaSuccess) {
        free_dynamic_plan_();
        return false;
    }
    return true;
}

SlotScratch * MoEAnyBcqExecutor::acquire_slot_scratch_(
    unsigned long long stream_key,
    int                n_tokens,
    int                n_used,
    int                M_gate_up,
    int                M_down)
{
    // slot_a is reused as gate-output (M=n_ff) and then as the
    // down-output's M=hidden_dim buffer that feeds weighted reduce.
    // Size for the larger of those Ms.
    const size_t a_M_max = (size_t) std::max(M_gate_up, M_down);
    const size_t a_bytes =
        (size_t) n_tokens * (size_t) n_used * a_M_max * sizeof(float);
    // slot_b holds up-output (M=n_ff) → SwiGLU writes silu(a)*b → b
    // → b feeds down matmul as F32 src1 [n_ff, n_used, n_tokens].
    const size_t b_bytes =
        (size_t) n_tokens * (size_t) n_used * (size_t) M_gate_up *
        sizeof(float);

    std::lock_guard<std::mutex> lk(slot_scratch_mu_);
    SlotScratch & ss = slot_scratch_[stream_key];
    if (ss.slot_a_bytes < a_bytes) {
        if (ss.slot_a) { cudaFree(ss.slot_a); ss.slot_a = nullptr; }
        if (cudaMalloc(&ss.slot_a, a_bytes) != cudaSuccess) {
            std::fprintf(stderr,
                "DPMoE: slot_scratch slot_a cudaMalloc(%zu) failed\n",
                a_bytes);
            ss.slot_a = nullptr;
            return nullptr;
        }
        ss.slot_a_bytes = a_bytes;
    }
    if (ss.slot_b_bytes < b_bytes) {
        if (ss.slot_b) { cudaFree(ss.slot_b); ss.slot_b = nullptr; }
        if (cudaMalloc(&ss.slot_b, b_bytes) != cudaSuccess) {
            std::fprintf(stderr,
                "DPMoE: slot_scratch slot_b cudaMalloc(%zu) failed\n",
                b_bytes);
            ss.slot_b = nullptr;
            return nullptr;
        }
        ss.slot_b_bytes = b_bytes;
    }
    return &ss;
}

bool MoEAnyBcqExecutor::validate_required_set_(
    const ChunkPlan & plan,
    StreamHandle      stream) const
{
    if (rt_ == nullptr) return false;
    if (plan.required_set.empty()) return true;

    const auto & pool = rt_->pool();
    int n_miss = 0;
    const ChunkKey * first_miss = nullptr;
    for (const auto & k : plan.required_set) {
        if (!pool.kernel_ready_on(k.wid, k.cid, stream)) {
            if (first_miss == nullptr) first_miss = &k;
            ++n_miss;
        }
    }
    if (n_miss == 0) return true;

    required_set_misses_.fetch_add((std::uint64_t) n_miss,
                                    std::memory_order_relaxed);

    // Build the failure message once — useful for both the
    // debug-build abort and the release-build rate-limited log.
    auto state_to_str = [](ChunkState s) -> const char * {
        switch (s) {
            case ChunkState::NOT_RESIDENT:        return "NOT_RESIDENT";
            case ChunkState::SLOT_ALLOCATED:      return "SLOT_ALLOCATED";
            case ChunkState::H2D_ISSUED:          return "H2D_ISSUED";
            case ChunkState::POINTER_TABLE_READY: return "POINTER_TABLE_READY";
        }
        return "?";
    };
    const ChunkState first_state =
        pool.chunk_state(first_miss->wid, first_miss->cid);
    char buf[512];
    std::snprintf(buf, sizeof(buf),
        "DPMoE: required_set residency violation — "
        "%d/%zu chunks not kernel-ready before launch "
        "(first miss: wid=%s cid=%d state=%s required >= POINTER_TABLE_READY)",
        n_miss, plan.required_set.size(),
        first_miss->wid.c_str(), first_miss->cid,
        state_to_str(first_state));

#ifndef NDEBUG
    GGML_ABORT("%s", buf);
#else
    // Release-build: rate-limit to once per wid via a small static set.
    static std::unordered_set<std::string> seen;
    static std::mutex                      seen_mu;
    {
        std::lock_guard<std::mutex> lk(seen_mu);
        if (seen.insert(first_miss->wid).second) {
            std::fprintf(stderr, "%s\n", buf);
        }
    }

    // Verbose per-violation dump for the T5 race investigation.
    // Gated by DP_MOE_VIOLATION_DUMP=1 to avoid log spam in
    // normal runs.  Reveals whether the miss is:
    //   - a load_set / required_set mismatch (required cid not in load_set)
    //   - a wait/H2D timing issue (load_set has it but state < ready)
    //   - an eviction race (resident earlier, NOT_RESIDENT now)
    static const bool dump_enabled =
        std::getenv("DP_MOE_VIOLATION_DUMP") != nullptr;
    if (dump_enabled) {
        static std::mutex dump_mu;
        std::lock_guard<std::mutex> lk(dump_mu);
        std::fprintf(stderr, "  >>> dump (first_miss wid=%s):\n",
                     first_miss->wid.c_str());

        // Build load_set membership lookup keyed by full ChunkKey.
        std::unordered_set<std::string> in_load_set;
        in_load_set.reserve(plan.load_set.size());
        for (const auto & k : plan.load_set) {
            in_load_set.insert(k.wid + "#" + std::to_string(k.cid));
        }

        // Tally per-state for the missed chunks + record which were
        // in load_set vs not.
        int missed_NR = 0, missed_SA = 0, missed_H2D = 0, missed_PTR = 0;
        int missed_in_load_set = 0, missed_not_in_load_set = 0;
        for (const auto & k : plan.required_set) {
            if (pool.kernel_ready_on(k.wid, k.cid, stream)) continue;
            const ChunkState s = pool.chunk_state(k.wid, k.cid);
            switch (s) {
                case ChunkState::NOT_RESIDENT:        ++missed_NR;  break;
                case ChunkState::SLOT_ALLOCATED:      ++missed_SA;  break;
                case ChunkState::H2D_ISSUED:          ++missed_H2D; break;
                case ChunkState::POINTER_TABLE_READY: ++missed_PTR; break;
            }
            const std::string key = k.wid + "#" + std::to_string(k.cid);
            if (in_load_set.count(key)) ++missed_in_load_set;
            else                        ++missed_not_in_load_set;
        }
        std::fprintf(stderr,
            "      missed by state:  NOT_RESIDENT=%d SLOT_ALLOCATED=%d "
            "H2D_ISSUED=%d POINTER_TABLE_READY=%d\n",
            missed_NR, missed_SA, missed_H2D, missed_PTR);
        std::fprintf(stderr,
            "      missed vs load_set:  in_load=%d  not_in_load=%d "
            "(load_set size=%zu, required_set size=%zu)\n",
            missed_in_load_set, missed_not_in_load_set,
            plan.load_set.size(), plan.required_set.size());

        // Print the full required_set for the first_miss's expert
        // (synthetic = "<canonical>:e<eid>") with per-cid state +
        // in_load_set flag. This is the per-expert column dump.
        const std::string focus_wid = first_miss->wid;
        std::fprintf(stderr, "      per-cid for %s:\n", focus_wid.c_str());
        for (const auto & k : plan.required_set) {
            if (k.wid != focus_wid) continue;
            const ChunkState s = pool.chunk_state(k.wid, k.cid);
            const std::string key = k.wid + "#" + std::to_string(k.cid);
            const bool in_load = in_load_set.count(key) > 0;
            const bool ready   = pool.kernel_ready_on(k.wid, k.cid, stream);
            std::fprintf(stderr,
                "        cid=%d state=%s in_load=%d ready=%d\n",
                k.cid, state_to_str(s),
                in_load ? 1 : 0, ready ? 1 : 0);
        }
    }
    return false;
#endif
}

// Batched per-layer dispatch.  See header doc.
//
// Pulls pre_inputs / sync / plan / reserve / load / wait / validate up
// to a single coalesced phase across all three canonicals before
// running the gate / up / SwiGLU / down / reduce kernel chain.  This
// is the only dispatch path; cap-pressure measurements showed the
// per-canonical fallback hurt TPS at every cap level worth
// supporting.
//
// Correctness gates:
//   - validate_required_set_ is called for each of the three plans;
//     any miss takes the release-then-return-false path.
//   - required_set reservation covers the *union* across gate/up/down
//     for the full layer; release happens at the very end so a victim
//     can't be picked from any of the three sets while down is still
//     reading its chunks.
//   - load_set submission is keyed on (wid, cid).  Gate / up / down
//     have distinct synthetic wids by construction, so the per-plan
//     load_sets don't collide; the pool's resident-skip handles any
//     overlap defensively.
bool MoEAnyBcqExecutor::dispatch_three_canonicals_(
    qwen3::MoEMatMulComp & gate_comp,
    qwen3::MoEMatMulComp & up_comp,
    qwen3::MoEMatMulComp & down_comp,
    const std::string &    gate_canonical,
    const std::string &    up_canonical,
    const std::string &    down_canonical,
    StreamHandle           stream_h,
    const ggml_tensor *    cur_3d,
    const ggml_tensor *    gated_3d,
    const ggml_tensor *    ids,
    const ggml_tensor *    probs,
    const ggml_tensor *    weights,
    const ggml_tensor *    /*layer_in*/,
    ggml_tensor *          layer_out,
    void *                 slot_a_data,
    void *                 slot_b_data,
    int                    n_tokens,
    int                    n_used,
    int                    n_expert_in_probs,
    int                    n_ff,
    int                    n_embd,
    int                    layer_idx)
{
    cudaStream_t stream = (cudaStream_t) stream_h;
    const bool decode_phase = n_tokens == 1;
    launch_diag::set_current_decode_phase(decode_phase);
    qwen3::scheduler_note_moe_layer(rt_->scheduler(), decode_phase, layer_idx);

    auto set_in = [&](qwen3::MoEInput & io,
                       const ggml_tensor * src1,
                       bool sx,
                       int K, int M) {
        io.src0              = nullptr;
        io.src1              = src1;
        io.ids               = ids;
        io.probs             = probs;
        io.weights           = weights;
        io.K                 = K;
        io.M                 = M;
        io.n_tokens          = n_tokens;
        io.n_used_per_tok    = n_used;
        io.n_expert_in_probs = n_expert_in_probs;
        io.shared_x          = sx;
        io.layer_index       = layer_idx;
    };

    qwen3::MoEInput in_gate, in_up, in_down;
    set_in(in_gate, cur_3d,   /*shared_x=*/true,  gate_comp.K(), gate_comp.M());
    set_in(in_up,   cur_3d,   /*shared_x=*/true,  up_comp.K(),   up_comp.M());
    set_in(in_down, gated_3d, /*shared_x=*/false, down_comp.K(), down_comp.M());

    // Synthesize per-canonical dst views.
    ggml_tensor dst_gate{};
    dst_gate.type = GGML_TYPE_F32;
    dst_gate.data = slot_a_data;
    ggml_tensor dst_up{};
    dst_up.type = GGML_TYPE_F32;
    dst_up.data = slot_b_data;
    ggml_tensor dst_down{};
    dst_down.type = GGML_TYPE_F32;
    dst_down.data = slot_a_data;  // reuses slot_a (down output target)

    qwen3::MoEOutput out_gate{}; out_gate.dst = &dst_gate;
    qwen3::MoEOutput out_up{};   out_up.dst   = &dst_up;
    qwen3::MoEOutput out_down{}; out_down.dst = &dst_down;

    struct CanonRef {
        qwen3::MoEMatMulComp * comp;
        qwen3::MoEInput *      in;
        const char *           name;
    };
    CanonRef canon[3] = {
        {&gate_comp, &in_gate, "gate"},
        {&up_comp,   &in_up,   "up"  },
        {&down_comp, &in_down, "down"},
    };

    const auto _t_disp_start = std::chrono::steady_clock::now();

    // Capture-mode fast path: every chunk is already pinned (scheduler
    // asserted this at install) so the load_set is permanently empty,
    // and the only remaining host dependency is the per-expert chunk-
    // count for the GEMV kernel. Compute that ON-DEVICE via
    // launch_plan_per_expert_planes — no D2H, no cudaStreamSynchronize,
    // no host plan, no reservation. Phase 0a-0f collapse into one
    // kernel launch per canonical.
    //
    // Runtime validation, including DP_MOE_PIN_ALL=1 full-resident
    // validation, intentionally does NOT take this path. Uniform and
    // dynamic both keep the GPU-plan -> CPU residency/load roundtrip so
    // their validation semantics match streaming runtime mode.
    const bool capture_path =
        qwen3::scheduler_allow_capture(rt_->scheduler());

    ChunkPlan plan_gate, plan_up, plan_down;

    auto release_all = [&]() {
        for (const auto & k : plan_gate.required_set) {
            qwen3::scheduler_release_from_dispatch(
                rt_->scheduler(), k.wid, k.cid);
        }
        for (const auto & k : plan_up.required_set) {
            qwen3::scheduler_release_from_dispatch(
                rt_->scheduler(), k.wid, k.cid);
        }
        for (const auto & k : plan_down.required_set) {
            qwen3::scheduler_release_from_dispatch(
                rt_->scheduler(), k.wid, k.cid);
        }
    };

    if (capture_path) {
        // Set per-canonical dims (cur_n_tokens_ / cur_n_used_ / etc.)
        // by calling pre_inputs and discarding its MemcpySpec list. We
        // don't issue any D2H — but execute() short-circuits when the
        // dims aren't set, so this side effect is required. The plan
        // kernel itself runs inside execute() after the ids D2D copy.
        for (auto & c : canon) {
            (void) c.comp->pre_inputs(*c.in);
        }
    } else {
        bool plans_ready = false;

        const char * gpu_plan_env = std::getenv("DP_MOE_DYNAMIC_GPU_PLANNER");
        const bool gpu_plan_disabled =
            gpu_plan_env != nullptr && gpu_plan_env[0] == '0';
        const float dyn_kbar = qwen3::scheduler_kbar(rt_->scheduler());
        const int K_min_dyn  = qwen3::scheduler_dynamic_K_min(rt_->scheduler());
        const int K_max_dyn  = qwen3::scheduler_dynamic_K_max(rt_->scheduler());
        const float * R_d    = qwen3::scheduler_dynamic_R_device(rt_->scheduler());
        const auto allocator_mode =
            qwen3::scheduler_kbar_allocator_mode(rt_->scheduler());
        const bool can_gpu_common =
            !gpu_plan_disabled &&
            dyn_kbar > 0.0f &&
            ids != nullptr && ids->data != nullptr &&
            n_expert_in_probs > 0 &&
            layer_idx >= 0;
        const bool use_profile_gpu_plan =
            can_gpu_common &&
            allocator_mode == qwen3::KBarAllocatorMode::Profile &&
            R_d != nullptr &&
            K_min_dyn >= 0 && K_max_dyn >= K_min_dyn;
        const bool use_uniform_gpu_plan =
            can_gpu_common &&
            allocator_mode == qwen3::KBarAllocatorMode::Uniform;
        const bool can_gpu_plan =
            use_profile_gpu_plan || use_uniform_gpu_plan;

        if (can_gpu_plan) {
            for (auto & c : canon) {
                (void) c.comp->pre_inputs(*c.in);
            }

            const int plan_extra_max = use_profile_gpu_plan
                ? std::max(0, K_max_dyn - K_min_dyn)
                : 0;
            const int B1 = n_expert_in_probs * plan_extra_max + 1;
            if (ensure_dynamic_plan_(n_expert_in_probs, B1)) {
                const size_t ids_row_stride = (size_t) ids->nb[1];
                size_t weights_row_stride = 0;
                if (weights != nullptr) {
                    weights_row_stride =
                        (weights->ne[0] == 1 && weights->ne[1] == n_used)
                        ? (size_t) weights->nb[2]
                        : (size_t) weights->nb[1];
                }
                const size_t probs_row_stride =
                    probs != nullptr ? (size_t) probs->nb[1] : 0;

                const bool profile_planner = planner_profile_enabled();
                cudaEvent_t planner_ev_start = nullptr;
                cudaEvent_t planner_ev_stop  = nullptr;
                bool planner_events_ready = false;
                if (profile_planner) {
                    planner_events_ready =
                        cudaEventCreate(&planner_ev_start) == cudaSuccess &&
                        cudaEventCreate(&planner_ev_stop) == cudaSuccess;
                    if (!planner_events_ready) {
                        if (planner_ev_start != nullptr) cudaEventDestroy(planner_ev_start);
                        if (planner_ev_stop  != nullptr) cudaEventDestroy(planner_ev_stop);
                        planner_ev_start = nullptr;
                        planner_ev_stop  = nullptr;
                    }
                }
                const auto planner_roundtrip_t0 = std::chrono::steady_clock::now();
                if (planner_events_ready) {
                    cudaEventRecord(planner_ev_start, stream);
                }
                const auto planner_launch_t0 = std::chrono::steady_clock::now();
                if (use_profile_gpu_plan) {
                    qwen3::launch_plan_chunks_kbar_exact_strided(
                        (const int32_t *) ids->data,
                        ids_row_stride,
                        weights != nullptr ? (const float *) weights->data : nullptr,
                        weights_row_stride,
                        probs != nullptr ? (const float *) probs->data : nullptr,
                        probs_row_stride,
                        R_d,
                        K_min_dyn,
                        K_max_dyn,
                        layer_idx,
                        dyn_kbar,
                        n_tokens,
                        n_used,
                        n_expert_in_probs,
                        gate_comp.n_chunks_max(),
                        dynamic_plan_.chunks_d,
                        dynamic_plan_.expert_order_d,
                        dynamic_plan_.n_active_d,
                        dynamic_plan_.dp_prev_d,
                        dynamic_plan_.dp_cur_d,
                        dynamic_plan_.trace_d,
                        stream_h);
                } else {
                    int uniform_k = (int) std::lrint((double) dyn_kbar);
                    if (uniform_k < 1) uniform_k = 1;
                    if (uniform_k > gate_comp.n_chunks_max()) {
                        uniform_k = gate_comp.n_chunks_max();
                    }
                    qwen3::launch_plan_chunks_uniform_strided(
                        (const int32_t *) ids->data,
                        ids_row_stride,
                        n_tokens,
                        n_used,
                        n_expert_in_probs,
                        uniform_k,
                        gate_comp.n_chunks_max(),
                        dynamic_plan_.chunks_d,
                        dynamic_plan_.expert_order_d,
                        dynamic_plan_.n_active_d,
                        stream_h);
                }
                const auto planner_launch_t1 = std::chrono::steady_clock::now();
                if (profile_planner) {
                    launch_diag::note_planner_launch_host(
                        decode_phase,
                        elapsed_ns(planner_launch_t0, planner_launch_t1));
                }
                if (planner_events_ready) {
                    cudaEventRecord(planner_ev_stop, stream);
                }
                launch_diag::note_launch(launch_diag::Kind::MemcpyAsync);
                cudaMemcpyAsync(dynamic_plan_.chunks_h,
                                dynamic_plan_.chunks_d,
                                (size_t) n_expert_in_probs * sizeof(int),
                                cudaMemcpyDeviceToHost,
                                stream);
                launch_diag::note_launch(launch_diag::Kind::MemcpyAsync);
                cudaMemcpyAsync(dynamic_plan_.expert_order_h,
                                dynamic_plan_.expert_order_d,
                                (size_t) n_expert_in_probs * sizeof(int),
                                cudaMemcpyDeviceToHost,
                                stream);
                launch_diag::note_launch(launch_diag::Kind::MemcpyAsync);
                cudaMemcpyAsync(dynamic_plan_.n_active_h,
                                dynamic_plan_.n_active_d,
                                sizeof(int),
                                cudaMemcpyDeviceToHost,
                                stream);
                launch_diag::note_stream_sync();
                const cudaError_t sync_err = cudaStreamSynchronize(stream);
                const auto planner_roundtrip_t1 = std::chrono::steady_clock::now();
                if (profile_planner) {
                    launch_diag::note_planner_roundtrip_host(
                        decode_phase,
                        elapsed_ns(planner_roundtrip_t0, planner_roundtrip_t1));
                    if (planner_events_ready && sync_err == cudaSuccess) {
                        float planner_ms = 0.0f;
                        if (cudaEventElapsedTime(&planner_ms,
                                                 planner_ev_start,
                                                 planner_ev_stop) == cudaSuccess) {
                            launch_diag::note_planner_kernel_event(
                                decode_phase,
                                (uint64_t) (planner_ms * 1000000.0f));
                        }
                    }
                }
                if (planner_ev_start != nullptr) cudaEventDestroy(planner_ev_start);
                if (planner_ev_stop  != nullptr) cudaEventDestroy(planner_ev_stop);
                if (sync_err == cudaSuccess) {
                    int n_external_order =
                        dynamic_plan_.n_active_h != nullptr ?
                        dynamic_plan_.n_active_h[0] : 0;
                    if (n_external_order < 0) n_external_order = 0;
                    if (n_external_order > n_expert_in_probs) {
                        n_external_order = n_expert_in_probs;
                    }
                    if (const char * cmp_env =
                            use_profile_gpu_plan
                                ? std::getenv("DP_MOE_COMPARE_GPU_CPU_PLANNER")
                                : nullptr) {
                        if (cmp_env[0] && cmp_env[0] != '0') {
                            std::vector<int32_t> ids_h(
                                (size_t) n_tokens * (size_t) n_used);
                            cudaMemcpy2D(ids_h.data(),
                                         (size_t) n_used * sizeof(int32_t),
                                         ids->data, ids_row_stride,
                                         (size_t) n_used * sizeof(int32_t),
                                         (size_t) n_tokens,
                                         cudaMemcpyDeviceToHost);

                            std::vector<float> weights_h;
                            if (weights != nullptr && weights->data != nullptr) {
                                weights_h.resize(
                                    (size_t) n_tokens * (size_t) n_used);
                                cudaMemcpy2D(weights_h.data(),
                                             (size_t) n_used * sizeof(float),
                                             weights->data, weights_row_stride,
                                             (size_t) n_used * sizeof(float),
                                             (size_t) n_tokens,
                                             cudaMemcpyDeviceToHost);
                            }

                            std::vector<float> probs_h;
                            if (weights_h.empty() &&
                                probs != nullptr && probs->data != nullptr) {
                                probs_h.resize((size_t) n_tokens *
                                               (size_t) n_expert_in_probs);
                                cudaMemcpy2D(probs_h.data(),
                                             (size_t) n_expert_in_probs * sizeof(float),
                                             probs->data, probs_row_stride,
                                             (size_t) n_expert_in_probs * sizeof(float),
                                             (size_t) n_tokens,
                                             cudaMemcpyDeviceToHost);
                            }

                            std::unordered_map<int, float> gate_sq_sum_by_expert;
                            std::vector<int> unique_experts;
                            gate_sq_sum_by_expert.reserve((size_t) n_used * 4);
                            unique_experts.reserve((size_t) n_used * 4);
                            for (int t = 0; t < n_tokens; ++t) {
                                for (int u = 0; u < n_used; ++u) {
                                    const int eid =
                                        ids_h[(size_t) t * n_used + u];
                                    if (eid < 0 || eid >= n_expert_in_probs) {
                                        continue;
                                    }
                                    float g;
                                    if (!weights_h.empty()) {
                                        g = weights_h[(size_t) t * n_used + u];
                                    } else if (!probs_h.empty()) {
                                        g = probs_h[(size_t) t *
                                                    n_expert_in_probs + eid];
                                    } else {
                                        g = 0.45f - 0.05f * (float) u;
                                        if (g < 0.05f) g = 0.05f;
                                    }
                                    const float g2 = g * g;
                                    auto it = gate_sq_sum_by_expert.find(eid);
                                    if (it == gate_sq_sum_by_expert.end()) {
                                        gate_sq_sum_by_expert.emplace(eid, g2);
                                        unique_experts.push_back(eid);
                                    } else {
                                        it->second += g2;
                                    }
                                }
                            }

                            std::vector<float> gates(unique_experts.size());
                            for (size_t i = 0; i < unique_experts.size(); ++i) {
                                gates[i] = std::sqrt(std::max(
                                    0.0f, gate_sq_sum_by_expert[unique_experts[i]]));
                            }
                            std::vector<int> cpu_K;
                            const bool cpu_ok =
                                qwen3::scheduler_allocate_dispatch_budget(
                                    rt_->scheduler(), layer_idx,
                                    unique_experts, gates, cpu_K);
                            int mismatch = 0;
                            int first_e = -1;
                            int first_cpu = -1;
                            int first_gpu = -1;
                            long cpu_sum = 0;
                            long gpu_sum = 0;
                            std::vector<int> cpu_by_eid(
                                (size_t) n_expert_in_probs, 0);
                            if (cpu_ok) {
                                for (size_t i = 0; i < unique_experts.size(); ++i) {
                                    const int eid = unique_experts[i];
                                    cpu_by_eid[(size_t) eid] = cpu_K[i];
                                    cpu_sum += cpu_K[i];
                                }
                            }
                            for (int eid = 0; eid < n_expert_in_probs; ++eid) {
                                const int c = cpu_by_eid[(size_t) eid];
                                const int g = dynamic_plan_.chunks_h[eid];
                                gpu_sum += g;
                                if (c != g) {
                                    ++mismatch;
                                    if (first_e < 0) {
                                        first_e = eid;
                                        first_cpu = c;
                                        first_gpu = g;
                                    }
                                }
                            }
                            static std::atomic<int> cmp_prints{0};
                            const int print_idx =
                                cmp_prints.fetch_add(1, std::memory_order_relaxed);
                            if (mismatch != 0 || print_idx < 8) {
                                std::fprintf(stderr,
                                    "dp_moe-planner-compare L=%d tok=%d "
                                    "active=%zu cpu_ok=%d mismatch=%d "
                                    "cpu_sum=%ld gpu_sum=%ld first_e=%d "
                                    "cpu=%d gpu=%d strides ids=%zu weights=%zu "
                                    "probs=%zu\n",
                                    layer_idx, n_tokens, unique_experts.size(),
                                    cpu_ok ? 1 : 0, mismatch, cpu_sum, gpu_sum,
                                    first_e, first_cpu, first_gpu,
                                    ids_row_stride, weights_row_stride,
                                    probs_row_stride);
                            }
                        }
                    }
                    gate_comp.use_external_chunk_plan(
                        dynamic_plan_.chunks_h,
                        dynamic_plan_.expert_order_h,
                        n_external_order);
                    up_comp.use_external_chunk_plan(
                        dynamic_plan_.chunks_h,
                        dynamic_plan_.expert_order_h,
                        n_external_order);
                    down_comp.use_external_chunk_plan(
                        dynamic_plan_.chunks_h,
                        dynamic_plan_.expert_order_h,
                        n_external_order);
                    plan_gate = gate_comp.plan(in_gate);
                    plan_up   = up_comp.plan(in_up);
                    plan_down = down_comp.plan(in_down);
                    gate_comp.clear_external_chunk_plan();
                    up_comp.clear_external_chunk_plan();
                    down_comp.clear_external_chunk_plan();
                    plans_ready = true;
                } else {
                    std::fprintf(stderr,
                        "DPMoE: dynamic GPU planner failed at L=%d: %s; "
                        "falling back to host planning\n",
                        layer_idx, cudaGetErrorString(sync_err));
                }
            }
        }

        if (!plans_ready) {
        // ── Phase 0a: enqueue pre_inputs D2Hs for all three canonicals.
        //    Each canonical's pre_inputs writes to its own pinned arrays;
        //    we're issuing 3× D2H_ids + 3× D2H_probs + 3× D2H_weights but
        //    serialised on one stream and followed by ONE sync.
        {
            launch_diag::PhaseTimer _pt(
                launch_diag::Phase::PreInputsD2H, decode_phase);
            for (auto & c : canon) {
                for (auto & m : c.comp->pre_inputs(*c.in)) {
                    if (m.bytes == 0) continue;
                    if (m.is_2d) {
                        launch_diag::note_launch(launch_diag::Kind::Memcpy2DAsync);
                        cudaMemcpy2DAsync(m.host_dst, m.dst_pitch,
                            m.device_src, m.src_pitch,
                            m.bytes, m.height,
                            cudaMemcpyDeviceToHost, stream);
                    } else {
                        launch_diag::note_launch(launch_diag::Kind::MemcpyAsync);
                        cudaMemcpyAsync(m.host_dst, m.device_src, m.bytes,
                            cudaMemcpyDeviceToHost, stream);
                    }
                }
            }

            // ── Phase 0b: single sync — host now sees the routed ids/probs/
            //    weights for every comp.
            launch_diag::note_stream_sync();
            cudaStreamSynchronize(stream);
        }

        // ── Phase 0c: plan all three.
        {
            launch_diag::PhaseTimer _pt(
                launch_diag::Phase::Plan, decode_phase);
            plan_gate = gate_comp.plan(in_gate);
            plan_up   = up_comp.plan(in_up);
            plan_down = down_comp.plan(in_down);
        }
        }

        auto count_residency = [&](const ChunkPlan & pl) {
            if (pl.required_set.empty()) return;
            size_t hits = 0;
            static std::mutex cache_trace_mu;
            static FILE * cache_trace_fp = []() -> FILE * {
                const char * path = std::getenv("DP_MOE_CACHE_TRACE_FILE");
                if (path == nullptr || path[0] == '\0') return nullptr;
                FILE * fp = std::fopen(path, "w");
                if (fp != nullptr) {
                    std::fprintf(fp, "#seq phase cid bytes resident wid\n");
                }
                return fp;
            }();
            static std::atomic<uint64_t> cache_trace_seq{0};
            for (const auto & k : pl.required_set) {
                const bool resident = rt_->pool().is_resident(k.wid, k.cid);
                if (resident) ++hits;
                launch_diag::note_required_chunk(n_tokens == 1,
                                                 k.cid, resident);
                if (cache_trace_fp != nullptr) {
                    size_t bytes = 0;
                    if (cid_is_chunk(k.cid)) {
                        ChunkedTensor * tensor = rt_->tensor(k.wid);
                        const int p = cid_chunk_index(k.cid);
                        if (tensor != nullptr && p >= 0) {
                            bytes = tensor->kernel_bytes(p);
                        }
                    }
                    const uint64_t seq = cache_trace_seq.fetch_add(
                        1, std::memory_order_relaxed);
                    std::lock_guard<std::mutex> lk(cache_trace_mu);
                    std::fprintf(cache_trace_fp, "%lu %c %d %zu %d %s\n",
                                 (unsigned long)seq,
                                 n_tokens == 1 ? 'D' : 'P',
                                 k.cid, bytes, resident ? 1 : 0,
                                 k.wid.c_str());
                }
            }
            rt_->pool().note_required_set_phase(pl.required_set.size(), hits,
                                                n_tokens == 1);
        };
        count_residency(plan_gate);
        count_residency(plan_up);
        count_residency(plan_down);

        // ── Phase 0d: reserve the union of all three required_sets.
        auto reserve_plan = [&](const ChunkPlan & pl) {
            for (const auto & k : pl.required_set) {
                qwen3::scheduler_reserve_for_dispatch(
                    rt_->scheduler(), k.wid, k.cid);
            }
        };
        {
            launch_diag::PhaseTimer _pt(
                launch_diag::Phase::Reserve, decode_phase);
            reserve_plan(plan_gate);
            reserve_plan(plan_up);
            reserve_plan(plan_down);
        }

        // ── Phase 0e: submit the union of all three load_sets as a
        //    single batch.  Gate/up/down canonical names are distinct
        //    (e.g. "blk.0.ffn_gate_exps.weight:e5" vs ".ffn_up_..." vs
        //    ".ffn_down_..."), so cross-plan dedupe isn't needed —
        //    each plan's own ``issued_keys`` already removed intra-
        //    plan dupes in the comp.plan code.  The pool is_resident
        //    probe still short-circuits chunks that landed in a prior
        //    layer.
        const size_t union_size =
            plan_gate.load_set.size() +
            plan_up.load_set.size() +
            plan_down.load_set.size();

        if (union_size > 0) {
            launch_diag::note_load_set(union_size);
            const char * batch_env = std::getenv("DP_MOE_BATCH_CHUNK_LOAD");
            const bool batch_on =
                batch_env != nullptr && batch_env[0] && batch_env[0] != '0';
            launch_diag::PhaseTimer _pt(
                launch_diag::Phase::Load, decode_phase);
            if (batch_on) {
                std::vector<ChunkKey> missing;
                missing.reserve(union_size);
                std::unordered_set<ChunkKey, ChunkKeyHash> seen;
                auto collect_plan = [&](const ChunkPlan & pl) {
                    for (const auto & k : pl.load_set) {
                        if (rt_->pool().is_resident(k.wid, k.cid)) continue;
                        if (seen.insert(k).second) {
                            missing.push_back(k);
                            launch_diag::note_unique_load_chunk(
                                n_tokens == 1, k.cid);
                        }
                    }
                };
                collect_plan(plan_gate);
                collect_plan(plan_up);
                collect_plan(plan_down);
                const char * bundle_env = std::getenv("DP_MOE_EXPERT_BUNDLE");
                const bool expert_bundle_on =
                    bundle_env != nullptr && bundle_env[0] && bundle_env[0] != '0';
                if (expert_bundle_on && missing.size() > 1) {
                    struct BundleOrder {
                        int layer  = 1 << 29;
                        int expert = 1 << 29;
                        int cid    = 1 << 29;
                        int kind   = 1 << 29;
                    };
                    auto order_for = [](const ChunkKey & k) {
                        BundleOrder o;
                        if (k.cid >= kCidChunkBase) {
                            o.cid = k.cid - kCidChunkBase;
                        }
                        const char * s = k.wid.c_str();
                        if (std::strncmp(s, "blk.", 4) == 0) {
                            char * end = nullptr;
                            const long layer = std::strtol(s + 4, &end, 10);
                            if (end != s + 4) o.layer = (int)layer;
                        }
                        if (k.wid.find(".ffn_gate_exps.weight") != std::string::npos) {
                            o.kind = 0;
                        } else if (k.wid.find(".ffn_up_exps.weight") != std::string::npos) {
                            o.kind = 1;
                        } else if (k.wid.find(".ffn_down_exps.weight") != std::string::npos) {
                            o.kind = 2;
                        }
                        const size_t epos = k.wid.rfind(":e");
                        if (epos != std::string::npos && epos + 2 < k.wid.size()) {
                            char * end = nullptr;
                            const long expert =
                                std::strtol(k.wid.c_str() + epos + 2, &end, 10);
                            if (end != k.wid.c_str() + epos + 2) {
                                o.expert = (int)expert;
                            }
                        }
                        return o;
                    };
                    std::stable_sort(
                        missing.begin(), missing.end(),
                        [&](const ChunkKey & a, const ChunkKey & b) {
                            const BundleOrder oa = order_for(a);
                            const BundleOrder ob = order_for(b);
                            if (oa.layer  != ob.layer)  return oa.layer  < ob.layer;
                            if (oa.expert != ob.expert) return oa.expert < ob.expert;
                            if (oa.cid    != ob.cid)    return oa.cid    < ob.cid;
                            if (oa.kind   != ob.kind)   return oa.kind   < ob.kind;
                            if (a.cid != b.cid) return a.cid < b.cid;
                            return a.wid < b.wid;
                        });
                }
                if (!missing.empty()) {
                    rt_->move_chunks_batch(missing, stream_h);
                }
            } else if (rt_->io_worker_count() > 0) {
                auto batch = std::make_shared<std::atomic<uint32_t>>(0);
                auto submit_plan = [&](const ChunkPlan & pl) {
                    for (const auto & k : pl.load_set) {
                        if (rt_->pool().is_resident(k.wid, k.cid)) continue;
                        rt_->submit_async_load(k.wid, k.cid, batch, stream_h);
                    }
                };
                submit_plan(plan_gate);
                submit_plan(plan_up);
                submit_plan(plan_down);
                launch_diag::note_async_load_wait();
                rt_->wait_async_load_batch(batch);
            } else {
                auto move_plan = [&](const ChunkPlan & pl) {
                    for (const auto & k : pl.load_set) {
                        if (rt_->pool().is_resident(k.wid, k.cid)) continue;
                        rt_->move_chunk(k.wid, k.cid, Tier::RAM, Tier::VRAM,
                                        stream_h);
                    }
                };
                move_plan(plan_gate);
                move_plan(plan_up);
                move_plan(plan_down);
            }
            if (!batch_on) {
                auto wait_plan = [&](const ChunkPlan & pl) {
                    for (const auto & k : pl.load_set) {
                        rt_->pool().wait_on_stream(k.wid, k.cid, stream_h);
                    }
                };
                wait_plan(plan_gate);
                wait_plan(plan_up);
                wait_plan(plan_down);
            }
        }

        // ── Phase 0f: validate all three.
        bool valid = false;
        {
            launch_diag::PhaseTimer _pt(
                launch_diag::Phase::Validate, decode_phase);
            valid = validate_required_set_(plan_gate, stream_h) &&
                    validate_required_set_(plan_up,   stream_h) &&
                    validate_required_set_(plan_down, stream_h);
        }
        if (!valid)
        {
            release_all();
            std::fprintf(stderr,
                "DPMoE: forward_moe_layer[L=%d] (batched): required_set "
                "validation failed (gate=%s up=%s down=%s)\n",
                layer_idx, gate_canonical.c_str(),
                up_canonical.c_str(), down_canonical.c_str());
            return false;
        }
    }  // !capture_path

    // Record dispatch time once for the whole batched phase, but
    // attribute it 3-way to keep the per-dispatch average comparable
    // to the per-canonical path.
    {
        const auto dt = std::chrono::steady_clock::now() - _t_disp_start;
        const uint64_t ns_total = (uint64_t) std::chrono::duration_cast<
            std::chrono::nanoseconds>(dt).count();
        launch_diag::note_dispatch_one_canonical(ns_total / 3);
        launch_diag::note_dispatch_one_canonical(ns_total / 3);
        launch_diag::note_dispatch_one_canonical(ns_total - 2 * (ns_total / 3));
    }

    // Diagnostic dump hook (DP_MOE_DEBUG_MOE_LAYER=<L>[,<L>...]):
    // accepts a comma-separated list of layer indices.  On the FIRST
    // call for EACH target layer, write full-vector binary dumps for
    // each compute phase under /tmp/dbg_moe_L<L>_<tag>.bin.  Plus a
    // stderr summary line per tag.  Per-layer atomic guards so each
    // target fires exactly once per process.
    const char * dbg_env = std::getenv("DP_MOE_DEBUG_MOE_LAYER");
    bool dbg_matches_this_layer = false;
    if (dbg_env != nullptr) {
        const char * p = dbg_env;
        while (*p) {
            char * end = nullptr;
            long v = std::strtol(p, &end, 10);
            if (end == p) break;
            if ((int) v == layer_idx) { dbg_matches_this_layer = true; break; }
            p = end;
            while (*p == ',' || *p == ' ') ++p;
        }
    }
    // Per-layer once-guard via 64-element static array (covers up to
    // 64 MoE layers — Qwen3-30B has 48).
    static std::atomic<bool> dbg_dumped_per_layer[64] = {};
    bool do_dump = false;
    if (dbg_matches_this_layer && layer_idx >= 0 && layer_idx < 64) {
        do_dump = !dbg_dumped_per_layer[layer_idx].exchange(true);
    }
    auto dump_buf = [&](const char * tag, const void * dev, int n_floats) {
        std::vector<float> hbuf((size_t) n_floats, 0.0f);
        cudaStreamSynchronize(stream);
        cudaMemcpy(hbuf.data(), dev, (size_t) n_floats * sizeof(float),
                    cudaMemcpyDeviceToHost);
        char path[256];
        std::snprintf(path, sizeof(path),
            "/tmp/dbg_moe_L%d_%s.bin", layer_idx, tag);
        FILE * fp = std::fopen(path, "wb");
        if (fp != nullptr) {
            std::fwrite(hbuf.data(), sizeof(float), (size_t) n_floats, fp);
            std::fclose(fp);
        }
        std::fprintf(stderr, "[dbg L=%d] %-20s n=%d  head=[", layer_idx, tag, n_floats);
        const int show = n_floats < 8 ? n_floats : 8;
        for (int i = 0; i < show; ++i) std::fprintf(stderr, " %+ .6e", hbuf[i]);
        std::fprintf(stderr, " ]  → %s\n", path);
    };
    auto dump_ids = [&](const char * tag, const void * dev, int n_ints) {
        std::vector<int32_t> hbuf((size_t) n_ints, 0);
        cudaStreamSynchronize(stream);
        cudaMemcpy(hbuf.data(), dev, (size_t) n_ints * sizeof(int32_t),
                    cudaMemcpyDeviceToHost);
        char path[256];
        std::snprintf(path, sizeof(path),
            "/tmp/dbg_moe_L%d_%s.bin", layer_idx, tag);
        FILE * fp = std::fopen(path, "wb");
        if (fp != nullptr) {
            std::fwrite(hbuf.data(), sizeof(int32_t), (size_t) n_ints, fp);
            std::fclose(fp);
        }
        std::fprintf(stderr, "[dbg L=%d] %-20s n=%d  [", layer_idx, tag, n_ints);
        for (int i = 0; i < n_ints; ++i) std::fprintf(stderr, " %d", hbuf[i]);
        std::fprintf(stderr, " ]  → %s\n", path);
    };
    if (do_dump) {
        std::fprintf(stderr,
            "[dbg L=%d] forward_moe_layer entry: n_tokens=%d n_used=%d "
            "n_ff=%d n_embd=%d\n",
            layer_idx, n_tokens, n_used, n_ff, n_embd);
        // Full vectors for token 0, slot 0 — enough to reconstruct
        // the per-expert compute in Python.
        dump_buf("layer_in_t0",    cur_3d->data, n_embd);
        dump_ids("ids_t0",         ids->data, n_used);
        dump_buf("weights_t0",     weights->data, n_used);
    }

    // ── Phase 1+2: Gate + Up matmul, each as its own launch.
    {
        launch_diag::PhaseTimer _pt(
            launch_diag::Phase::Execute, decode_phase);
        gate_comp.execute(in_gate, out_gate, stream_h);
        if (do_dump) dump_buf("gate_out_t0u0", slot_a_data, n_ff);
        up_comp.execute(in_up, out_up, stream_h);
        if (do_dump) dump_buf("up_out_t0u0",   slot_b_data, n_ff);

        // ── Phase 3: gated activation: slot_b ← act(slot_a) * slot_b.
        {
            const size_t N_swiglu =
                (size_t) n_tokens * (size_t) n_used * (size_t) n_ff;
            qwen3::launch_swiglu_mul(
                (const float *) slot_a_data,
                (const float *) slot_b_data,
                (float *)       slot_b_data,
                N_swiglu, stream_h, activation_);
        }
        if (do_dump) dump_buf("gated_t0u0",    slot_b_data, n_ff);
        // ── Phase 4: Down matmul → slot_a (reused buffer).
        down_comp.execute(in_down, out_down, stream_h);
        if (do_dump) dump_buf("down_out_t0u0", slot_a_data, n_embd);
        // ── Phase 5: weighted reduce slot_a → layer_out.
        qwen3::launch_weighted_reduce_slots(
            (const float *) slot_a_data,
            (const float *) weights->data,
            (float *)       layer_out->data,
            n_tokens, n_used, n_embd, stream_h);
        if (do_dump) dump_buf("layer_out_t0",  layer_out->data, n_embd);
    }

    // ── Phase 6: release the union of reserved chunks.
    {
        launch_diag::PhaseTimer _pt(
            launch_diag::Phase::Release, decode_phase);
        release_all();
    }
    return true;
}


// M1 cutover S6: per-layer execution-time entry point (criterion 3).
//
// Replaces the per-canonical mul_mat_id_hook dispatch with a single
// per-layer call. Routes through three MoEMatMulComps (gate / up /
// down) for layer L, runs SwiGLU between gate/up outputs, and
// collapses the per-slot down output into layer_out via the
// weighted-reduce kernel. layer_out is F32 [n_embd, n_tokens] (the
// sentinel-node dst from S5's cgraph emission).
//
// Memory layout (matches comp.execute()'s expectations):
//   slot_a  [n_tokens, n_used, n_ff]      f32 — gate output
//   slot_b  [n_tokens, n_used, n_ff]      f32 — up output → SwiGLU result
//   slot_a  [n_tokens, n_used, hidden]    f32 — down output (reused buffer)
//   layer_out [n_tokens, hidden]          f32 — weighted reduce target
bool MoEAnyBcqExecutor::forward_moe_layer(
    StreamHandle        stream_h,
    const ggml_tensor * layer_in,
    const ggml_tensor * ids,
    const ggml_tensor * probs,
    const ggml_tensor * weights,
    ggml_tensor *       layer_out,
    int                 layer_idx)
{
    const auto _t_fwd_start = std::chrono::steady_clock::now();
    struct FwdScope {
        std::chrono::steady_clock::time_point t0;
        ~FwdScope() {
            const auto dt = std::chrono::steady_clock::now() - t0;
            launch_diag::note_forward_moe_layer(
                (uint64_t) std::chrono::duration_cast<
                    std::chrono::nanoseconds>(dt).count());
        }
    } _fwd_scope{_t_fwd_start};

    if (rt_ == nullptr) return false;
    if (layer_in == nullptr || ids == nullptr) return false;
    if (probs == nullptr || weights == nullptr) return false;
    if (layer_out == nullptr || layer_out->data == nullptr) return false;
    if (layer_idx < 0) return false;

    // Dtype sanity.
    if (layer_in->type  != GGML_TYPE_F32 ||
        layer_out->type != GGML_TYPE_F32 ||
        probs->type     != GGML_TYPE_F32 ||
        weights->type   != GGML_TYPE_F32 ||
        ids->type       != GGML_TYPE_I32) {
        std::fprintf(stderr,
            "DPMoE: forward_moe_layer[L=%d] dtype mismatch — "
            "layer_in=%d ids=%d probs=%d weights=%d layer_out=%d\n",
            layer_idx,
            (int)layer_in->type, (int)ids->type, (int)probs->type,
            (int)weights->type, (int)layer_out->type);
        return false;
    }

    // Shape extraction. layer_in is 2D [n_embd, n_tokens]; ids is
    // [n_used, n_tokens]; probs is [n_expert, n_tokens]; weights is
    // [1, n_used, n_tokens] (the helper's reshape) or [n_used,
    // n_tokens] (alt branch) — pre_inputs() reads byte-count from
    // cur_n_used_*cur_n_tokens_ so either layout works.
    const int n_embd  = (int) layer_in->ne[0];
    const int n_tokens = (int) layer_in->ne[1];
    const int n_used  = (int) ids->ne[0];
    const int n_expert_in_probs = (int) probs->ne[0];
    if (n_embd <= 0 || n_tokens <= 0 || n_used <= 0) return false;

    // Resolve the three canonicals for this layer.
    const std::string gate_canonical =
        "blk." + std::to_string(layer_idx) + ".ffn_gate_exps.weight";
    const std::string up_canonical =
        "blk." + std::to_string(layer_idx) + ".ffn_up_exps.weight";
    const std::string down_canonical =
        "blk." + std::to_string(layer_idx) + ".ffn_down_exps.weight";

    auto * gate_comp = qwen3::scheduler_lookup_moe_comp(
        rt_->scheduler(), gate_canonical);
    auto * up_comp = qwen3::scheduler_lookup_moe_comp(
        rt_->scheduler(), up_canonical);
    auto * down_comp = qwen3::scheduler_lookup_moe_comp(
        rt_->scheduler(), down_canonical);
    if (gate_comp == nullptr || up_comp == nullptr || down_comp == nullptr ||
        !gate_comp->valid() || !up_comp->valid() || !down_comp->valid()) {
        GGML_ABORT(
            "DPMoE: forward_moe_layer[L=%d]: missing/invalid comp "
            "(gate=%p up=%p down=%p)",
            layer_idx, (void*)gate_comp, (void*)up_comp, (void*)down_comp);
    }

    // Canonical-shape sanity: gate.M == up.M; down.K == gate.M.
    const int n_ff   = gate_comp->M();
    const int M_gate = gate_comp->M();
    const int M_up   = up_comp->M();
    const int K_down = down_comp->K();
    const int M_down = down_comp->M();
    if (M_gate != M_up || K_down != n_ff || M_down != n_embd) {
        GGML_ABORT(
            "DPMoE: forward_moe_layer[L=%d]: shape mismatch — "
            "gate.M=%d up.M=%d down.K=%d down.M=%d n_embd=%d",
            layer_idx, M_gate, M_up, K_down, M_down, n_embd);
    }
    if (gate_comp->K() != n_embd || up_comp->K() != n_embd) {
        GGML_ABORT(
            "DPMoE: forward_moe_layer[L=%d]: input K mismatch — "
            "gate.K=%d up.K=%d n_embd=%d",
            layer_idx, gate_comp->K(), up_comp->K(), n_embd);
    }

    // Acquire / size the per-stream slot scratch.
    const unsigned long long stream_key =
        (unsigned long long) (uintptr_t) stream_h;
    SlotScratch * ss = acquire_slot_scratch_(
        stream_key, n_tokens, n_used, n_ff, n_embd);
    if (ss == nullptr) return false;

    cudaStream_t stream = (cudaStream_t) stream_h;

    // The Step-4b LayerBegin/LayerEnd instrumenter call was retired
    // in S8 along with the per-canonical layer-marker prewalk. The
    // sentinel name carries the per-layer index directly; no
    // marker-emission walk needed for Mode A.

    if (moe_dispatch::scratch_for_stream(stream) == nullptr) {
        std::fprintf(stderr,
            "DPMoE: forward_moe_layer[L=%d]: scratch alloc failed\n",
            layer_idx);
        return false;
    }
    moe_dispatch::profile_inc_hook_calls();

    // ── Synthesize src1 views.
    //
    // For gate/up the input is ``cur`` (= layer_in, 2D [n_embd,
    // n_tokens]) consumed as shared_x=true. comp.execute()'s contig
    // check reads nb[2] and compares to K*sizeof(float). For a 2D
    // tensor nb[2] == n_embd * n_tokens * 4 ≠ k_bytes_f32, so the
    // contig path would miss. Synthesize a 3D view [n_embd, 1,
    // n_tokens] with nb[2] = n_embd*4 to hit the fast path.
    ggml_tensor cur_3d{};
    cur_3d.type    = GGML_TYPE_F32;
    cur_3d.data    = layer_in->data;
    cur_3d.ne[0]   = n_embd;
    cur_3d.ne[1]   = 1;
    cur_3d.ne[2]   = n_tokens;
    cur_3d.ne[3]   = 1;
    cur_3d.nb[0]   = sizeof(float);
    cur_3d.nb[1]   = (size_t) n_embd * sizeof(float);
    cur_3d.nb[2]   = (size_t) n_embd * sizeof(float);
    cur_3d.nb[3]   = (size_t) n_embd * (size_t) n_tokens * sizeof(float);

    // src1 for the down matmul: slot_b as a 3D [n_ff, n_used, n_tokens]
    // view (per-slot X, shared_x=false).  Built upfront so the batched
    // dispatch can pass it as part of the down canonical's
    // pre_inputs/plan — execute reads slot_b only AFTER SwiGLU has
    // written it, which the compute stream's in-order semantics
    // guarantee.
    ggml_tensor gated_3d{};
    gated_3d.type  = GGML_TYPE_F32;
    gated_3d.data  = ss->slot_b;
    gated_3d.ne[0] = n_ff;
    gated_3d.ne[1] = n_used;
    gated_3d.ne[2] = n_tokens;
    gated_3d.ne[3] = 1;
    gated_3d.nb[0] = sizeof(float);
    gated_3d.nb[1] = (size_t) n_ff * sizeof(float);
    gated_3d.nb[2] = (size_t) n_ff * (size_t) n_used * sizeof(float);
    gated_3d.nb[3] = (size_t) n_ff * (size_t) n_used *
                     (size_t) n_tokens * sizeof(float);

    return dispatch_three_canonicals_(
        *gate_comp, *up_comp, *down_comp,
        gate_canonical, up_canonical, down_canonical,
        stream_h,
        &cur_3d, &gated_3d,
        ids, probs, weights,
        layer_in, layer_out,
        ss->slot_a, ss->slot_b,
        n_tokens, n_used, n_expert_in_probs,
        n_ff, n_embd, layer_idx);
}

// M1 cutover S2: router-gate binding (criterion 5). Caches the
// per-layer ffn_gate_inp pointers for a post-M1 move; M1 does not
// read from this cache (router/topk is built by the arch builder
// in S5 using the shared helper).
void MoEAnyBcqExecutor::attach_router_gates(
    const struct ggml_tensor * const * ffn_gate_inp_per_layer,
    int                                n_layer)
{
    ffn_gate_inp_.clear();
    if (ffn_gate_inp_per_layer == nullptr || n_layer <= 0) return;
    ffn_gate_inp_.reserve((size_t) n_layer);
    for (int il = 0; il < n_layer; ++il) {
        ffn_gate_inp_.push_back(ffn_gate_inp_per_layer[il]);
    }
}

}  // namespace qwen3
}  // namespace dp_moe_ext

// ───────────────────────────────────────────────────────────────────
// Per-arch register entry points (each is a thin shim that creates
// its arch's subclass instance).  Forward-declared here so the
// legacy ``register_qwen3_moe_anybcq_executor`` below can fan out
// without pulling per-arch headers into this TU.
// ───────────────────────────────────────────────────────────────────
namespace dp_moe_ext {
namespace qwen3        { void register_qwen3_moe_executor();     }
namespace deepseek_moe { void register_deepseek_moe_executor();  }
#if defined(DP_MOE_HAVE_GEMMA4_EXECUTOR)
namespace gemma_4      { void register_gemma4_moe_executor();    }
#endif

namespace qwen3 {

void register_qwen3_moe_anybcq_executor() {
    // Legacy compat: a single call from ``runtime_glue.cpp`` covers
    // every supported arch.  Each per-arch register_*() is itself
    // idempotent (static once-flag), so the legacy entry can be
    // invoked alongside direct per-arch calls without
    // double-registration risk.
    qwen3::register_qwen3_moe_executor();
    deepseek_moe::register_deepseek_moe_executor();
#if defined(DP_MOE_HAVE_GEMMA4_EXECUTOR)
    gemma_4::register_gemma4_moe_executor();
#endif
}

}}  // namespace dp_moe_ext::qwen3
