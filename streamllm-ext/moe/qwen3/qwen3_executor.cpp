// streamllm-ext / qwen3 — Qwen3 MoE AnyBCQ executor implementation.
//
// Milestone 1 Step 3 (2026-05-12): the dispatch body moved here from
// moe_dispatch::handle_mul_mat_id_impl. The executor now owns the
// canonical Mode A flow (current-batch routing → plan → reserve → load
// → wait → kernel → release). The per-op hook entry point in
// runtime_glue.cpp::streamllm_try_cuda_mul_mat_id reaches us
// directly; the legacy scheduler.dispatch_node path (used when a
// managed MUL_MAT_ID surfaces through the dense-matmul hook) reaches
// us through the thin shim that survives in dispatch.cpp.
//
// `rt_` is set once at bind_to_model time and is read-only from then
// on — the body does NOT take g_runtime_mu. The runtime's internal
// state (pool, scheduler, layouts) carries its own thread-safety.

#include "qwen3_executor.h"
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

#include <ggml.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>   // getenv (STREAMLLM_VIOLATION_DUMP)
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_set>

namespace streamllm_ext { namespace qwen3 {

std::atomic<std::uint64_t> Qwen3MoEAnyBcqExecutor::required_set_misses_{0};

Qwen3MoEAnyBcqExecutor::~Qwen3MoEAnyBcqExecutor()
{
    std::lock_guard<std::mutex> lk(slot_scratch_mu_);
    for (auto & [_, ss] : slot_scratch_) {
        if (ss.slot_a) cudaFree(ss.slot_a);
        if (ss.slot_b) cudaFree(ss.slot_b);
    }
    slot_scratch_.clear();
}

void Qwen3MoEAnyBcqExecutor::bind_to_model(
    StreamllmRuntime &  rt,
    const StreamReader & /*reader*/,
    const std::string &  /*gguf_path*/)
{
    rt_ = &rt;
}

SlotScratch * Qwen3MoEAnyBcqExecutor::acquire_slot_scratch_(
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
                "streamllm-ext: slot_scratch slot_a cudaMalloc(%zu) failed\n",
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
                "streamllm-ext: slot_scratch slot_b cudaMalloc(%zu) failed\n",
                b_bytes);
            ss.slot_b = nullptr;
            return nullptr;
        }
        ss.slot_b_bytes = b_bytes;
    }
    return &ss;
}

bool Qwen3MoEAnyBcqExecutor::validate_required_set_(
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
        "streamllm-ext: required_set residency violation — "
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
    // Gated by STREAMLLM_VIOLATION_DUMP=1 to avoid log spam in
    // normal runs.  Reveals whether the miss is:
    //   - a load_set / required_set mismatch (required cid not in load_set)
    //   - a wait/H2D timing issue (load_set has it but state < ready)
    //   - an eviction race (resident earlier, NOT_RESIDENT now)
    static const bool dump_enabled =
        std::getenv("STREAMLLM_VIOLATION_DUMP") != nullptr;
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
bool Qwen3MoEAnyBcqExecutor::dispatch_three_canonicals_(
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
        // ── Phase 0a: enqueue pre_inputs D2Hs for all three canonicals.
        //    Each canonical's pre_inputs writes to its own pinned arrays;
        //    we're issuing 3× D2H_ids + 3× D2H_probs + 3× D2H_weights but
        //    serialised on one stream and followed by ONE sync.
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

        // ── Phase 0c: plan all three.
        plan_gate = gate_comp.plan(in_gate);
        plan_up   = up_comp.plan(in_up);
        plan_down = down_comp.plan(in_down);

        auto count_residency = [&](const ChunkPlan & pl) {
            if (pl.required_set.empty()) return;
            size_t hits = 0;
            for (const auto & k : pl.required_set) {
                if (rt_->pool().is_resident(k.wid, k.cid)) ++hits;
            }
            rt_->pool().note_required_set(pl.required_set.size(), hits);
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
        reserve_plan(plan_gate);
        reserve_plan(plan_up);
        reserve_plan(plan_down);

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
            const bool async_on = rt_->io_worker_count() > 0;
            if (async_on) {
                auto batch = std::make_shared<std::atomic<uint32_t>>(0);
                auto submit_plan = [&](const ChunkPlan & pl) {
                    for (const auto & k : pl.load_set) {
                        if (rt_->pool().is_resident(k.wid, k.cid)) continue;
                        rt_->submit_async_load(k.wid, k.cid, batch);
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
                                        /*compute_stream=*/nullptr);
                    }
                };
                move_plan(plan_gate);
                move_plan(plan_up);
                move_plan(plan_down);
            }
            auto wait_plan = [&](const ChunkPlan & pl) {
                for (const auto & k : pl.load_set) {
                    rt_->pool().wait_on_stream(k.wid, k.cid, stream_h);
                }
            };
            wait_plan(plan_gate);
            wait_plan(plan_up);
            wait_plan(plan_down);
        }

        // ── Phase 0f: validate all three.
        if (!validate_required_set_(plan_gate, stream_h) ||
            !validate_required_set_(plan_up,   stream_h) ||
            !validate_required_set_(plan_down, stream_h))
        {
            release_all();
            std::fprintf(stderr,
                "streamllm-ext: forward_moe_layer[L=%d] (batched): required_set "
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

    // ── Phase 1+2: Gate + Up matmul, each as its own launch.
    gate_comp.execute(in_gate, out_gate, stream_h);
    up_comp.execute(in_up, out_up, stream_h);

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
    // ── Phase 4: Down matmul → slot_a (reused buffer).
    down_comp.execute(in_down, out_down, stream_h);
    // ── Phase 5: weighted reduce slot_a → layer_out.
    qwen3::launch_weighted_reduce_slots(
        (const float *) slot_a_data,
        (const float *) weights->data,
        (float *)       layer_out->data,
        n_tokens, n_used, n_embd, stream_h);

    // ── Phase 6: release the union of reserved chunks.
    release_all();
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
bool Qwen3MoEAnyBcqExecutor::forward_moe_layer(
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
            "streamllm-ext: forward_moe_layer[L=%d] dtype mismatch — "
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
            "streamllm-ext: forward_moe_layer[L=%d]: missing/invalid comp "
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
            "streamllm-ext: forward_moe_layer[L=%d]: shape mismatch — "
            "gate.M=%d up.M=%d down.K=%d down.M=%d n_embd=%d",
            layer_idx, M_gate, M_up, K_down, M_down, n_embd);
    }
    if (gate_comp->K() != n_embd || up_comp->K() != n_embd) {
        GGML_ABORT(
            "streamllm-ext: forward_moe_layer[L=%d]: input K mismatch — "
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
            "streamllm-ext: forward_moe_layer[L=%d]: scratch alloc failed\n",
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
void Qwen3MoEAnyBcqExecutor::attach_router_gates(
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

void register_qwen3_moe_anybcq_executor() {
    static bool once = false;
    if (once) return;
    once = true;
    register_executor(
        "qwen3_moe_anybcq_v1",
        []() -> std::unique_ptr<ModelExecutor> {
            return std::unique_ptr<ModelExecutor>(
                new Qwen3MoEAnyBcqExecutor());
        });
}

}}  // namespace streamllm_ext::qwen3
