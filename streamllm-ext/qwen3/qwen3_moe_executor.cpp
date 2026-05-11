// streamllm-ext / qwen3 — Qwen3 MoE AnyBCQ executor implementation.
//
// Milestone 1 Step 3 (2026-05-12): the dispatch body moved here from
// moe_dispatch::handle_mul_mat_id_impl. The executor now owns the
// canonical Mode A flow (current-batch routing → plan → reserve → load
// → wait → kernel → release). The per-op hook entry point in
// qwen3_runtime_glue.cpp::streamllm_try_cuda_mul_mat_id reaches us
// directly; the legacy scheduler.dispatch_node path (used when a
// managed MUL_MAT_ID surfaces through the dense-matmul hook) reaches
// us through the thin shim that survives in qwen3_moe_dispatch.cpp.
//
// `rt_` is set once at bind_to_model time and is read-only from then
// on — the body does NOT take g_runtime_mu. The runtime's internal
// state (pool, scheduler, layouts) carries its own thread-safety.

#include "qwen3_moe_executor.h"
#include "qwen3_moe_dispatch.h"        // scratch_for_stream, topk_weights_lookup, profile_inc_hook_calls
#include "qwen3_moe_matmul_comp.h"     // MoEMatMulComp, MoEInput/Output
#include "qwen3_moe_scheduler.h"       // qwen3::scheduler_* helpers
#include "computation.h"               // ChunkPlan, ChunkKey
#include "vram_pool.h"                 // ChunkState
#include "runtime.h"
#include "scheduler.h"

#include <ggml.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_set>

namespace streamllm_ext { namespace qwen3 {

std::atomic<std::uint64_t> Qwen3MoEAnyBcqExecutor::required_set_misses_{0};

void Qwen3MoEAnyBcqExecutor::bind_to_model(
    StreamllmRuntime &  rt,
    const StreamReader & /*reader*/,
    const std::string &  /*gguf_path*/)
{
    rt_ = &rt;
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
    return false;
#endif
}

const ggml_tensor * Qwen3MoEAnyBcqExecutor::probe_probs_tensor_(
    const ggml_tensor * ids, int n_tokens)
{
    if (ids == nullptr || ids->src[0] == nullptr) return nullptr;
    const ggml_tensor * a = ids->src[0];
    if (a->type == GGML_TYPE_F32 && a->data != nullptr &&
        (int64_t)a->ne[1] == n_tokens) {
        return a;
    }
    const ggml_tensor * b = a->src[0];
    if (b != nullptr && b->type == GGML_TYPE_F32 && b->data != nullptr &&
        (int64_t)b->ne[1] == n_tokens) {
        return b;
    }
    return nullptr;
}

bool Qwen3MoEAnyBcqExecutor::forward_moe_block(
    StreamHandle stream,
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    const ggml_tensor * ids,
    ggml_tensor *       dst)
{
    if (rt_ == nullptr) return false;
    if (src0 == nullptr || src0->name[0] == '\0') return false;
    if (src1 == nullptr || ids == nullptr || dst == nullptr) return false;

    const std::string canonical(src0->name);
    // Scheduler decides if this canonical is claimed; we don't query a
    // runtime-side managed_names set anymore (P2★ moved that onto the
    // scheduler). claims_tensor on the same src0 object is exactly the
    // predicate we want.
    if (!rt_->scheduler().claims_tensor(src0)) return false;
    if (rt_->layout(canonical + ":e0") == nullptr) return false;

    qwen3::MoEMatMulComp * comp =
        qwen3::scheduler_lookup_moe_comp(rt_->scheduler(), canonical);
    if (comp == nullptr || !comp->valid()) return false;

    // Step-4b instrumenter: fire LayerBegin/LayerEnd markers when
    // crossing a layer boundary. dst is the unique node identity in
    // the prewalk-built node→layer map.
    qwen3::scheduler_on_managed_node_visit(
        rt_->scheduler(), dst, stream);

    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
        ids->type != GGML_TYPE_I32) {
        std::fprintf(stderr,
            "streamllm-ext: forward_moe_block bailing on wid=%s "
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
            "streamllm-ext: forward_moe_block bail %s "
            "(src1->ne[1]=%lld, expected 1 or %d)\n",
            canonical.c_str(), (long long)src1->ne[1], n_used_per_tok);
        return false;
    }
    if ((int)ids->ne[1] != n_tokens ||
        (int)dst->ne[1] != n_used_per_tok ||
        (int)dst->ne[2] != n_tokens) {
        std::fprintf(stderr,
            "streamllm-ext: forward_moe_block bail %s "
            "(ids/dst shape mismatch)\n",
            canonical.c_str());
        return false;
    }

    if (moe_dispatch::scratch_for_stream((cudaStream_t) stream) == nullptr) {
        std::fprintf(stderr,
            "streamllm-ext: forward_moe_block bailing — scratch alloc failed\n");
        return false;
    }

    moe_dispatch::profile_inc_hook_calls();
    if (getenv("STREAMLLM_TRACE")) {
        static std::unordered_set<std::string> seen;
        static std::mutex seen_mu;
        std::lock_guard<std::mutex> sk(seen_mu);
        if (seen.insert(canonical).second) {
            std::fprintf(stderr,
                "streamllm-ext: forward_moe_block fired wid=%s "
                "K=%d M=%d n_used=%d n_tokens=%d\n",
                canonical.c_str(), K, M, n_used_per_tok, n_tokens);
        }
    }

    // Probe the cgraph for the F32 probs tensor (lives one or two
    // parents up from ``ids``); the LOAD walk uses it for per-expert
    // gate-score lookup when the topk_moe weights side-channel isn't
    // populated. Topology is graph-stable so this probe runs once per
    // recording and is reused on every replay.
    const ggml_tensor * probs = probe_probs_tensor_(ids, n_tokens);
    int n_expert_in_probs = probs != nullptr ? (int) probs->ne[0] : 0;

    // Renormalised topk weights side-channel (captured at recording
    // time by ``on_topk_moe_observed``; ids->data is stable for the
    // lifetime of one capture session).
    const ggml_tensor * weights = nullptr;
    {
        int n_used_unused = 0;
        moe_dispatch::topk_weights_lookup(ids->data, &weights, &n_used_unused);
    }

    qwen3::MoEInput  in;
    in.src0              = src0;
    in.src1              = src1;
    in.ids               = ids;
    in.probs             = probs;
    in.weights           = weights;
    in.K                 = K;
    in.M                 = M;
    in.n_tokens          = n_tokens;
    in.n_used_per_tok    = n_used_per_tok;
    in.n_expert_in_probs = n_expert_in_probs;
    in.shared_x          = shared_x;
    in.layer_index       =
        std::strncmp(canonical.c_str(), "blk.", 4) == 0
            ? std::atoi(canonical.c_str() + 4) : -1;

    qwen3::MoEOutput out;
    out.dst = dst;

    // Eager dispatch: pre_inputs D2H + host-side plan + worker-pool
    // chunk loads + per-chunk wait_on_stream + comp.execute().
    //
    // streamllm is an eager-only framework: ggml-cuda's
    // user_node_claims hook (consulted via Scheduler::claims_node)
    // disables cuda-graph capture for any cgraph containing managed
    // MoE ops, so this dispatch is always invoked outside capture.
    // The earlier cudaLaunchHostFunc-based path was retired in P4
    // (deadlocks against copy_stream event processing on the CUDA
    // driver-internal thread; per-replay ordering on a same-handle
    // event doesn't work).

    for (auto & m : comp->pre_inputs(in)) {
        if (m.bytes == 0) continue;
        if (m.is_2d) {
            cudaMemcpy2DAsync(m.host_dst, m.dst_pitch,
                m.device_src, m.src_pitch,
                m.bytes, m.height,
                cudaMemcpyDeviceToHost, (cudaStream_t) stream);
        } else {
            cudaMemcpyAsync(m.host_dst, m.device_src, m.bytes,
                cudaMemcpyDeviceToHost, (cudaStream_t) stream);
        }
    }
    cudaStreamSynchronize((cudaStream_t) stream);
    ChunkPlan plan = comp->plan(in);

    if (!plan.load_set.empty()) {
        // Reserve the load_set in the scheduler's tracker so workers'
        // ``make_room`` (called when the pool is full) skips chunks
        // we're about to depend on within this dispatch window.
        // Released after execute() queues its kernels — the tracker's
        // recency stamps from plan() then take over (the just-touched
        // chunks score lowest for eviction). Mirrors the legacy
        // pre-Step 2 dispatch path.
        for (const auto & k : plan.load_set) {
            qwen3::scheduler_reserve_for_dispatch(
                rt_->scheduler(), k.wid, k.cid);
        }
        const bool async_on = rt_->io_worker_count() > 0;
        if (async_on) {
            auto batch = std::make_shared<std::atomic<uint32_t>>(0);
            for (const auto & k : plan.load_set) {
                if (rt_->pool().is_resident(k.wid, k.cid)) continue;
                rt_->submit_async_load(k.wid, k.cid, batch);
            }
            rt_->wait_async_load_batch(batch);
        } else {
            for (const auto & k : plan.load_set) {
                if (rt_->pool().is_resident(k.wid, k.cid)) continue;
                rt_->move_chunk(k.wid, k.cid, Tier::RAM, Tier::VRAM,
                                /*compute_stream=*/nullptr);
            }
        }
        // Order the compute stream after the loader's per-chunk
        // ready_event so the kernel reads up-to-date plane pointers.
        for (const auto & k : plan.load_set) {
            rt_->pool().wait_on_stream(k.wid, k.cid, stream);
        }
    }

    // Step 6 (Milestone 1): host-side pre-launch validation. After
    // submit_loads + wait_loads_on, every chunk in plan.required_set
    // (load_set ∪ keep_set) MUST be at POINTER_TABLE_READY. Catches
    // a missed load or a stale keep_set entry before the kernel
    // trap fires — debug build aborts, release build counts + logs.
    if (!validate_required_set_(plan, stream)) {
        // Release-build path: the validation already logged the
        // failure and incremented required_set_misses_. Bail
        // instead of launching the kernel, which would M1-trap on
        // device anyway.
        if (!plan.load_set.empty()) {
            for (const auto & k : plan.load_set) {
                qwen3::scheduler_release_from_dispatch(
                    rt_->scheduler(), k.wid, k.cid);
            }
        }
        return false;
    }

    comp->execute(in, out, stream);
    if (!plan.load_set.empty()) {
        for (const auto & k : plan.load_set) {
            qwen3::scheduler_release_from_dispatch(
                rt_->scheduler(), k.wid, k.cid);
        }
    }
    return true;
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
