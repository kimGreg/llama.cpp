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
#include "qwen3_moe_fused.h"           // launch_swiglu_mul, launch_weighted_reduce_slots
#include "qwen3_moe_matmul_comp.h"     // MoEMatMulComp, MoEInput/Output
#include "qwen3_moe_scheduler.h"       // qwen3::scheduler_* helpers
#include "computation.h"               // ChunkPlan, ChunkKey
#include "vram_pool.h"                 // ChunkState
#include "runtime.h"
#include "scheduler.h"

#include <ggml.h>
#include <cuda_runtime.h>

#include <algorithm>
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

// S8 (Mode A): the per-canonical ``forward_moe_block`` body was
// retired with the legacy MUL_MAT_ID rail. ``forward_moe_layer``
// (below) is the sole managed-MoE dispatch path.

// M1 cutover S6: per-canonical chunked matmul dispatch helper.
// Replicates forward_moe_block's plan→reserve→load→wait→validate→
// execute→release sequence using a caller-supplied comp + synthesized
// src1/dst views. ``src1_synth`` is a stack-allocated ggml_tensor in
// the caller (forward_moe_layer) wrapping a raw F32 device pointer;
// the comp's execute() reads only {.type, .data, .nb[1..2]} from
// it, plus the shared_x flag carried via MoEInput.
bool Qwen3MoEAnyBcqExecutor::dispatch_one_canonical_(
    qwen3::MoEMatMulComp & comp,
    const std::string &    canonical,
    StreamHandle           stream_h,
    const ggml_tensor *    src1_synth,
    const ggml_tensor *    ids,
    const ggml_tensor *    probs,
    const ggml_tensor *    weights,
    void *                 dst_data_f32,
    int                    n_tokens,
    int                    n_used,
    int                    n_expert_in_probs,
    bool                   shared_x,
    int                    layer_idx)
{
    cudaStream_t stream = (cudaStream_t) stream_h;

    // Synthesize a dst ggml_tensor wrapping the slot scratch.
    // comp.execute() reads .data only. Its memset uses
    // (n_tokens * n_used * M) so we leave .ne / .nb unset.
    ggml_tensor dst_synth{};
    dst_synth.type = GGML_TYPE_F32;
    dst_synth.data = dst_data_f32;

    qwen3::MoEInput  in;
    in.src0              = nullptr;
    in.src1              = src1_synth;
    in.ids               = ids;
    in.probs             = probs;
    in.weights           = weights;
    in.K                 = comp.K();
    in.M                 = comp.M();
    in.n_tokens          = n_tokens;
    in.n_used_per_tok    = n_used;
    in.n_expert_in_probs = n_expert_in_probs;
    in.shared_x          = shared_x;
    in.layer_index       = layer_idx;

    qwen3::MoEOutput out;
    out.dst = &dst_synth;

    // D2H ids/probs/weights into comp's pinned buffers; sync once.
    for (auto & m : comp.pre_inputs(in)) {
        if (m.bytes == 0) continue;
        if (m.is_2d) {
            cudaMemcpy2DAsync(m.host_dst, m.dst_pitch,
                m.device_src, m.src_pitch,
                m.bytes, m.height,
                cudaMemcpyDeviceToHost, stream);
        } else {
            cudaMemcpyAsync(m.host_dst, m.device_src, m.bytes,
                cudaMemcpyDeviceToHost, stream);
        }
    }
    cudaStreamSynchronize(stream);

    ChunkPlan plan = comp.plan(in);

    if (!plan.load_set.empty()) {
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
        for (const auto & k : plan.load_set) {
            rt_->pool().wait_on_stream(k.wid, k.cid, stream_h);
        }
    }

    if (!validate_required_set_(plan, stream_h)) {
        if (!plan.load_set.empty()) {
            for (const auto & k : plan.load_set) {
                qwen3::scheduler_release_from_dispatch(
                    rt_->scheduler(), k.wid, k.cid);
            }
        }
        std::fprintf(stderr,
            "streamllm-ext: forward_moe_layer[%s, L=%d]: "
            "required_set validation failed\n",
            canonical.c_str(), layer_idx);
        return false;
    }

    comp.execute(in, out, stream_h);

    if (!plan.load_set.empty()) {
        for (const auto & k : plan.load_set) {
            qwen3::scheduler_release_from_dispatch(
                rt_->scheduler(), k.wid, k.cid);
        }
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
bool Qwen3MoEAnyBcqExecutor::forward_moe_layer(
    StreamHandle        stream_h,
    const ggml_tensor * layer_in,
    const ggml_tensor * ids,
    const ggml_tensor * probs,
    const ggml_tensor * weights,
    ggml_tensor *       layer_out,
    int                 layer_idx)
{
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

    // === Phase 1: Gate matmul → slot_a [n_tokens, n_used, n_ff] f32.
    if (!dispatch_one_canonical_(
            *gate_comp, gate_canonical, stream_h,
            &cur_3d, ids, probs, weights,
            ss->slot_a, n_tokens, n_used, n_expert_in_probs,
            /*shared_x=*/true, layer_idx)) {
        return false;
    }

    // === Phase 2: Up matmul → slot_b [n_tokens, n_used, n_ff] f32.
    if (!dispatch_one_canonical_(
            *up_comp, up_canonical, stream_h,
            &cur_3d, ids, probs, weights,
            ss->slot_b, n_tokens, n_used, n_expert_in_probs,
            /*shared_x=*/true, layer_idx)) {
        return false;
    }

    // === Phase 3: SwiGLU: slot_b ← silu(slot_a) * slot_b.
    const size_t N_swiglu =
        (size_t) n_tokens * (size_t) n_used * (size_t) n_ff;
    qwen3::launch_swiglu_mul(
        (const float *) ss->slot_a,
        (const float *) ss->slot_b,
        (float *)       ss->slot_b,
        N_swiglu, stream_h);

    // === Phase 4: Down matmul.
    // src1 = slot_b viewed as 3D [n_ff, n_used, n_tokens] (per-slot X,
    // shared_x=false). dst = slot_a (reused; sized for hidden_dim).
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

    if (!dispatch_one_canonical_(
            *down_comp, down_canonical, stream_h,
            &gated_3d, ids, probs, weights,
            ss->slot_a, n_tokens, n_used, n_expert_in_probs,
            /*shared_x=*/false, layer_idx)) {
        return false;
    }

    // === Phase 5: weighted reduce slot_a → layer_out.
    // weights from llm_build_moe_routing_softmax_topk has memory
    // layout [t, u] contiguous (whether the ne is [n_used, n_tokens]
    // or [1, n_used, n_tokens] — both reduce to the same byte order).
    qwen3::launch_weighted_reduce_slots(
        (const float *) ss->slot_a,
        (const float *) weights->data,
        (float *)       layer_out->data,
        n_tokens, n_used, n_embd, stream_h);

    return true;
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
