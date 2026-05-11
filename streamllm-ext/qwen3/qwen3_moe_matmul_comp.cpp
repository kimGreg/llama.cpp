// streamllm-ext / qwen3 — MoEMatMulComp implementation.
//
// Lifts the LOAD walk and kernel-launch blocks from
// ``qwen3_moe_dispatch.cpp`` into the three ChunkedComputation
// virtuals.  ``Runtime::run`` orchestrates the captured sequence of
// (D2H pre_inputs → host-fn that calls plan → cudaStreamWaitEvent →
// captured kernels in execute) — same code under cuda-graph capture
// and under eager dispatch, with the planner running on every replay
// so freshly-routed chunks land before the captured kernel reads them.

#include "qwen3_moe_matmul_comp.h"

#include "qwen3_moe_dispatch.h"   // moe_dispatch::scratch_for_stream / topk_weights_lookup
#include "qwen3_moe_scheduler.h"  // qwen3::scheduler_plan_for_expert_with_precision
#include "runtime.h"
#include "runtime_diag.h"
#include "anybcq_gemm.h"          // launch_f32_to_f16

#include <ggml.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <unordered_set>

namespace streamllm_ext {
namespace qwen3 {

namespace {

// Bounds the per-(t, u) precision the kernel reads. Matches the
// kMaxChunksPerTensor limit imposed by the AnyBCQ kernel.
constexpr int kCompMaxPlanes = kMaxChunksPerTensor;

}  // anonymous namespace


MoEMatMulComp::MoEMatMulComp(
        StreamllmRuntime &           rt,
        Scheduler &                  sched,
        std::string                  canonical,
        const MoeExpertTable *       fuse_table,
        const UpstreamLayoutDevice * any_layout,
        int                          n_experts,
        int                          max_n_tokens,
        int                          max_n_used)
    : rt_(&rt), sched_(&sched),
      canonical_(std::move(canonical)),
      fuse_table_(fuse_table), any_layout_(any_layout),
      n_experts_(n_experts),
      max_n_tokens_(max_n_tokens),
      max_n_used_(max_n_used)
{
    if (any_layout_) {
        K_              = any_layout_->K;
        M_              = any_layout_->M;
        n_chunks_       = any_layout_->n_chunks;
        group_size_     = any_layout_->group_size;
        base_precision_ = any_layout_->base_precision;
        any_precision_  = any_layout_->any_precision;
        uniform_precision_static_ = any_precision_
            ? base_precision_ + n_chunks_ - 1
            : n_chunks_;
        if (uniform_precision_static_ < 1)
            uniform_precision_static_ = 1;
        if (uniform_precision_static_ > kCompMaxPlanes)
            uniform_precision_static_ = kCompMaxPlanes;
    }

    // Layer index parsed from the canonical's ``blk.<L>.…`` prefix.
    // Used only by diag::record_gate_event; harmless if -1.
    if (std::strncmp(canonical_.c_str(), "blk.", 4) == 0) {
        layer_index_ = std::atoi(canonical_.c_str() + 4);
    }

    // Allocate pinned host buffers.  Sized to the per-canonical worst
    // case so the captured graph's D2H copies always fit.
    auto host_alloc = [&](size_t bytes) -> void * {
        if (bytes == 0) return nullptr;
        void * p = nullptr;
        cudaError_t err = cudaHostAlloc(&p, bytes, cudaHostAllocDefault);
        if (err != cudaSuccess) {
            std::fprintf(stderr,
                "streamllm-ext: MoEMatMulComp(%s): cudaHostAlloc(%zuB) failed: %s\n",
                canonical_.c_str(), bytes, cudaGetErrorString(err));
            return nullptr;
        }
        return p;
    };

    const size_t tu_count =
        (size_t) max_n_tokens_ * (size_t) max_n_used_;
    const size_t tx_count =
        (size_t) max_n_tokens_ * (size_t) std::max(n_experts_, 1);

    ids_pinned_           = (int32_t *) host_alloc(tu_count * sizeof(int32_t));
    weights_pinned_       = (float *)   host_alloc(tu_count * sizeof(float));
    probs_pinned_         = (float *)   host_alloc(tx_count * sizeof(float));

    // Per-expert precision buffer (SSOT §6.9 M1 / row 9).  Sized
    // n_experts × int.  plan() writes target_planes[e] per expert;
    // execute() H2Ds to ``prec_per_eid_d_`` for the kernel.  Both
    // expert id (from ids[tu]) and precision boundary come from
    // current-batch routing, so no per-(t, u) broadcast is needed.
    const size_t prec_bytes = (size_t) std::max(n_experts_, 1) * sizeof(int);
    host_prec_per_expert_ = (int *) host_alloc(prec_bytes);
    if (prec_bytes > 0) {
        cudaError_t err = cudaMalloc(&prec_per_eid_d_, prec_bytes);
        if (err != cudaSuccess) {
            std::fprintf(stderr,
                "streamllm-ext: MoEMatMulComp(%s): cudaMalloc(prec_per_eid_d, %zuB) failed: %s\n",
                canonical_.c_str(), prec_bytes, cudaGetErrorString(err));
            prec_per_eid_d_ = nullptr;
        }
    }
}


MoEMatMulComp::~MoEMatMulComp() {
    if (ids_pinned_)           cudaFreeHost(ids_pinned_);
    if (weights_pinned_)       cudaFreeHost(weights_pinned_);
    if (host_prec_per_expert_) cudaFreeHost(host_prec_per_expert_);
    if (probs_pinned_)         cudaFreeHost(probs_pinned_);
    if (prec_per_eid_d_)       cudaFree(prec_per_eid_d_);
}


std::vector<MemcpySpec> MoEMatMulComp::pre_inputs(
    const ComputationInput & in_base)
{
    const MoEInput & in = static_cast<const MoEInput &>(in_base);

    cur_n_tokens_ = in.n_tokens;
    cur_n_used_   = in.n_used_per_tok;
    cur_n_expert_ = in.n_expert_in_probs;

    have_real_scores_   = false;
    have_renorm_weights_ = false;

    // Pre-fill host_prec_per_expert_ with the canonical's max precision
    // as a safe default.  plan() overwrites per-expert entries below.
    // Defending against a code path that calls execute() without first
    // calling plan(): the kernel reads a valid uniform precision
    // rather than zero.
    if (host_prec_per_expert_ != nullptr && n_experts_ > 0) {
        const int fill = uniform_precision_static_ > 0
            ? uniform_precision_static_ : 8;
        std::fill_n(host_prec_per_expert_, (size_t) n_experts_, fill);
    }

    std::vector<MemcpySpec> out;

    if (cur_n_tokens_ <= 0 || cur_n_used_ <= 0) return out;
    if (cur_n_tokens_ > max_n_tokens_ || cur_n_used_ > max_n_used_) {
        std::fprintf(stderr,
            "streamllm-ext: MoEMatMulComp(%s): dispatch shape (%d × %d) "
            "exceeds pinned buffer (%d × %d) — bypass\n",
            canonical_.c_str(), cur_n_tokens_, cur_n_used_,
            max_n_tokens_, max_n_used_);
        cur_n_tokens_ = 0;
        cur_n_used_   = 0;
        return out;
    }

    // ids: 2D pitched D2H into ``ids_pinned_`` packed [t × n_used].
    if (in.ids != nullptr && in.ids->data != nullptr && ids_pinned_ != nullptr) {
        const size_t row_bytes = (size_t) cur_n_used_ * sizeof(int32_t);
        MemcpySpec m;
        m.device_src = in.ids->data;
        m.host_dst   = ids_pinned_;
        m.bytes      = row_bytes;
        m.is_2d      = true;
        m.src_pitch  = (size_t) in.ids->nb[1];
        m.dst_pitch  = row_bytes;
        m.height     = (size_t) cur_n_tokens_;
        out.push_back(m);
    }

    // probs (optional): contiguous [n_tokens × n_expert] floats.  The
    // existing dispatch found this tensor by walking ``ids->src[0]``
    // (or its child); the shim does the walk and forwards via
    // ``MoEInput::probs``.
    if (in.probs != nullptr && in.probs->data != nullptr &&
        cur_n_expert_ > 0 && probs_pinned_ != nullptr &&
        cur_n_expert_ <= n_experts_)
    {
        const size_t bytes = (size_t) cur_n_expert_ *
                             (size_t) cur_n_tokens_ * sizeof(float);
        MemcpySpec m;
        m.device_src = in.probs->data;
        m.host_dst   = probs_pinned_;
        m.bytes      = bytes;
        out.push_back(m);
        have_real_scores_ = true;
    }

    // weights (optional): renorm weights from the topk_moe side
    // channel.  Same layout as ids.
    if (in.weights != nullptr && in.weights->data != nullptr &&
        weights_pinned_ != nullptr)
    {
        const size_t bytes = (size_t) cur_n_used_ *
                             (size_t) cur_n_tokens_ * sizeof(float);
        MemcpySpec m;
        m.device_src = in.weights->data;
        m.host_dst   = weights_pinned_;
        m.bytes      = bytes;
        out.push_back(m);
        have_renorm_weights_ = true;
    }

    return out;
}


ChunkPlan MoEMatMulComp::plan(const ComputationInput & /*in_base*/) {
    ChunkPlan out_plan;

    if (cur_n_tokens_ <= 0 || cur_n_used_ <= 0) return out_plan;
    if (sched_ == nullptr || rt_ == nullptr || any_layout_ == nullptr) {
        return out_plan;
    }

    const int n_tokens = cur_n_tokens_;
    const int n_used   = cur_n_used_;
    const int n_expert = cur_n_expert_;
    const int base_p   = base_precision_ > 0 ? base_precision_ : 1;

    // ── Score-table snapshot (taken once per replay in
    // streamllm_graph_compute_begin; constant across all managed
    // dispatches in this token).
    const std::vector<float> & sc_thresh =
        rt_->current_replay_score_table();

    auto score_lookup = [&](float g) -> int {
        int k = -1;
        for (int i = (int) sc_thresh.size() - 1; i >= 0; --i) {
            if (g >= sc_thresh[i]) { k = i; break; }
        }
        if (k < 0) {
            // Below every threshold — happens when the gate signal is
            // raw logits (negative) instead of post-softmax probs
            // (Qwen3 case).  Fall back to MAX planes for quality —
            // returning base_p drops every expert to its precision
            // floor and the model goes garbled.
            if (!sc_thresh.empty()) {
                int planes = base_p + (int) sc_thresh.size() - 1;
                if (planes > kCompMaxPlanes) planes = kCompMaxPlanes;
                return planes;
            }
            return kCompMaxPlanes;
        }
        int planes = base_p + k;
        if (planes < 1)              planes = 1;
        if (planes > kCompMaxPlanes) planes = kCompMaxPlanes;
        return planes;
    };

    // For any-prec wids, ``desired_precision`` is in PLANES (bits) but
    // chunks store [base_p planes][+1 plane]×(Pa−1).  Reaching plane
    // count P_t needs n_chunks = max(1, P_t − base_p + 1) clamped to
    // Pa, yielding planes_served = base_p + n_chunks − 1.  The
    // per-expert precision written into host_prec_per_expert_ must
    // equal planes_served to avoid reading past the last loaded
    // plane pointer.
    auto translate_planes = [&](int target) -> int {
        if (any_precision_) {
            const int Pa = n_chunks_;
            int n_chunks = target - base_p + 1;
            if (n_chunks < 1)  n_chunks = 1;
            if (n_chunks > Pa) n_chunks = Pa;
            return base_p + n_chunks - 1;
        }
        return target;
    };

    // ── Pass 1: per-expert max gate score across the minibatch.
    std::unordered_map<int, float> max_gate_by_expert;
    std::unordered_map<int, int>   max_prec_by_expert;
    std::vector<int>               unique_experts;
    const size_t reserve_n = (size_t) n_used * 4;
    max_gate_by_expert.reserve(reserve_n);
    max_prec_by_expert.reserve(reserve_n);
    unique_experts.reserve(reserve_n);

    std::vector<float> g_per_tu((size_t) n_tokens * n_used, 0.0f);
    for (int t = 0; t < n_tokens; ++t) {
        for (int u = 0; u < n_used; ++u) {
            const int eid =
                ids_pinned_[(size_t) t * n_used + u];
            float g;
            if (have_renorm_weights_) {
                g = weights_pinned_[(size_t) t * n_used + u];
            } else if (have_real_scores_ &&
                       eid >= 0 && eid < n_expert) {
                g = probs_pinned_[(size_t) t * n_expert + (size_t) eid];
            } else {
                g = 0.45f - 0.05f * (float) u;
                if (g < 0.05f) g = 0.05f;
            }
            auto it_g = max_gate_by_expert.find(eid);
            if (it_g == max_gate_by_expert.end()) {
                max_gate_by_expert.emplace(eid, g);
                unique_experts.push_back(eid);
            } else if (g > it_g->second) {
                it_g->second = g;
            }
            g_per_tu[(size_t) t * n_used + u] = g;
        }
    }

    // ── Pass 2: per-expert target precision.  Write directly into
    // host_prec_per_expert_[eid] (size n_experts).  Kernel reads
    // ``P = prec_per_eid_d[ids[tu]]`` so both expert id and precision
    // boundary come from current-batch routing (§6.9 M1 / row 9).
    for (int eid : unique_experts) {
        const int desired =
            translate_planes(score_lookup(max_gate_by_expert[eid]));
        max_prec_by_expert[eid] = desired;
        if (host_prec_per_expert_ != nullptr &&
            eid >= 0 && eid < n_experts_) {
            host_prec_per_expert_[eid] = desired;
        }
    }
    for (int t = 0; t < n_tokens; ++t) {
        for (int u = 0; u < n_used; ++u) {
            const int eid = ids_pinned_[(size_t) t * n_used + u];
            const int desired = max_prec_by_expert[eid];
            const float g = g_per_tu[(size_t) t * n_used + u];
            diag::record_gate_event(layer_index_, t, u, eid,
                                     g, /*cum_before=*/0.0f, desired);
            const std::string synthetic =
                canonical_ + ":e" + std::to_string(eid);
            for (int pp = 0; pp < desired; ++pp) {
                diag::record_chunk(*rt_, synthetic, u, pp, cid_chunk(pp));
            }
        }
    }


    // ── Pass 3: per-expert plan → load_set (deduplicated).
    std::unordered_set<std::string> issued_keys;
    issued_keys.reserve(unique_experts.size() * 8);
    out_plan.load_set.reserve(unique_experts.size() * 8);

    for (int eid : unique_experts) {
        const int desired = max_prec_by_expert[eid];
        const Plan * exp_plan =
            qwen3::scheduler_plan_for_expert_with_precision(
                *sched_, canonical_, eid, desired,
                /*compute_stream=*/nullptr);
        if (exp_plan == nullptr) continue;
        for (const auto & mv : exp_plan->moves) {
            std::string key = mv.wid + "#" + std::to_string(mv.cid);
            if (!issued_keys.insert(std::move(key)).second) continue;
            out_plan.load_set.push_back(ChunkKey{mv.wid, mv.cid});
        }
    }

    // No explicit eviction set — the loader's reactive
    // ``make_room_for`` covers cap pressure, and replay reservations
    // (added by Runtime::run before the load is submitted) keep the
    // current load_set safe from cross-layer eviction within the same
    // captured replay.
    return out_plan;
}


void MoEMatMulComp::execute(const ComputationInput & in_base,
                             ComputationOutput &      out_base,
                             StreamHandle             stream_h)
{
    const MoEInput & in  = static_cast<const MoEInput &>(in_base);
    MoEOutput &      out = static_cast<MoEOutput &>(out_base);

    if (cur_n_tokens_ <= 0 || cur_n_used_ <= 0) return;
    if (out.dst == nullptr || out.dst->data == nullptr) return;
    if (in.src1 == nullptr || in.ids == nullptr) return;
    if (fuse_table_ == nullptr) return;

    cudaStream_t stream = (cudaStream_t) stream_h;

    moe_dispatch::StreamScratch * sc =
        moe_dispatch::scratch_for_stream(stream);
    if (sc == nullptr || sc->ids_d == nullptr || sc->xb_f16 == nullptr) {
        std::fprintf(stderr,
            "streamllm-ext: MoEMatMulComp(%s): scratch unavailable\n",
            canonical_.c_str());
        return;
    }

    const int    n_tokens = cur_n_tokens_;
    const int    n_used   = cur_n_used_;
    const int    K        = K_;
    const int    M        = M_;
    const bool   shared_x = in.shared_x;
    const size_t k_bytes_f32   = (size_t) K * sizeof(float);
    const size_t x_tile_bytes  = (size_t) K * sizeof(__half);
    const size_t x_used_stride = in.src1->nb[1];
    const size_t x_tok_stride  = in.src1->nb[2];

    // ── 1. F32 → F16 cast of src1 into xb_f16.
    const int n_x_slots = shared_x ? n_tokens : n_tokens * n_used;
    const bool x_contig =
        shared_x ? (x_tok_stride == k_bytes_f32)
                 : (x_used_stride == k_bytes_f32 &&
                    x_tok_stride  == (size_t) n_used * k_bytes_f32);
    if (x_contig) {
        launch_f32_to_f16(in.src1->data, sc->xb_f16,
                          n_x_slots * K, stream);
    } else {
        const uint8_t * x_base = (const uint8_t *) in.src1->data;
        for (int i = 0; i < n_x_slots; ++i) {
            const uint8_t * x_src = shared_x
                ? x_base + (size_t) i * x_tok_stride
                : x_base + (size_t)(i / n_used) * x_tok_stride
                         + (size_t)(i % n_used) * x_used_stride;
            launch_f32_to_f16(
                x_src,
                (uint8_t *) sc->xb_f16 + (size_t) i * x_tile_bytes,
                K, stream);
        }
    }

    // ── 2. ids → device (D2D, 2D pitched).
    {
        const size_t row_bytes   = (size_t) n_used * sizeof(int32_t);
        const size_t total_bytes = (size_t) n_tokens * row_bytes;
        if (total_bytes > moe_dispatch::scratch_ids_bytes_total()) {
            std::fprintf(stderr,
                "streamllm-ext: MoEMatMulComp(%s): ids_d scratch too small "
                "(%zu B needed, %zu available)\n",
                canonical_.c_str(), total_bytes,
                moe_dispatch::scratch_ids_bytes_total());
            return;
        }
        cudaMemcpy2DAsync(
            sc->ids_d, /*dpitch=*/row_bytes,
            in.ids->data, /*spitch=*/(size_t) in.ids->nb[1],
            /*width=*/row_bytes,
            /*height=*/(size_t) n_tokens,
            cudaMemcpyDeviceToDevice, stream);
    }

    // ── 3. Zero dst (the kernel accumulates into Y).
    {
        const size_t dst_bytes =
            (size_t) n_tokens * (size_t) n_used * (size_t) M * sizeof(float);
        cudaMemsetAsync(out.dst->data, 0, dst_bytes, stream);
    }

    // ── 4. H2D host_prec_per_expert_ → prec_per_eid_d_ (SSOT §6.9
    // M1).  Per-expert precision buffer, size n_experts × int.  plan()
    // wrote target_planes[e] for every routed expert; the kernel reads
    // ``P = prec_per_eid_d[ids[tu]]`` so both expert id and precision
    // boundary come from current-batch routing.
    const int * prec_per_eid_d = nullptr;
    if (prec_per_eid_d_ != nullptr && host_prec_per_expert_ != nullptr &&
        n_experts_ > 0) {
        const size_t bytes = (size_t) n_experts_ * sizeof(int);
        cudaMemcpyAsync(prec_per_eid_d_, host_prec_per_expert_,
                        bytes, cudaMemcpyHostToDevice, stream);
        prec_per_eid_d = (const int *) prec_per_eid_d_;
    }

    // ── 5. Refresh per-expert q_bias pointers (any-prec only no-op for
    // shortcut canonicals).  Captured kernel that scatters d_qbias_slot
    // into the table.
    qwen3::refresh_q_bias_for_anyprec_launch(*fuse_table_, stream_h);

    // ── 6. Fused MoE GEMV.  Required planes [0, prec_per_eid_d[eid])
    // MUST be non-null at this point (SSOT §6.9 M1).  The kernel
    // ``__trap()``s on null required pointer.
    qwen3::naver_gemv_moe_launch(
        sc->xb_f16, out.dst->data, (const int32_t *) sc->ids_d,
        *fuse_table_,
        M, K, n_tokens, n_used,
        uniform_precision_static_, prec_per_eid_d,
        group_size_,
        shared_x ? 1 : 0,
        stream_h);

    rt_->pool().record_compute_event(stream_h);
}


}  // namespace qwen3
}  // namespace streamllm_ext
