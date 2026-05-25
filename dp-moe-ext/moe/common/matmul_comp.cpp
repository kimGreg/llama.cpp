// DPMoE / qwen3 — MoEMatMulComp implementation.
//
// Lifts the LOAD walk and kernel-launch glue from
// ``dispatch.cpp`` into the three ChunkedComputation
// virtuals.  ``Runtime::run`` orchestrates the captured sequence of
// (D2H pre_inputs → host-fn that calls plan → cudaStreamWaitEvent →
// captured kernels in execute) — same code under cuda-graph capture
// and under eager dispatch, with the planner running on every replay
// so freshly-routed chunks land before the captured kernel reads them.
//
// Layer boundary (constraints 1 + 5): this TU is in the **model**
// layer.  It works in CHUNKS only.  The plane↔chunk translation, H2D
// of the kernel-facing per-expert precision array, the any-prec
// q_bias refresh, and the fused MoE GEMV launch all live in
// ``decoder/anybcq/chunked_matmul.cu``.

#include "matmul_comp.h"

#include "dispatch.h"   // moe_dispatch::scratch_for_stream
#include "moe_scheduler.h"  // qwen3::scheduler_plan_for_expert_with_chunks
#include "launch_diag.h"          // launch_diag counters
#include "runtime.h"
#include "runtime_diag.h"
#include "anybcq_gemm.h"          // launch_f32_to_f16
#include "decoder/anybcq/chunked_matmul.h"  // anybcq::moe_chunk_matmul

#include <ggml.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <unordered_set>

namespace dp_moe_ext {
namespace qwen3 {


MoEMatMulComp::MoEMatMulComp(
        DPMoERuntime &           rt,
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
    }
    if (n_chunks_ < 1) n_chunks_ = 1;
    if (n_chunks_ > (int) kMaxChunksPerTensor) {
        n_chunks_ = (int) kMaxChunksPerTensor;
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
                "DPMoE: MoEMatMulComp(%s): cudaHostAlloc(%zuB) failed: %s\n",
                canonical_.c_str(), bytes, cudaGetErrorString(err));
            return nullptr;
        }
        return p;
    };

    const size_t tu_count =
        (size_t) max_n_tokens_ * (size_t) max_n_used_;
    const size_t tx_count =
        (size_t) max_n_tokens_ * (size_t) std::max(n_experts_, 1);

    ids_pinned_     = (int32_t *) host_alloc(tu_count * sizeof(int32_t));
    weights_pinned_ = (float *)   host_alloc(tu_count * sizeof(float));
    probs_pinned_   = (float *)   host_alloc(tx_count * sizeof(float));

    // Per-expert CHUNK COUNT buffer (constraint 1).  Model-side unit.
    // plan() writes ``n_chunks_per_eid[e]`` per routed expert;
    // execute() passes this to anybcq::moe_chunk_matmul, which
    // translates to planes internally before H2D + kernel launch.
    const size_t n_chunks_bytes = (size_t) std::max(n_experts_, 1) * sizeof(int);
    host_n_chunks_per_expert_ = (int *) host_alloc(n_chunks_bytes);

    // Device scratch the decoder fills with kernel-facing per-expert
    // precision.  We allocate it (model owns lifetime) but never read
    // or write its contents from this layer.
    if (n_chunks_bytes > 0) {
        cudaError_t err = cudaMalloc(&prec_per_eid_d_, n_chunks_bytes);
        if (err != cudaSuccess) {
            std::fprintf(stderr,
                "DPMoE: MoEMatMulComp(%s): cudaMalloc(prec_per_eid_d, %zuB) failed: %s\n",
                canonical_.c_str(), n_chunks_bytes, cudaGetErrorString(err));
            prec_per_eid_d_ = nullptr;
        }
    }
    const int B1 = std::max(1, n_experts_) * std::max(1, n_chunks_) + 1;
    const size_t dp_bytes = (size_t) B1 * sizeof(float);
    const size_t trace_bytes = (size_t) std::max(1, n_experts_) *
                               (size_t) B1 * sizeof(uint8_t);
    if (cudaMalloc((void **) &plan_dp_prev_d_, dp_bytes) != cudaSuccess ||
        cudaMalloc((void **) &plan_dp_cur_d_,  dp_bytes) != cudaSuccess ||
        cudaMalloc((void **) &plan_trace_d_,   trace_bytes) != cudaSuccess) {
        std::fprintf(stderr,
            "DPMoE: MoEMatMulComp(%s): capture planner scratch "
            "allocation failed\n",
            canonical_.c_str());
        if (plan_dp_prev_d_) cudaFree(plan_dp_prev_d_);
        if (plan_dp_cur_d_)  cudaFree(plan_dp_cur_d_);
        if (plan_trace_d_)   cudaFree(plan_trace_d_);
        plan_dp_prev_d_ = nullptr;
        plan_dp_cur_d_ = nullptr;
        plan_trace_d_ = nullptr;
    }

}


MoEMatMulComp::~MoEMatMulComp() {
    if (ids_pinned_)               cudaFreeHost(ids_pinned_);
    if (weights_pinned_)           cudaFreeHost(weights_pinned_);
    if (host_n_chunks_per_expert_) cudaFreeHost(host_n_chunks_per_expert_);
    if (probs_pinned_)             cudaFreeHost(probs_pinned_);
    if (prec_per_eid_d_)           cudaFree(prec_per_eid_d_);
    if (plan_dp_prev_d_)           cudaFree(plan_dp_prev_d_);
    if (plan_dp_cur_d_)            cudaFree(plan_dp_cur_d_);
    if (plan_trace_d_)             cudaFree(plan_trace_d_);
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

    // Pre-fill host_n_chunks_per_expert_ with the encoder's max
    // chunk count as a safe default.  plan() overwrites per-expert
    // entries below.  Defends against a code path that calls
    // execute() without first calling plan(): the decoder gets a
    // valid in-range chunk count rather than zero.
    if (host_n_chunks_per_expert_ != nullptr && n_experts_ > 0) {
        const int fill = n_chunks_ > 0 ? n_chunks_ : 1;
        std::fill_n(host_n_chunks_per_expert_, (size_t) n_experts_, fill);
    }

    std::vector<MemcpySpec> out;

    if (cur_n_tokens_ <= 0 || cur_n_used_ <= 0) return out;
    if (cur_n_tokens_ > max_n_tokens_ || cur_n_used_ > max_n_used_) {
        std::fprintf(stderr,
            "DPMoE: MoEMatMulComp(%s): dispatch shape (%d × %d) "
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

    // probs (optional): contiguous [n_tokens × n_expert] floats.
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

    const int n_tokens   = cur_n_tokens_;
    const int n_used     = cur_n_used_;
    const int n_expert   = cur_n_expert_;
    const int n_chunks_max = n_chunks_;

    const float replay_kbar = rt_->current_replay_kbar();

    // ── Pass 1: aggregate gate energy per expert across the minibatch.
    // Correct batched semantics are Σ_{tokens, top-k slots for e} g², not
    // max(g).  The allocator API squares its gate input internally, so we
    // pass sqrt(Σg²) below to preserve one cost definition.
    std::unordered_map<int, float> gate_sq_sum_by_expert;
    std::unordered_map<int, int>   chunks_by_expert;
    std::vector<int>               unique_experts;
    const size_t reserve_n = (size_t) n_used * 4;
    gate_sq_sum_by_expert.reserve(reserve_n);
    chunks_by_expert.reserve(reserve_n);
    unique_experts.reserve(reserve_n);

    std::vector<float> g_per_tu((size_t) n_tokens * n_used, 0.0f);
    if (external_chunks_per_expert_ != nullptr) {
        auto accept_external_eid = [&](int eid) {
            if (eid < 0 || eid >= n_experts_) return;
            int n_c = external_chunks_per_expert_[eid];
            if (n_c <= 0) return;
            if (n_c > n_chunks_max) n_c = n_chunks_max;
            unique_experts.push_back(eid);
            gate_sq_sum_by_expert.emplace(eid, 0.0f);
            chunks_by_expert[eid] = n_c;
            if (host_n_chunks_per_expert_ != nullptr) {
                host_n_chunks_per_expert_[eid] = n_c;
            }
        };
        if (external_expert_order_ != nullptr && external_expert_order_n_ > 0) {
            for (int i = 0; i < external_expert_order_n_; ++i) {
                accept_external_eid(external_expert_order_[i]);
            }
        } else {
            for (int eid = 0; eid < n_experts_; ++eid) {
                accept_external_eid(eid);
            }
        }
    } else {
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
                const float g2 = g * g;
                auto it_g = gate_sq_sum_by_expert.find(eid);
                if (it_g == gate_sq_sum_by_expert.end()) {
                    gate_sq_sum_by_expert.emplace(eid, g2);
                    unique_experts.push_back(eid);
                } else {
                    it_g->second += g2;
                }
                g_per_tu[(size_t) t * n_used + u] = g;
            }
        }
    }

    // ── Pass 2: per-expert CHUNK count. KBar is the only dial.
    const bool use_profile =
        qwen3::scheduler_kbar_allocator_mode(*sched_) ==
            qwen3::KBarAllocatorMode::Profile &&
        qwen3::scheduler_has_dynamic_residuals(*sched_) &&
        replay_kbar > 0.0f && layer_index_ >= 0;
    int uniform_k = (int) std::lrint((double) replay_kbar);
    if (uniform_k < 1) uniform_k = n_chunks_max;
    if (uniform_k > n_chunks_max) uniform_k = n_chunks_max;

    // Wall-clock the dynamic K-decision loop so we can report overhead.
    auto _decision_t0 = std::chrono::steady_clock::now();

    if (external_chunks_per_expert_ != nullptr) {
        // Already populated by the shared layer-level device planner.
    } else
    if (!use_profile) {
        for (int eid : unique_experts) {
            chunks_by_expert[eid] = uniform_k;
            if (host_n_chunks_per_expert_ != nullptr &&
                eid >= 0 && eid < n_experts_) {
                host_n_chunks_per_expert_[eid] = uniform_k;
            }
        }
    } else
    {
        std::vector<float> gates(unique_experts.size());
        for (size_t i = 0; i < unique_experts.size(); ++i) {
            gates[i] = std::sqrt(std::max(
                0.0f, gate_sq_sum_by_expert[unique_experts[i]]));
        }
        std::vector<int> K_out;
        const bool ok = qwen3::scheduler_allocate_dispatch_budget(
            *sched_, layer_index_, unique_experts, gates, K_out);
        for (size_t i = 0; i < unique_experts.size(); ++i) {
            const int eid = unique_experts[i];
            int required_chunks;
            if (ok) {
                required_chunks = K_out[i];
            } else {
                required_chunks = uniform_k;
            }
            if (required_chunks < 1)            required_chunks = 1;
            if (required_chunks > n_chunks_max) required_chunks = n_chunks_max;
            chunks_by_expert[eid] = required_chunks;
            if (host_n_chunks_per_expert_ != nullptr &&
                eid >= 0 && eid < n_experts_) {
                host_n_chunks_per_expert_[eid] = required_chunks;
            }
        }
    }
    // Capture the per-dispatch decision-loop time so we can budget the
    // dynamic dispatcher's CPU overhead vs static / threshold paths.
    const auto _decision_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - _decision_t0).count();
    // Per-layer running-total of decision time so the final dump reports
    // average ns/decision-call.
    static std::atomic<uint64_t> decision_ns_per_layer[256] = {};
    if (layer_index_ >= 0 && layer_index_ < 256) {
        decision_ns_per_layer[layer_index_].fetch_add(
            (uint64_t) _decision_ns, std::memory_order_relaxed);
    }

    // Running-average K̄ across all dispatches per layer.  Dump at
    // dispatch counts {1, 256, 512, 1024, …}.  Each dispatch (matmul-
    // canonical call inside the per-layer MoE forward) contributes
    // ``Σ K_e / |active experts|`` to the running mean.  The final
    // line (highest dispatch count) is the cleanest empirical K̄ over
    // the calibration corpus.
    if (const char * s = std::getenv("DP_MOE_LOG_KBAR")) {
        if (s[0] && s[0] != '0' && layer_index_ >= 0 && layer_index_ < 256) {
            static std::atomic<uint64_t> sum_k_per_layer[256] = {};
            static std::atomic<uint64_t> sum_experts_per_layer[256] = {};
            static std::atomic<uint64_t> dispatches_per_layer[256] = {};
            uint64_t dispatch_k = 0;
            for (auto & kv : chunks_by_expert) dispatch_k += (uint64_t) kv.second;
            const uint64_t s_sum  = sum_k_per_layer[layer_index_].fetch_add(
                dispatch_k, std::memory_order_relaxed) + dispatch_k;
            const uint64_t s_e    = sum_experts_per_layer[layer_index_].fetch_add(
                (uint64_t) chunks_by_expert.size(), std::memory_order_relaxed)
                + (uint64_t) chunks_by_expert.size();
            const uint64_t d_count = dispatches_per_layer[layer_index_].fetch_add(
                1, std::memory_order_relaxed) + 1;
            if (d_count == 1 || (d_count & 0xff) == 0) {
                const double kbar = s_e > 0 ? (double) s_sum / (double) s_e : 0.0;
                const uint64_t total_ns =
                    decision_ns_per_layer[layer_index_].load(std::memory_order_relaxed);
                const double avg_decision_us =
                    d_count > 0 ? (double) total_ns / (double) d_count / 1000.0 : 0.0;
                std::fprintf(stderr,
                    "DPMoE kbar L=%d  dispatches=%lu  active_experts=%zu  "
                    "K_mean=%.3f  gate_sq_sum0=%.6g  decision_us/dispatch=%.3f\n",
                    layer_index_, (unsigned long) d_count,
                    chunks_by_expert.size(), kbar,
                    chunks_by_expert.empty() ? 0.0 :
                        (double) gate_sq_sum_by_expert.begin()->second,
                    avg_decision_us);
            }
        }
    }
    if (external_chunks_per_expert_ == nullptr) {
        for (int t = 0; t < n_tokens; ++t) {
            for (int u = 0; u < n_used; ++u) {
                const int eid = ids_pinned_[(size_t) t * n_used + u];
                const int n_c = chunks_by_expert[eid];
                const float g = g_per_tu[(size_t) t * n_used + u];
                diag::record_gate_event(layer_index_, t, u, eid,
                                         g, /*cum_before=*/0.0f, n_c);
                const std::string synthetic =
                    canonical_ + ":e" + std::to_string(eid);
                for (int c = 0; c < n_c; ++c) {
                    diag::record_chunk(*rt_, synthetic, u, c, cid_chunk(c));
                }
            }
        }
    }

    // ── Pass 3: per-expert plan → load_set (deduplicated) +
    //            required_set (every CHUNK the kernel will read).
    //            BOTH are derived from the same chunks_by_expert
    //            (constraint 3: single source of chunk count).
    std::unordered_set<std::string> issued_keys;
    issued_keys.reserve(unique_experts.size() * 8);
    out_plan.load_set.reserve(unique_experts.size() * 8);
    out_plan.required_set.reserve(unique_experts.size() * 8);

    for (int eid : unique_experts) {
        const int n_c = chunks_by_expert[eid];   // CHUNKS, constraint 2
        const std::string synthetic =
            canonical_ + ":e" + std::to_string(eid);

        // Required set: every chunk the kernel reads, cid-indexed.
        // c is a CHUNK INDEX in [0, n_chunks_max].  cid_chunk(c)
        // never escapes this range because n_c ≤ n_chunks_max.
        for (int c = 0; c < n_c; ++c) {
            out_plan.required_set.push_back(
                ChunkKey{synthetic, cid_chunk(c)});
        }

        // Load set: scheduler emits cid-indexed chunk moves for the
        // same n_c.  Both load_set and required_set derive from this
        // single chunk count (constraint 3).  The scheduler entry
        // point is chunks-direct — no plane translation between
        // model and scheduler.
        const Plan * exp_plan =
            qwen3::scheduler_plan_for_expert_with_chunks(
                *sched_, canonical_, eid, n_c,
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
            "DPMoE: MoEMatMulComp(%s): scratch unavailable\n",
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
                "DPMoE: MoEMatMulComp(%s): ids_d scratch too small "
                "(%zu B needed, %zu available)\n",
                canonical_.c_str(), total_bytes,
                moe_dispatch::scratch_ids_bytes_total());
            return;
        }
        launch_diag::note_launch(launch_diag::Kind::Memcpy2DAsync);
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
        launch_diag::note_launch(launch_diag::Kind::MemsetAsync);
        cudaMemsetAsync(out.dst->data, 0, dst_bytes, stream);
    }

    // ── 3a. Capture-mode plan: device-side per-expert chunk-count
    //       computation. Reads the contiguous ids buffer we just
    //       wrote (sc->ids_d), the weights/probs tensors (which the
    //       eager pre_inputs D2H also treats as contiguous), and the
    //       device-side score-thresholds buffer maintained by the
    //       scheduler — writes prec_per_eid_d_ entirely on device.
    //       The H2D + cudaStreamSynchronize in dispatch_three_
    //       canonicals_'s eager path is skipped when this runs.
    const bool capture_path =
        sched_ != nullptr &&
        qwen3::scheduler_allow_capture(*sched_);
    if (capture_path) {
        const auto * layout = any_layout_;
        // Capture-mode rung-2: when dynamic residuals + K̄ are loaded,
        // run the on-device knapsack instead of the threshold plan.
        // Falls back to the threshold path if either is missing.
        const float * R_d =
            qwen3::scheduler_dynamic_R_device(*sched_);
        const float   dyn_kbar = qwen3::scheduler_kbar(*sched_);
        const bool use_kbar =
            R_d != nullptr && dyn_kbar > 0.0f &&
            layout != nullptr && layer_index_ >= 0;
        if (use_kbar) {
            qwen3::launch_plan_per_expert_kbar(
                /*ids_d=*/      (const int32_t *) sc->ids_d,
                /*weights_d=*/  in.weights ? (const float *) in.weights->data : nullptr,
                /*probs_d=*/    in.probs   ? (const float *) in.probs->data   : nullptr,
                /*R_d=*/        R_d,
                /*K_min=*/      qwen3::scheduler_dynamic_K_min(*sched_),
                /*K_max=*/      qwen3::scheduler_dynamic_K_max(*sched_),
                /*layer_index=*/layer_index_,
                /*kbar=*/       dyn_kbar,
                n_tokens, n_used, n_experts_,
                /*n_chunks_max=*/ n_chunks_,
                layout->base_precision > 0 ? layout->base_precision : 1,
                layout->any_precision,
                plan_dp_prev_d_,
                plan_dp_cur_d_,
                plan_trace_d_,
                (int *) prec_per_eid_d_,
                stream_h);
        } else if (layout != nullptr) {
            const int uniform_k = std::max(1, std::min(n_chunks_,
                (int) std::lrint((double) qwen3::scheduler_kbar(*sched_))));
            qwen3::launch_plan_per_expert_uniform(
                (const int32_t *) sc->ids_d,
                n_tokens, n_used, n_experts_,
                uniform_k, n_chunks_,
                layout->base_precision > 0 ? layout->base_precision : 1,
                layout->any_precision,
                (int *) prec_per_eid_d_,
                stream_h);
        }
    }

    // ── 4. Hand off to the decoder.  anybcq::moe_chunk_matmul owns
    //      chunks→planes conversion, H2D of the kernel-facing
    //      per-expert precision array (prec_per_eid_d_), any-prec
    //      q_bias slot refresh, and the fused MoE GEMV launch.
    //      Model layer never sees plane vocabulary (constraint 1).
    //
    //      Capture-mode opt-in: the plan kernel in Phase 3a already
    //      populated prec_per_eid_d_ on the device. Passing nullptr
    //      for host_n_chunks_per_eid signals moe_chunk_matmul to
    //      skip the host conversion + H2D and trust the device
    //      buffer.
    anybcq::moe_chunk_matmul(
        *rt_,
        canonical_,
        *any_layout_,
        *fuse_table_,
        (int *) prec_per_eid_d_,
        capture_path ? nullptr : host_n_chunks_per_expert_,
        (const int32_t *) sc->ids_d,
        sc->xb_f16,
        out.dst->data,
        n_tokens, n_used, n_experts_,
        M, K,
        group_size_,
        shared_x ? 1 : 0,
        stream_h);

    rt_->pool().record_compute_event(stream_h);
}


}  // namespace qwen3
}  // namespace dp_moe_ext
