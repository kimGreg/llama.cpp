// streamllm-ext / moe / qwen3 — Qwen3-MoE executor registration.

#include "qwen3_moe_executor.h"
#include "chunked_tensor.h"
#include "executor.h"        // register_executor
#include "moe_scheduler.h"
#include "runtime.h"
#include "stream_reader.h"
#include "launch_diag.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <stdexcept>
#include <memory>

namespace streamllm_ext { namespace qwen3 {
namespace {

void check_cuda(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("qwen3_baseline: CUDA error in ") +
                                 what + ": " + cudaGetErrorString(e));
    }
}

bool parse_moe_canonical(const char * name, int & layer, int & kind) {
    if (name == nullptr) return false;
    int l = -1;
    char mid[32] = {};
    if (std::sscanf(name, "blk.%d.ffn_%31[^.].weight", &l, mid) != 2) {
        return false;
    }
    std::string s(mid);
    if (s == "gate_exps") kind = 0;
    else if (s == "up_exps") kind = 1;
    else if (s == "down_exps") kind = 2;
    else return false;
    layer = l;
    return true;
}

std::string expert_wid(int layer, int expert) {
    return "blk." + std::to_string(layer) + ".ffn_expert:e" +
           std::to_string(expert);
}

std::string staging_wid(int kind) {
    static constexpr const char * names[3] = {
        "__streamllm_direct_stage_gate",
        "__streamllm_direct_stage_up",
        "__streamllm_direct_stage_down",
    };
    return names[kind];
}

bool direct_refresh_decode_every_call() {
    const char * env = std::getenv("STREAMLLM_DIRECT_REFRESH_EVERY_DECODE");
    return env != nullptr && std::strcmp(env, "0") != 0;
}

} // namespace

void register_qwen3_moe_executor() {
    static bool once = false;
    if (once) return;
    once = true;
    auto factory = []() -> std::unique_ptr<ModelExecutor> {
        return std::unique_ptr<ModelExecutor>(new Qwen3MoEExecutor());
    };
    register_executor("qwen3_ss_anybcq_v1", factory);
    // Legacy metadata alias for already-materialized Qwen3 SS-AnyBCQ GGUFs.
    // The implementation is the current KBar-based executor; this only
    // bridges stale metadata while new encoders emit qwen3_ss_anybcq_v1.
    register_executor("qwen3_anybcq_v1", factory);

    auto baseline_factory = []() -> std::unique_ptr<ModelExecutor> {
        return std::unique_ptr<ModelExecutor>(new Qwen3BaselineExecutor());
    };
    register_executor("qwen3_direct_stock_v1", baseline_factory);
    // Legacy metadata alias for direct expert baseline artifacts produced
    // before the executor rename.
    register_executor("qwen3_baseline_v1", baseline_factory);
}

void Qwen3BaselineExecutor::bind_to_model(
    StreamllmRuntime & rt,
    const StreamReader & reader,
    const std::string &) {
    rt_ = &rt;
    std::array<size_t, 3> bytes = {0, 0, 0};
    for (const auto & wid : reader.chunked_tensor_names()) {
        ChunkedTensor * tensor = rt_->tensor(wid);
        if (tensor == nullptr || !tensor->host().direct_expert_block) {
            continue;
        }
        const auto & host = tensor->host();
        if (host.direct_expert < 0) {
            throw std::runtime_error(
                "qwen3_direct_stock_v1: direct expert block missing expert id");
        }
        for (int k = 0; k < 3; ++k) {
            const auto & r = host.expert_records[k];
            if (r.nbytes == 0) {
                throw std::runtime_error(
                    "qwen3_direct_stock_v1: empty expert subrecord " + wid);
            }
            const size_t need =
                (size_t)(host.direct_expert + 1) * (size_t)r.nbytes;
            if (need > bytes[k]) bytes[k] = need;
        }
    }
    for (int k = 0; k < 3; ++k) {
        staging_wid_[k] = staging_wid(k);
        if (bytes[k] > 0) allocate_staging_(k, bytes[k]);
    }
}

bool Qwen3BaselineExecutor::forward_moe_layer(
    StreamHandle,
    const ggml_tensor *,
    const ggml_tensor *,
    const ggml_tensor *,
    const ggml_tensor *,
    ggml_tensor *,
    int) {
    std::fprintf(stderr,
        "qwen3_direct_stock_v1: sentinel forward_moe_layer is not used; "
        "the baseline executor requires the stock MoE graph\n");
    return false;
}

void Qwen3BaselineExecutor::allocate_staging_(int kind, size_t bytes) {
    if (rt_ == nullptr || kind < 0 || kind >= 3 || bytes == 0) return;
    if (staging_bytes_[kind] >= bytes && staging_[kind] != nullptr) return;

    if (staging_[kind] != nullptr) {
        rt_->pool().evict(staging_wid_[kind], 0);
        staging_[kind] = nullptr;
        staging_bytes_[kind] = 0;
    }

    for (;;) {
        auto h = rt_->pool().load(staging_wid_[kind], 0, nullptr, bytes);
        if (h.device_ptr != nullptr) {
            staging_[kind] = h.device_ptr;
            staging_bytes_[kind] = h.nbytes;
            return;
        }
        if (!rt_->scheduler().make_room_for(rt_->pool(), bytes)) {
            throw std::runtime_error(
                "qwen3_direct_stock_v1: unable to allocate pool staging");
        }
    }
}

bool Qwen3BaselineExecutor::prepare_moe_mul_mat_id(
    StreamHandle stream_h,
    ggml_tensor * dst) {
    if (rt_ == nullptr || dst == nullptr || dst->op != GGML_OP_MUL_MAT_ID) {
        return false;
    }
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * ids  = dst->src[2];
    if (src0 == nullptr || ids == nullptr || ids->type != GGML_TYPE_I32) {
        return false;
    }

    int layer = -1;
    int kind = -1;
    if (!parse_moe_canonical(src0->name, layer, kind)) {
        return false;
    }

    LayerState & st = layers_[layer];
    const bool direct_decode_phase = (int) ids->ne[1] == 1;
    const bool force_decode_refresh =
        direct_decode_phase && direct_refresh_decode_every_call();
    const bool need_prepare =
        force_decode_refresh ||
        st.ids_data != ids->data ||
        st.n_used != (int) ids->ne[0] ||
        st.n_tokens != (int) ids->ne[1] ||
        st.active_experts.empty();

    cudaStream_t stream = (cudaStream_t) stream_h;

    if (need_prepare) {
        launch_diag::PhaseTimer _pt_prepare(
            launch_diag::Phase::DirectPrepare, direct_decode_phase);
        st.ids_data = ids->data;
        st.n_used = (int) ids->ne[0];
        st.n_tokens = (int) ids->ne[1];
        st.active_experts.clear();
        if (st.n_used <= 0 || st.n_tokens <= 0) return false;

        std::vector<uint8_t> ids_host(ggml_nbytes(ids));
        {
            launch_diag::PhaseTimer _pt(
                launch_diag::Phase::DirectIdsD2H, direct_decode_phase);
            if (ids->buffer != nullptr && ggml_backend_buffer_is_host(ids->buffer)) {
                std::memcpy(ids_host.data(), ids->data, ids_host.size());
            } else {
                check_cuda(cudaMemcpyAsync(ids_host.data(), ids->data, ids_host.size(),
                                           cudaMemcpyDeviceToHost, stream),
                           "ids D2H");
                check_cuda(cudaStreamSynchronize(stream), "ids D2H sync");
            }
        }

        for (int t = 0; t < st.n_tokens; ++t) {
            for (int u = 0; u < st.n_used; ++u) {
                const int32_t * p = reinterpret_cast<const int32_t *>(
                    ids_host.data() + (size_t)t * ids->nb[1] +
                    (size_t)u * ids->nb[0]);
                const int eid = *p;
                if (eid < 0) {
                    throw std::runtime_error("qwen3_baseline: negative expert id");
                }
                if (std::find(st.active_experts.begin(),
                              st.active_experts.end(), eid) ==
                    st.active_experts.end()) {
                    st.active_experts.push_back(eid);
                }
            }
        }

        for (int k = 0; k < 3; ++k) st.slice_bytes[k] = 0;
        const int expert_cid = cid_chunk(0);
        for (int eid : st.active_experts) {
            const std::string wid = expert_wid(layer, eid);
            qwen3::scheduler_reserve_for_dispatch(
                rt_->scheduler(), wid, expert_cid);
        }

        try {
            {
                launch_diag::PhaseTimer _pt(
                    launch_diag::Phase::DirectLoad, direct_decode_phase);
                struct ScopedDecodePhase {
                    bool prev;
                    explicit ScopedDecodePhase(bool decode)
                        : prev(launch_diag::current_decode_phase()) {
                        launch_diag::set_current_decode_phase(decode);
                    }
                    ~ScopedDecodePhase() {
                        launch_diag::set_current_decode_phase(prev);
                    }
                } _decode_phase(direct_decode_phase);
                for (int eid : st.active_experts) {
                    const std::string wid = expert_wid(layer, eid);
                    rt_->move_chunk(wid, expert_cid, Tier::RAM, Tier::VRAM, stream_h);
                    rt_->pool().wait_on_stream(wid, expert_cid, stream_h);
                    qwen3::scheduler_touch_resident(
                        rt_->scheduler(), wid, expert_cid, 0);
                }
            }

            {
                launch_diag::PhaseTimer _pt(
                    launch_diag::Phase::DirectD2D, direct_decode_phase);
                for (int eid : st.active_experts) {
                    const std::string wid = expert_wid(layer, eid);
                    ChunkedTensor * tensor = rt_->tensor(wid);
                    if (tensor == nullptr || !tensor->host().direct_expert_block) {
                        throw std::runtime_error(
                            "qwen3_direct_stock_v1: missing direct expert block " + wid);
                    }
                    const auto & host = tensor->host();
                    auto h = rt_->pool().view(wid, expert_cid);
                    if (h.device_ptr == nullptr) {
                        throw std::runtime_error(
                            "qwen3_direct_stock_v1: expert block not resident " + wid);
                    }
                    for (int k = 0; k < 3; ++k) {
                        const auto & r = host.expert_records[k];
                        if (r.nbytes == 0) {
                            throw std::runtime_error(
                                "qwen3_direct_stock_v1: empty expert subrecord " + wid);
                        }
                        if (st.slice_bytes[k] == 0) st.slice_bytes[k] = r.nbytes;
                        if (st.slice_bytes[k] != r.nbytes) {
                            throw std::runtime_error(
                                "qwen3_direct_stock_v1: inconsistent expert slice size");
                        }
                        allocate_staging_(k, (size_t) src0->ne[2] * r.nbytes);
                        check_cuda(cudaMemcpyAsync(
                            (char *) staging_[k] + (size_t) eid * r.nbytes,
                            (const char *) h.device_ptr + r.offset,
                            r.nbytes,
                            cudaMemcpyDeviceToDevice,
                            stream),
                            "expert D2D hydrate");
                    }
                }
            }
            rt_->pool().record_compute_event(stream_h);
        } catch (...) {
            for (int eid : st.active_experts) {
                qwen3::scheduler_release_from_dispatch(
                    rt_->scheduler(), expert_wid(layer, eid), expert_cid);
            }
            throw;
        }
        for (int eid : st.active_experts) {
            qwen3::scheduler_release_from_dispatch(
                rt_->scheduler(), expert_wid(layer, eid), expert_cid);
        }
    }

    if (kind >= 0 && kind < 3 && staging_[kind] != nullptr) {
        src0->data = staging_[kind];
    }
    return false;
}

}}  // namespace streamllm_ext::qwen3
