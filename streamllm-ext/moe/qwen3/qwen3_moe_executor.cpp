// streamllm-ext / moe / qwen3 — Qwen3-MoE executor registration.

#include "qwen3_moe_executor.h"
#include "chunked_tensor.h"
#include "executor.h"        // register_executor
#include "runtime.h"
#include "stream_reader.h"

#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
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

} // namespace

void register_qwen3_moe_executor() {
    static bool once = false;
    if (once) return;
    once = true;
    auto factory = []() -> std::unique_ptr<ModelExecutor> {
        return std::unique_ptr<ModelExecutor>(new Qwen3MoEExecutor());
    };
    register_executor("qwen3_ss_anybcq_v1", factory);

    auto baseline_factory = []() -> std::unique_ptr<ModelExecutor> {
        return std::unique_ptr<ModelExecutor>(new Qwen3BaselineExecutor());
    };
    register_executor("qwen3_direct_stock_v1", baseline_factory);
}

Qwen3BaselineExecutor::~Qwen3BaselineExecutor() {
    for (void *& p : full_) {
        if (p != nullptr) {
            cudaFree(p);
            p = nullptr;
        }
    }
}

void Qwen3BaselineExecutor::bind_to_model(
    StreamllmRuntime & rt,
    const StreamReader &,
    const std::string &) {
    rt_ = &rt;
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

void Qwen3BaselineExecutor::ensure_full_(int kind, size_t bytes) {
    if (kind < 0 || kind >= 3) return;
    if (full_bytes_[kind] >= bytes) return;
    if (full_[kind] != nullptr) {
        cudaFree(full_[kind]);
        full_[kind] = nullptr;
        full_bytes_[kind] = 0;
    }
    check_cuda(cudaMalloc(&full_[kind], bytes), "full-stack cudaMalloc");
    full_bytes_[kind] = bytes;
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
    const bool need_prepare =
        st.ids_data != ids->data ||
        st.n_used != (int) ids->ne[0] ||
        st.n_tokens != (int) ids->ne[1] ||
        st.active_experts.empty();

    cudaStream_t stream = (cudaStream_t) stream_h;

    if (need_prepare) {
        st.ids_data = ids->data;
        st.n_used = (int) ids->ne[0];
        st.n_tokens = (int) ids->ne[1];
        st.active_experts.clear();
        if (st.n_used <= 0 || st.n_tokens <= 0) return false;

        std::vector<uint8_t> ids_host(ggml_nbytes(ids));
        check_cuda(cudaMemcpyAsync(ids_host.data(), ids->data, ids_host.size(),
                                   cudaMemcpyDeviceToHost, stream),
                   "ids D2H");
        check_cuda(cudaStreamSynchronize(stream), "ids D2H sync");

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

        for (int eid : st.active_experts) {
            const std::string wid = expert_wid(layer, eid);
            rt_->move_chunk(wid, 0, Tier::RAM, Tier::VRAM, stream_h);
            rt_->pool().wait_on_stream(wid, 0, stream_h);

            ChunkedTensor * tensor = rt_->tensor(wid);
            if (tensor == nullptr || !tensor->host().direct_expert_block) {
                throw std::runtime_error(
                    "qwen3_baseline: missing direct expert block " + wid);
            }
            const auto & host = tensor->host();
            auto h = rt_->pool().view(wid, 0);
            if (h.device_ptr == nullptr) {
                throw std::runtime_error(
                    "qwen3_baseline: expert block not resident " + wid);
            }
            for (int k = 0; k < 3; ++k) {
                const auto & r = host.expert_records[k];
                if (r.nbytes == 0) {
                    throw std::runtime_error(
                        "qwen3_baseline: empty expert subrecord " + wid);
                }
                if (st.slice_bytes[k] == 0) st.slice_bytes[k] = r.nbytes;
                if (st.slice_bytes[k] != r.nbytes) {
                    throw std::runtime_error(
                        "qwen3_baseline: inconsistent expert slice size");
                }
                ensure_full_(k, (size_t) src0->ne[2] * r.nbytes);
                check_cuda(cudaMemcpyAsync(
                    (char *) full_[k] + (size_t) eid * r.nbytes,
                    (const char *) h.device_ptr + r.offset,
                    r.nbytes,
                    cudaMemcpyDeviceToDevice,
                    stream),
                    "expert D2D hydrate");
            }
        }
    }

    if (kind >= 0 && kind < 3 && full_[kind] != nullptr) {
        src0->data = full_[kind];
    }
    return false;
}

}}  // namespace streamllm_ext::qwen3
