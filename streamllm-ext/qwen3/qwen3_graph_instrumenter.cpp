// streamllm-ext / qwen3 — graph instrumenter impl.

#include "qwen3_graph_instrumenter.h"
#include "scheduler.h"

#include <ggml.h>

#include <cstdlib>
#include <cstring>

namespace streamllm_ext { namespace qwen3 {

namespace {

// Parse the "blk.<N>." prefix from a tensor name. Returns -1 if the
// name doesn't follow that pattern (e.g. embed / output tensors).
int parse_layer_from_name(const char * name) {
    if (name == nullptr || std::strncmp(name, "blk.", 4) != 0) return -1;
    int v = std::atoi(name + 4);
    return v >= 0 ? v : -1;
}

}  // anonymous namespace

void GraphInstrumenter::on_graph_begin(const struct ggml_cgraph * cg) {
    node_to_layer_.clear();
    last_layer_   = -1;
    total_layers_ = 0;
    if (cg == nullptr) return;

    const int n = ggml_graph_n_nodes(const_cast<ggml_cgraph *>(cg));
    node_to_layer_.reserve((size_t)n);
    for (int i = 0; i < n; ++i) {
        const ggml_tensor * node = ggml_graph_node(
            const_cast<ggml_cgraph *>(cg), i);
        if (node == nullptr) continue;
        // Use src0 (the weight tensor for mul_mat / mul_mat_id) as
        // the layer-name source. Non-mul_mat managed paths can extend
        // this once they exist.
        const ggml_tensor * w = node->src[0];
        if (w == nullptr || w->name[0] == '\0') continue;
        int L = parse_layer_from_name(w->name);
        if (L < 0) continue;
        node_to_layer_.emplace(node, L);
        if (L + 1 > total_layers_) total_layers_ = L + 1;
    }
}

void GraphInstrumenter::on_graph_end(Scheduler & s, void * compute_stream) {
    if (last_layer_ >= 0) {
        MarkerEvent ev;
        ev.kind           = MarkerKind::LayerEnd;
        ev.layer_index    = last_layer_;
        ev.compute_stream = (StreamHandle) compute_stream;
        s.on_marker(ev);
    }
    last_layer_ = -1;
}

void GraphInstrumenter::on_managed_node_visit(
    const struct ggml_tensor * dst,
    Scheduler &              s,
    void *                   compute_stream)
{
    if (dst == nullptr) return;
    auto it = node_to_layer_.find(dst);
    if (it == node_to_layer_.end()) return;
    const int L = it->second;
    if (L == last_layer_) return;

    if (last_layer_ >= 0) {
        MarkerEvent end_ev;
        end_ev.kind           = MarkerKind::LayerEnd;
        end_ev.layer_index    = last_layer_;
        end_ev.compute_stream = (StreamHandle) compute_stream;
        s.on_marker(end_ev);
    }
    MarkerEvent begin_ev;
    begin_ev.kind           = MarkerKind::LayerBegin;
    begin_ev.layer_index    = L;
    begin_ev.compute_stream = (StreamHandle) compute_stream;
    s.on_marker(begin_ev);

    last_layer_ = L;
}

}}  // namespace streamllm_ext::qwen3
