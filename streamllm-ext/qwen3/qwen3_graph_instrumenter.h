// streamllm-ext / qwen3 — graph instrumenter.
//
// Step-4b mechanism: at on_graph_compute_begin, walk the ggml cgraph
// and build a node→layer index map (parsed from each managed node's
// src0 weight name, "blk.<N>.<...>"). As ggml-cuda visits managed
// mul_mat / mul_mat_id nodes during compute, the dispatch glue calls
// ``on_managed_node_visit``; the instrumenter detects layer transitions
// and fires Scheduler::on_marker(MarkerKind::LayerBegin/LayerEnd).
//
// No ggml graph mutation — the instrumenter is purely observational.
// It synthesises semantic-boundary markers from per-op hook progress
// using the prewalk-built map.  This delivers the marker abstraction
// the architecture plan calls for without ggml-cuda surgery; the
// dispatch path keeps using the existing per-op hooks.

#pragma once

#include <unordered_map>

struct ggml_cgraph;
struct ggml_tensor;

namespace streamllm_ext {
class Scheduler;
}

namespace streamllm_ext { namespace qwen3 {

class GraphInstrumenter {
public:
    // Called from MoEScheduler::on_graph_compute_begin once per
    // graph compute. Walks ``cg`` and builds the node→layer map.
    // Resets the per-graph cursor.
    void on_graph_begin(const struct ggml_cgraph * cg);

    // Called from MoEScheduler::on_graph_compute_end. Flushes the
    // final LayerEnd marker (if a layer was active) so callers see a
    // well-formed Begin/End marker stream.  ``s`` and
    // ``compute_stream`` mirror the on_managed_node_visit args.
    void on_graph_end(Scheduler & s, void * compute_stream);

    // Called from the per-op dispatch glue (qwen3_moe_dispatch.cpp)
    // before any compute for a managed node.  ``dst`` is the node
    // tensor (the unique identifier for that cgraph node).  If
    // ``dst``'s layer differs from the last-seen layer, fires
    // LayerEnd (for the previous layer) + LayerBegin (for the new
    // layer) via ``s.on_marker``.
    void on_managed_node_visit(const struct ggml_tensor * dst,
                                Scheduler &              s,
                                void *                   compute_stream);

    // Diagnostics.
    int  total_layers() const { return total_layers_; }
    bool is_ready()     const { return !node_to_layer_.empty(); }

private:
    std::unordered_map<const struct ggml_tensor *, int> node_to_layer_;
    int last_layer_   = -1;
    int total_layers_ = 0;
};

}}  // namespace streamllm_ext::qwen3
