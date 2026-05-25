// DPMoE — M1 cutover S3: router/topk equivalence test.
//
// Verifies that ``llm_build_moe_routing_softmax_topk`` (the factored
// helper in llama-graph.cpp) produces bit-exact ids and weights vs a
// hand-rolled reference that emits the same raw ggml ops in the same
// order. Both paths build into the same ggml_context and run on the
// same CUDA backend, so the only way they could diverge is if the
// helper's op sequence drifted from the reference. This test fails
// loudly on any such drift.
//
// Build target: dp_moe-router-equiv-test
//
// Run: CUDA_VISIBLE_DEVICES=0 dp_moe-router-equiv-test
//
// Exits 0 on PASS, 1 on first mismatch.

#include "llama-graph.h"     // llm_build_moe_routing_softmax_topk, llm_moe_router_output

#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cuda.h>

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

void check_cuda(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s: %s\n", what,
                     cudaGetErrorString(e));
        std::exit(1);
    }
}

// Hand-rolled reference matching the helper's op sequence verbatim.
// Returns the three output tensors (caller is responsible for the
// surrounding cgraph / allocator).
llm_moe_router_output reference_router_softmax_topk(
    ggml_context * ctx,
    ggml_tensor *  logits,
    int64_t        n_expert,
    int64_t        n_expert_used,
    bool           norm_w)
{
    const int64_t n_tokens = logits->ne[1];

    ggml_tensor * probs            = ggml_soft_max(ctx, logits);
    ggml_tensor * selected_experts = ggml_argsort_top_k(ctx, probs, n_expert_used);

    ggml_tensor * probs_3d = ggml_reshape_3d(ctx, probs, 1, n_expert, n_tokens);
    ggml_tensor * weights  = ggml_get_rows(ctx, probs_3d, selected_experts);

    if (norm_w) {
        weights = ggml_reshape_2d(ctx, weights, n_expert_used, n_tokens);
        ggml_tensor * weights_sum = ggml_sum_rows(ctx, weights);
        weights_sum = ggml_clamp(ctx, weights_sum, 6.103515625e-5, INFINITY);
        weights = ggml_div(ctx, weights, weights_sum);
        weights = ggml_reshape_3d(ctx, weights, 1, n_expert_used, n_tokens);
    }

    return { selected_experts, probs, weights };
}

bool run_case(int n_expert, int n_expert_used, int n_tokens,
              bool norm_w, uint32_t seed, const char * tag)
{
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (backend == nullptr) {
        std::fprintf(stderr, "test_router_equiv: ggml_backend_cuda_init failed\n");
        std::exit(1);
    }

    ggml_init_params iparams{};
    iparams.mem_size   = 1024 * ggml_tensor_overhead() + ggml_graph_overhead();
    iparams.mem_buffer = nullptr;
    iparams.no_alloc   = true;
    ggml_context * ctx = ggml_init(iparams);

    // Single shared logits input.
    ggml_tensor * logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                              n_expert, n_tokens);
    ggml_set_input(logits);
    ggml_set_name(logits, "logits_in");

    // Build both router subgraphs from the same input. Their outputs
    // are separate tensors in the same ggml_context — the backend
    // computes both in one cgraph.
    auto helper_out = llm_build_moe_routing_softmax_topk(
        ctx, logits, n_expert, n_expert_used, norm_w);
    auto ref_out    = reference_router_softmax_topk(
        ctx, logits, n_expert, n_expert_used, norm_w);

    ggml_set_name(helper_out.ids,     "helper_ids");
    ggml_set_name(helper_out.probs,   "helper_probs");
    ggml_set_name(helper_out.weights, "helper_weights");
    ggml_set_name(ref_out.ids,        "ref_ids");
    ggml_set_name(ref_out.probs,      "ref_probs");
    ggml_set_name(ref_out.weights,    "ref_weights");

    ggml_set_output(helper_out.ids);
    ggml_set_output(helper_out.probs);
    ggml_set_output(helper_out.weights);
    ggml_set_output(ref_out.ids);
    ggml_set_output(ref_out.probs);
    ggml_set_output(ref_out.weights);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, helper_out.ids);
    ggml_build_forward_expand(gf, helper_out.weights);
    ggml_build_forward_expand(gf, ref_out.ids);
    ggml_build_forward_expand(gf, ref_out.weights);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) {
        std::fprintf(stderr, "test_router_equiv[%s]: alloc_ctx_tensors failed\n", tag);
        std::exit(1);
    }

    // Seed logits with reproducible noise.
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-2.f, 2.f);
    std::vector<float> logits_host((size_t) n_expert * n_tokens);
    for (auto & v : logits_host) v = dist(rng);
    ggml_backend_tensor_set(logits, logits_host.data(), 0,
                             logits_host.size() * sizeof(float));

    ggml_status st = ggml_backend_graph_compute(backend, gf);
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "test_router_equiv[%s]: compute -> status=%d\n",
                     tag, (int)st);
        std::exit(1);
    }

    auto get_tensor = [&](ggml_tensor * t) {
        std::vector<uint8_t> bytes(ggml_nbytes(t));
        ggml_backend_tensor_get(t, bytes.data(), 0, bytes.size());
        return bytes;
    };

    auto helper_ids_bytes     = get_tensor(helper_out.ids);
    auto helper_probs_bytes   = get_tensor(helper_out.probs);
    auto helper_weights_bytes = get_tensor(helper_out.weights);
    auto ref_ids_bytes        = get_tensor(ref_out.ids);
    auto ref_probs_bytes      = get_tensor(ref_out.probs);
    auto ref_weights_bytes    = get_tensor(ref_out.weights);

    bool ok = true;

    // Criterion 5 (Q5 sign-off): ids bit-exact required.
    if (helper_ids_bytes != ref_ids_bytes) {
        std::fprintf(stderr,
            "test_router_equiv[%s]: ids MISMATCH (bit-exact required for criterion 5)\n",
            tag);
        const int32_t * h = (const int32_t *) helper_ids_bytes.data();
        const int32_t * r = (const int32_t *) ref_ids_bytes.data();
        int n = (int) (helper_ids_bytes.size() / sizeof(int32_t));
        int n_diff = 0;
        for (int i = 0; i < n; ++i) {
            if (h[i] != r[i]) {
                if (n_diff < 5) {
                    std::fprintf(stderr,
                        "  ids[%d]: helper=%d ref=%d\n", i, h[i], r[i]);
                }
                ++n_diff;
            }
        }
        std::fprintf(stderr, "  total mismatches: %d/%d\n", n_diff, n);
        ok = false;
    }

    // probs / weights: bit-exact on the deterministic CUDA backend
    // (Q5 sign-off: same backend, same primitives → bit-exact).
    if (helper_probs_bytes != ref_probs_bytes) {
        std::fprintf(stderr,
            "test_router_equiv[%s]: probs MISMATCH (bit-exact required)\n",
            tag);
        ok = false;
    }
    if (helper_weights_bytes != ref_weights_bytes) {
        std::fprintf(stderr,
            "test_router_equiv[%s]: weights MISMATCH (bit-exact required)\n",
            tag);
        ok = false;
    }

    std::fprintf(stderr,
        "test_router_equiv[%s]: n_expert=%d n_expert_used=%d n_tokens=%d norm_w=%d  %s\n",
        tag, n_expert, n_expert_used, n_tokens, (int)norm_w,
        ok ? "PASS" : "FAIL");

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return ok;
}

}  // anon

int main() {
    std::fprintf(stderr, "test_router_equiv: starting\n");

    bool all_ok = true;

    // Qwen3-30B-A3B shape: 128 experts, top-8, tg=1 / tg=32.
    all_ok &= run_case(/*n_expert*/128, /*n_used*/ 8, /*n_tokens*/ 1, /*norm_w*/true,  0xC0FFEEu, "qwen3-30b-a3b tg=1 norm_w");
    all_ok &= run_case(/*n_expert*/128, /*n_used*/ 8, /*n_tokens*/32, /*norm_w*/true,  0xDEADu,   "qwen3-30b-a3b tg=32 norm_w");

    // norm_w=false (no renormalisation) — exercises the alt branch in
    // the helper.
    all_ok &= run_case(/*n_expert*/128, /*n_used*/ 8, /*n_tokens*/ 4, /*norm_w*/false, 0x1234u,   "no-renorm");

    // Smaller config (edge case for shapes).
    all_ok &= run_case(/*n_expert*/  8, /*n_used*/ 2, /*n_tokens*/ 1, /*norm_w*/true,  0xBEEFu,   "small 8x2");
    all_ok &= run_case(/*n_expert*/  8, /*n_used*/ 2, /*n_tokens*/16, /*norm_w*/true,  0xCAFEu,   "small 8x2 batched");

    // Larger experts / more top-k (mixtral 8x22B shape ish).
    all_ok &= run_case(/*n_expert*/ 64, /*n_used*/ 6, /*n_tokens*/ 8, /*norm_w*/true,  0x5678u,   "mixtral-ish 64x6");

    std::fprintf(stderr, "test_router_equiv: %s\n",
                 all_ok ? "ALL PASS" : "FAILED");
    return all_ok ? 0 : 1;
}
