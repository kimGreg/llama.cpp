// streamllm-ext — M1 cutover S5 hard-gate POC: compute-node-fed sentinel.
//
// Throwaway exploratory test that decides whether the M1 cutover
// (S5+S6+S7) can use a GGML_OP_DUP sentinel with manually-wired
// src[1..3] when those srcs are **outputs of upstream compute nodes**
// (the router/topk mini-graph) rather than host-seeded leaves.
//
// The original Step 7-POC (test_sentinel_poc.cpp) verified six
// sentinel properties for host-seeded src[1..3]. This POC extends
// that to the production case and asserts SEVEN properties:
//
//   1. **Dependency ordering** — manually-wired src[1..3] from
//      compute nodes are scheduled BEFORE the sentinel in the
//      topological sort.
//   2. **Sentinel claiming** — pre_op_hook intercepts the sentinel
//      by name and returns true.
//   3. **Stock-op suppression** — when the hook returns true, the
//      default GGML_OP_DUP kernel does NOT execute. Verified by
//      pre-filling dst with a poison value and confirming the
//      executor-written value survives.
//   4. **Executor dst write + downstream consumption** — the hook
//      writes into sentinel->dst; a downstream consumer
//      (add(sentinel, sentinel)) reads that value correctly.
//   5. **CUDA graph capture disablement** — cudaStreamIsCapturing
//      reports cudaStreamCaptureStatusNone inside the hook
//      (user_node_claims_hook keeps capture disabled for
//      sentinel-containing cgraphs).
//   6. **CUDA backend behavior** — the entire flow runs on
//      ggml_backend_cuda, not the CPU backend. The hook fires on
//      the CUDA compute stream; src[1..3] live on CUDA device
//      memory; the consumer reads CUDA device memory.
//   7. **Legitimate src[1..3] attachment** — manually setting
//      `sentinel->src[i]` after ggml_dup actually plumbs through
//      ggml's allocator and topo sort. src[1..3] receive real
//      device buffers AND get their compute nodes executed before
//      the sentinel.
//
// Pass criterion: all 7 properties hold.  Fail criterion: any of
// (1), (3), (4), (5), (7) fail (the others are softer indicators).
// On fail: replan around GGML_OP_CUSTOM or another explicit
// sentinel mechanism before writing any production S5/S6 code.
//
// Build target: streamllm-sentinel-compute-poc
//
// Run: CUDA_VISIBLE_DEVICES=7 streamllm-sentinel-compute-poc
//
// Exits 0 on all-pass, 1 on any failure.

#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cuda.h>

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

// Test parameters — small but exercising real compute nodes.
constexpr int   kK         = 16;      // contraction dim of mul_mat
constexpr int   kM         = 8;       // output dim
constexpr int   kNTokens   = 4;       // batch dim
constexpr float kHookPat   = 42.0f;   // sentinel hook writes this into dst
constexpr float kPoisonPat = 99.0f;   // dst pre-filled with this; survives only
                                       // if default DUP didn't run
constexpr float kInputPat  = 7.0f;    // x value; if default DUP ran, dst becomes this

// Captured state at hook-fire time.
struct PocState {
    bool         hook_fired              = false;
    int          hook_call_count         = 0;
    bool         capture_status_ok       = false;
    bool         aux1_data_nonnull       = false;
    bool         aux2_data_nonnull       = false;
    bool         aux3_data_nonnull       = false;
    bool         aux1_value_ok           = false;
    bool         aux2_value_ok           = false;
    bool         aux3_value_ok           = false;
    cudaStream_t hook_stream             = nullptr;
    int          n_default_ops_observed  = 0;  // count of non-sentinel ops the hook saw
};
static PocState g_state;

// References for expected aux values. The cgraph computes:
//   tmp  = W * x          [M, n_tokens]
//   aux1 = relu(tmp)
//   aux2 = scale(tmp, 2)
//   aux3 = neg(tmp)
// We compute the same on host to compare.
static std::vector<float> g_expected_aux1;
static std::vector<float> g_expected_aux2;
static std::vector<float> g_expected_aux3;

bool tensor_near(const std::vector<float> & got,
                 const std::vector<float> & expected,
                 float tol = 1e-4f)
{
    if (got.size() != expected.size()) return false;
    for (size_t i = 0; i < got.size(); ++i) {
        if (std::fabs(got[i] - expected[i]) > tol) {
            std::fprintf(stderr,
                "  tensor diff at i=%zu: got=%f expected=%f\n",
                i, got[i], expected[i]);
            return false;
        }
    }
    return true;
}

// Pre-op hook: claim sentinel nodes only.
extern "C" bool sentinel_compute_pre_op_hook(
    cudaStream_t stream, struct ggml_tensor * dst)
{
    if (dst == nullptr || dst->name[0] == '\0') return false;
    if (std::strncmp(dst->name, "streamllm.moe_layer_", 20) != 0) {
        // Property 6 indicator — the hook is being asked about
        // every op on the CUDA stream, not just the sentinel.
        ++g_state.n_default_ops_observed;
        return false;
    }

    g_state.hook_fired = true;
    ++g_state.hook_call_count;
    g_state.hook_stream = stream;

    // Property 5: capture disabled.
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &cap) == cudaSuccess) {
        g_state.capture_status_ok = (cap == cudaStreamCaptureStatusNone);
    }

    // Property 1 + 7: src[1..3] are concrete compute outputs at this
    // point. Read each via D2H and compare against the host-computed
    // reference. If src[i] wasn't scheduled before the sentinel, the
    // values won't match.
    auto check_src = [&](int idx,
                          const std::vector<float> & expected,
                          bool * data_nonnull,
                          bool * value_ok) -> void {
        *data_nonnull = false;
        *value_ok     = false;
        const ggml_tensor * a = dst->src[idx];
        if (a == nullptr || a->data == nullptr) return;
        *data_nonnull = true;
        const size_t nbytes = ggml_nbytes(a);
        std::vector<float> host(nbytes / sizeof(float));
        if (cudaMemcpyAsync(host.data(), a->data, nbytes,
                             cudaMemcpyDeviceToHost, stream)
            != cudaSuccess) return;
        if (cudaStreamSynchronize(stream) != cudaSuccess) return;
        *value_ok = tensor_near(host, expected);
    };
    check_src(1, g_expected_aux1, &g_state.aux1_data_nonnull, &g_state.aux1_value_ok);
    check_src(2, g_expected_aux2, &g_state.aux2_data_nonnull, &g_state.aux2_value_ok);
    check_src(3, g_expected_aux3, &g_state.aux3_data_nonnull, &g_state.aux3_value_ok);

    // Property 3 + 4: write the hook pattern into dst on stream. The
    // downstream consumer will read this.
    const size_t dst_bytes = ggml_nbytes(dst);
    std::vector<float> dst_host(dst_bytes / sizeof(float), kHookPat);
    cudaMemcpyAsync(dst->data, dst_host.data(), dst_bytes,
                     cudaMemcpyHostToDevice, stream);

    // Claim — default DUP must not run.
    return true;
}

// user_node_claims hook: tells ggml-cuda to disable capture for
// any cgraph containing a sentinel node. This is what keeps
// property 5 true.
extern "C" bool sentinel_compute_user_node_claims(const ggml_tensor * node) {
    if (node == nullptr || node->name[0] == '\0') return false;
    return std::strncmp(node->name, "streamllm.moe_layer_", 20) == 0;
}

void fail(const char * prop, const char * detail) {
    std::fprintf(stderr,
        "test_sentinel_compute_poc: FAIL — %s — %s\n", prop, detail);
    std::exit(1);
}

}  // anon

int main() {
    std::fprintf(stderr, "test_sentinel_compute_poc: starting\n");

    // Property 6 (initial): force the CUDA backend.
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (backend == nullptr) {
        std::fprintf(stderr,
            "test_sentinel_compute_poc: ggml_backend_cuda_init failed\n");
        return 1;
    }

    ggml_init_params iparams{};
    iparams.mem_size   = 2048 * ggml_tensor_overhead() + ggml_graph_overhead();
    iparams.mem_buffer = nullptr;
    iparams.no_alloc   = true;
    ggml_context * ctx = ggml_init(iparams);

    // Inputs: W [K, M], x [K, n_tokens]
    ggml_tensor * W = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kK, kM);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kK, kNTokens);
    ggml_set_input(W);
    ggml_set_input(x);
    ggml_set_name(W, "W");
    ggml_set_name(x, "x");

    // Compute nodes that feed src[1..3]:
    //   tmp  = mul_mat(W, x)         -> [M, n_tokens]
    //   aux1 = relu(tmp)
    //   aux2 = scale(tmp, 2.0)
    //   aux3 = neg(tmp)
    ggml_tensor * tmp  = ggml_mul_mat(ctx, W, x);
    ggml_set_name(tmp, "tmp");
    ggml_tensor * aux1 = ggml_relu(ctx, tmp);
    ggml_tensor * aux2 = ggml_scale(ctx, tmp, 2.0f);
    ggml_tensor * aux3 = ggml_neg(ctx, tmp);
    ggml_set_name(aux1, "aux1_relu");
    ggml_set_name(aux2, "aux2_scale");
    ggml_set_name(aux3, "aux3_neg");

    // Build the sentinel as ggml_dup(x), shape [K, n_tokens]. Manually
    // wire src[1..3] to the compute outputs.
    ggml_tensor * sentinel = ggml_dup(ctx, x);
    sentinel->src[1] = aux1;
    sentinel->src[2] = aux2;
    sentinel->src[3] = aux3;
    ggml_set_name(sentinel, "streamllm.moe_layer_0");
    ggml_set_output(sentinel);

    // Consumer downstream of the sentinel.
    ggml_tensor * consumer = ggml_add(ctx, sentinel, sentinel);
    ggml_set_name(consumer, "consumer_b");
    ggml_set_output(consumer);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, consumer);

    // Property 1 (topo check): walk the graph and verify aux1/aux2/aux3
    // appear before sentinel, and sentinel appears before consumer.
    {
        const int n_nodes = ggml_graph_n_nodes(gf);
        int pos_aux1 = -1, pos_aux2 = -1, pos_aux3 = -1;
        int pos_tmp  = -1, pos_sentinel = -1, pos_consumer = -1;
        for (int i = 0; i < n_nodes; ++i) {
            const ggml_tensor * n = ggml_graph_node(gf, i);
            if (n == tmp)      pos_tmp = i;
            if (n == aux1)     pos_aux1 = i;
            if (n == aux2)     pos_aux2 = i;
            if (n == aux3)     pos_aux3 = i;
            if (n == sentinel) pos_sentinel = i;
            if (n == consumer) pos_consumer = i;
        }
        std::fprintf(stderr,
            "test_sentinel_compute_poc: n_nodes=%d  positions: "
            "tmp=%d aux1=%d aux2=%d aux3=%d sentinel=%d consumer=%d\n",
            n_nodes, pos_tmp, pos_aux1, pos_aux2, pos_aux3,
            pos_sentinel, pos_consumer);
        if (pos_sentinel < 0 || pos_consumer < 0 ||
            pos_aux1 < 0 || pos_aux2 < 0 || pos_aux3 < 0) {
            fail("property-1/7 topo (presence)",
                 "some compute node or sentinel missing from cgraph");
        }
        if (!(pos_aux1 < pos_sentinel && pos_aux2 < pos_sentinel &&
              pos_aux3 < pos_sentinel)) {
            fail("property-1 topo ordering",
                 "manually-wired src[1..3] are NOT scheduled before the sentinel");
        }
        if (pos_sentinel >= pos_consumer) {
            fail("property-1 topo ordering",
                 "sentinel not scheduled before its consumer");
        }
        std::fprintf(stderr,
            "test_sentinel_compute_poc: P1 OK — aux1/aux2/aux3 < sentinel < consumer\n");
    }

    // Allocate backend buffers. Property 7 check: src[1..3] each get
    // a real device buffer.
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) {
        fail("backend buffer alloc",
             "ggml_backend_alloc_ctx_tensors returned nullptr");
    }
    if (aux1->data == nullptr || aux2->data == nullptr || aux3->data == nullptr) {
        fail("property-7 src[1..3] allocator",
             "ggml allocator did NOT back manually-wired src[1..3] with device buffers");
    }
    std::fprintf(stderr,
        "test_sentinel_compute_poc: P7 OK — src[1..3] each got a device buffer\n");

    // Seed inputs.
    {
        std::vector<float> W_host((size_t) kK * kM);
        std::vector<float> x_host((size_t) kK * kNTokens);
        for (int i = 0; i < (int)W_host.size(); ++i) {
            // Mix of positive and negative so relu has something to clip.
            W_host[i] = (i % 5) - 2;  // values in {-2,-1,0,1,2}
        }
        for (auto & v : x_host) v = kInputPat;  // constant for easy reference
        ggml_backend_tensor_set(W, W_host.data(), 0, W_host.size() * sizeof(float));
        ggml_backend_tensor_set(x, x_host.data(), 0, x_host.size() * sizeof(float));

        // Compute reference on host: tmp = W * x  ([M, n_tokens]).
        // ggml_mul_mat(W [K, M], x [K, n_tokens]) → [M, n_tokens]
        // tmp[m, t] = sum_k W[k, m] * x[k, t]
        std::vector<float> tmp_h((size_t) kM * kNTokens, 0.0f);
        for (int t = 0; t < kNTokens; ++t) {
            for (int m = 0; m < kM; ++m) {
                float acc = 0.f;
                for (int k = 0; k < kK; ++k) {
                    acc += W_host[(size_t) m * kK + k] *
                           x_host[(size_t) t * kK + k];
                }
                tmp_h[(size_t) t * kM + m] = acc;
            }
        }
        g_expected_aux1.resize(tmp_h.size());
        g_expected_aux2.resize(tmp_h.size());
        g_expected_aux3.resize(tmp_h.size());
        for (size_t i = 0; i < tmp_h.size(); ++i) {
            g_expected_aux1[i] = tmp_h[i] > 0 ? tmp_h[i] : 0.f;  // relu
            g_expected_aux2[i] = 2.0f * tmp_h[i];                 // scale
            g_expected_aux3[i] = -tmp_h[i];                       // neg
        }
    }

    // Pre-fill sentinel's dst with poison so we can detect whether the
    // default DUP overwrote it. If DUP ran, dst would now contain x
    // (kInputPat). If only the hook ran, dst contains kHookPat. If
    // neither, poison survives — that's a separate failure mode (hook
    // didn't fire).
    {
        std::vector<float> poison(ggml_nbytes(sentinel) / sizeof(float),
                                   kPoisonPat);
        ggml_backend_tensor_set(sentinel, poison.data(), 0,
                                 poison.size() * sizeof(float));
    }

    // Install both hooks: pre_op_hook to claim sentinels, user_node_claims
    // to disable capture for cgraphs containing them.
    ggml_cuda_set_pre_op_hook((void *) &sentinel_compute_pre_op_hook);
    ggml_cuda_set_user_node_claims_hook((void *) &sentinel_compute_user_node_claims);

    // Compute. Property 6 confirmed by going through ggml_backend_cuda.
    ggml_status st = ggml_backend_graph_compute(backend, gf);

    ggml_cuda_set_pre_op_hook(nullptr);
    ggml_cuda_set_user_node_claims_hook(nullptr);

    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr,
            "test_sentinel_compute_poc: compute -> status=%d\n", (int) st);
        fail("compute", "graph compute failed");
    }

    // Property 2: hook fired.
    if (!g_state.hook_fired) {
        fail("property-2 sentinel claim",
             "pre_op_hook never fired on the sentinel");
    }
    if (g_state.hook_call_count != 1) {
        std::fprintf(stderr,
            "test_sentinel_compute_poc: hook fired %d times, expected 1\n",
            g_state.hook_call_count);
        fail("property-2 sentinel claim",
             "pre_op_hook fired the wrong number of times on the sentinel");
    }
    std::fprintf(stderr,
        "test_sentinel_compute_poc: P2 OK — hook fired exactly once on sentinel\n");

    // Property 5: capture disabled.
    if (!g_state.capture_status_ok) {
        fail("property-5 capture disabled",
             "stream was in cudaStreamCaptureStatusActive inside the hook");
    }
    std::fprintf(stderr,
        "test_sentinel_compute_poc: P5 OK — stream not in capture\n");

    // Property 6: ran on CUDA backend; hook saw non-sentinel ops too.
    if (g_state.n_default_ops_observed == 0) {
        std::fprintf(stderr,
            "test_sentinel_compute_poc: hook saw %d non-sentinel ops "
            "— pre_op_hook fires for ALL ops on the CUDA backend, "
            "expected > 0\n",
            g_state.n_default_ops_observed);
        fail("property-6 cuda backend",
             "pre_op_hook only saw the sentinel; expected non-sentinel "
             "ops too (mul_mat / relu / scale / neg / add). Check backend.");
    }
    std::fprintf(stderr,
        "test_sentinel_compute_poc: P6 OK — CUDA backend, hook saw %d non-sentinel ops + the sentinel\n",
        g_state.n_default_ops_observed);

    // Properties 1 + 7: src[1..3] had real data with correct values at
    // hook-fire time.
    auto check_aux = [&](int idx, bool data_ok, bool val_ok) {
        if (!data_ok) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "src[%d]->data was nullptr at hook fire", idx);
            fail("property-7 src[i] allocator", buf);
        }
        if (!val_ok) {
            char buf[96];
            std::snprintf(buf, sizeof(buf),
                "src[%d] did not carry the expected compute output at hook fire", idx);
            fail("property-1 dependency ordering / property-7 src[i] data", buf);
        }
    };
    check_aux(1, g_state.aux1_data_nonnull, g_state.aux1_value_ok);
    check_aux(2, g_state.aux2_data_nonnull, g_state.aux2_value_ok);
    check_aux(3, g_state.aux3_data_nonnull, g_state.aux3_value_ok);
    std::fprintf(stderr,
        "test_sentinel_compute_poc: P1 + P7 OK — src[1..3] carry correct compute output at hook fire\n");

    // Read sentinel + consumer back.
    std::vector<float> sentinel_host(ggml_nbytes(sentinel) / sizeof(float));
    std::vector<float> consumer_host(ggml_nbytes(consumer) / sizeof(float));
    ggml_backend_tensor_get(sentinel, sentinel_host.data(), 0,
                             ggml_nbytes(sentinel));
    ggml_backend_tensor_get(consumer, consumer_host.data(), 0,
                             ggml_nbytes(consumer));

    // Property 3: default DUP did NOT run. sentinel still carries
    // kHookPat, not kInputPat (which DUP would have written by copying x).
    for (size_t i = 0; i < sentinel_host.size(); ++i) {
        if (std::fabs(sentinel_host[i] - kHookPat) > 1e-4f) {
            std::fprintf(stderr,
                "test_sentinel_compute_poc: sentinel[%zu] = %f (expected kHookPat=%f). "
                "If %.1f → kInputPat, the default DUP ran. If %.1f → kPoisonPat, "
                "neither the hook nor DUP wrote (impossible if hook_fired==true).\n",
                i, sentinel_host[i], kHookPat, kInputPat, kPoisonPat);
            fail("property-3 stock-op suppression",
                 "default GGML_OP_DUP overwrote the hook's value");
        }
    }
    std::fprintf(stderr,
        "test_sentinel_compute_poc: P3 OK — default DUP did NOT execute; hook value preserved\n");

    // Property 4: consumer = sentinel + sentinel = 2 * kHookPat.
    const float expected_consumer = 2.0f * kHookPat;
    for (size_t i = 0; i < consumer_host.size(); ++i) {
        if (std::fabs(consumer_host[i] - expected_consumer) > 1e-4f) {
            std::fprintf(stderr,
                "test_sentinel_compute_poc: consumer[%zu] = %f (expected %f)\n",
                i, consumer_host[i], expected_consumer);
            fail("property-4 downstream consumption",
                 "consumer did not read the hook-written sentinel value");
        }
    }
    std::fprintf(stderr,
        "test_sentinel_compute_poc: P4 OK — consumer reads %f = 2 * kHookPat\n",
        expected_consumer);

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);

    std::fprintf(stderr,
        "test_sentinel_compute_poc: ALL SEVEN PROPERTIES PASS — "
        "compute-node-fed sentinel design is viable; proceed with S5+S6+S7.\n");
    return 0;
}
