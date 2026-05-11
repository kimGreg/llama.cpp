// streamllm-ext — Step 7-POC: sentinel-dispatch proof-of-concept.
//
// Throwaway exploratory test that verifies the six sentinel properties
// listed in docs/MODE_A_MILESTONE1.md §"Step 7-POC". The mechanism
// under test is:
//
//   1. ggml_cuda_set_pre_op_hook(...)  — new ggml-cuda integration point
//      that fires before every op's default dispatch. The hook claims
//      sentinel-named GGML_OP_DUP nodes and short-circuits their kernel.
//   2. A 3-node cgraph:
//        sentinel_in  = ggml_new_tensor_2d(...)        (precomputed)
//        sentinel_aux = ggml_new_tensor_2d(...) × 3    (precomputed —
//                          will be wired as A->src[1..3])
//        A            = ggml_dup(sentinel_in)          (the SENTINEL,
//                          renamed "streamllm.moe_layer_0";
//                          src[1..3] wired manually)
//        B            = ggml_add(A, A)                 (consumer)
//
// Properties verified:
//   (1) GGML_OP_DUP with manually-wired src[1..3] parses + schedules.
//   (2) The pre_op_hook intercepts the sentinel (named match) and
//       returns true.
//   (3) Default ggml_cuda_dup does NOT execute: we pre-fill A's buffer
//       with a sentinel pattern 0xDEAD…, and the hook writes a different
//       pattern 0xCAFE…; after compute, A still carries 0xCAFE (hook
//       wrote it, DUP didn't overwrite).
//   (4) src[1..3] are computed BEFORE the sentinel: the hook reads
//       their device data and verifies it matches the precomputed
//       fingerprints.
//   (5) Consumer B = A + A reads the executor-written value of A —
//       so post-compute B[i] == 2 * 0xCAFE-pattern[i].
//   (6) CUDA graph capture is disabled — verified via
//       cudaStreamIsCapturing inside the hook (must report
//       cudaStreamCaptureStatusNone).
//
// On all-pass: prints "POC PASS — proceed with sentinel design".
// On any failure: prints the failing property + diagnostics, exits 1.

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

// Test parameters — small tensors to keep the POC self-contained.
constexpr int    kN          = 16;
constexpr int    kM          = 4;
constexpr float  kInPattern  = 7.0f;       // sentinel_in fill
constexpr float  kAux1Pat    = 11.0f;      // src[1] fingerprint
constexpr float  kAux2Pat    = 13.0f;      // src[2] fingerprint
constexpr float  kAux3Pat    = 17.0f;      // src[3] fingerprint
constexpr float  kDeadPat    = 99.0f;      // pre-fill A with this; if
                                            // default DUP runs it gets
                                            // overwritten with kInPattern
constexpr float  kHookPat    = 42.0f;      // hook writes this into A;
                                            // distinct from kInPattern
                                            // and kDeadPat so we can tell
                                            // which path wrote A.

// Property-tracking globals — read at end-of-test.
struct PocState {
    bool hook_fired           = false;
    bool capture_status_ok    = false;
    bool aux_src1_match       = false;
    bool aux_src2_match       = false;
    bool aux_src3_match       = false;
    int  hook_call_count      = 0;
};

static PocState g_state;

// Pre-op hook installed via ggml_cuda_set_pre_op_hook. Fires before
// EVERY op dispatch — we filter to the sentinel by name.
extern "C" bool sentinel_pre_op_hook(cudaStream_t stream, struct ggml_tensor * dst)
{
    if (dst == nullptr || dst->name[0] == '\0') return false;
    if (std::strncmp(dst->name, "streamllm.moe_layer_", 20) != 0) {
        return false;   // not a sentinel — let default dispatch run
    }

    ++g_state.hook_call_count;
    g_state.hook_fired = true;

    // Property 6: confirm we're NOT inside CUDA graph capture.
    cudaStreamCaptureStatus cap_status = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &cap_status) == cudaSuccess) {
        g_state.capture_status_ok = (cap_status == cudaStreamCaptureStatusNone);
    }

    // Property 4: src[1..3] must have concrete device data by now.
    // Read each via D2H + check fingerprint.
    auto check_aux = [&](int idx, float expected) -> bool {
        const struct ggml_tensor * a = dst->src[idx];
        if (a == nullptr || a->data == nullptr) return false;
        const size_t nbytes = ggml_nbytes(a);
        std::vector<float> host(nbytes / sizeof(float));
        cudaError_t e = cudaMemcpyAsync(host.data(), a->data,
                                         nbytes, cudaMemcpyDeviceToHost,
                                         stream);
        if (e != cudaSuccess) return false;
        if (cudaStreamSynchronize(stream) != cudaSuccess) return false;
        for (float v : host) {
            if (std::fabs(v - expected) > 1e-4f) return false;
        }
        return true;
    };
    g_state.aux_src1_match = check_aux(1, kAux1Pat);
    g_state.aux_src2_match = check_aux(2, kAux2Pat);
    g_state.aux_src3_match = check_aux(3, kAux3Pat);

    // Property 3 / 5: write the hook pattern into dst on the stream.
    // The consumer B = A + A will read this.
    const size_t dst_bytes = ggml_nbytes(dst);
    std::vector<float> dst_host(dst_bytes / sizeof(float), kHookPat);
    cudaMemcpyAsync(dst->data, dst_host.data(), dst_bytes,
                     cudaMemcpyHostToDevice, stream);

    return true;  // claim the op — default DUP does NOT run
}

void fail(const char * prop, const char * detail) {
    std::fprintf(stderr, "test_sentinel_poc: FAIL — %s — %s\n", prop, detail);
    std::exit(1);
}

}  // anonymous

int main() {
    std::fprintf(stderr, "test_sentinel_poc: starting\n");

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (backend == nullptr) {
        std::fprintf(stderr, "test_sentinel_poc: ggml_backend_cuda_init failed\n");
        return 1;
    }

    ggml_init_params iparams{};
    iparams.mem_size   = 1024 * ggml_tensor_overhead() + ggml_graph_overhead();
    iparams.mem_buffer = nullptr;
    iparams.no_alloc   = true;
    struct ggml_context * ctx = ggml_init(iparams);

    // Create the input + three aux tensors. All 2-D, all F32.
    struct ggml_tensor * sentinel_in   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kN, kM);
    struct ggml_tensor * sentinel_aux1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kN, kM);
    struct ggml_tensor * sentinel_aux2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kN, kM);
    struct ggml_tensor * sentinel_aux3 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kN, kM);
    ggml_set_input(sentinel_in);
    ggml_set_input(sentinel_aux1);
    ggml_set_input(sentinel_aux2);
    ggml_set_input(sentinel_aux3);
    ggml_set_name(sentinel_in,   "sentinel_in");
    ggml_set_name(sentinel_aux1, "sentinel_aux1");
    ggml_set_name(sentinel_aux2, "sentinel_aux2");
    ggml_set_name(sentinel_aux3, "sentinel_aux3");

    // Build the sentinel — a GGML_OP_DUP with src[0] = sentinel_in.
    // Manually wire src[1..3] = sentinel_aux{1,2,3} after construction.
    struct ggml_tensor * sentinel = ggml_dup(ctx, sentinel_in);
    sentinel->src[1] = sentinel_aux1;
    sentinel->src[2] = sentinel_aux2;
    sentinel->src[3] = sentinel_aux3;
    ggml_set_name(sentinel, "streamllm.moe_layer_0");
    ggml_set_output(sentinel);

    // Consumer B = sentinel + sentinel — reads the executor-written
    // value of A.
    struct ggml_tensor * b = ggml_add(ctx, sentinel, sentinel);
    ggml_set_name(b, "consumer_b");
    ggml_set_output(b);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, b);

    // Property 1: cgraph parses + topo-sorts. Verify the order:
    // src[0..3] of sentinel must all appear before sentinel; sentinel
    // must appear before b.
    int pos_sentinel = -1, pos_b = -1;
    const int n_nodes = ggml_graph_n_nodes(gf);
    for (int i = 0; i < n_nodes; ++i) {
        const ggml_tensor * n = ggml_graph_node(gf, i);
        if (n == sentinel)      pos_sentinel = i;
        if (n == b)             pos_b = i;
    }
    // sentinel_in / aux* are leaves and may not appear in gf->nodes —
    // ggml_graph only records computed nodes. The constraint we actually
    // care about is sentinel < b.
    if (pos_sentinel < 0 || pos_b < 0 || pos_sentinel >= pos_b) {
        std::fprintf(stderr,
            "test_sentinel_poc: topo order: sentinel=%d b=%d\n",
            pos_sentinel, pos_b);
        fail("property-1 topo order",
             "sentinel does not come before its consumer in cgraph");
    }
    std::fprintf(stderr,
        "test_sentinel_poc: P1 OK — n_nodes=%d, sentinel at %d, b at %d\n",
        n_nodes, pos_sentinel, pos_b);

    // Allocate backend buffers for the whole graph (inputs + outputs +
    // computed nodes). This MUST allocate src[1..3] because they're
    // referenced as srcs of a computed node (the sentinel), even though
    // GGML_OP_DUP canonically only reads src[0].
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) {
        fail("backend buffer alloc",
             "ggml_backend_alloc_ctx_tensors returned nullptr");
    }
    if (sentinel_in->data == nullptr || sentinel_aux1->data == nullptr ||
        sentinel_aux2->data == nullptr || sentinel_aux3->data == nullptr) {
        fail("property-1 src[1..3] allocator",
             "ggml allocator did not back src[1..3] with device buffers");
    }
    std::fprintf(stderr,
        "test_sentinel_poc: P1 OK — input + src[1..3] all allocated\n");

    // Seed values. sentinel_in = kInPattern; aux{1,2,3} = their fingerprints.
    auto fill = [&](struct ggml_tensor * t, float v) {
        const size_t bytes = ggml_nbytes(t);
        std::vector<float> host(bytes / sizeof(float), v);
        ggml_backend_tensor_set(t, host.data(), 0, bytes);
    };
    fill(sentinel_in,   kInPattern);
    fill(sentinel_aux1, kAux1Pat);
    fill(sentinel_aux2, kAux2Pat);
    fill(sentinel_aux3, kAux3Pat);
    // Pre-fill the sentinel's dst with kDeadPat — if the default DUP
    // runs anyway, it'll overwrite with kInPattern, which we'd detect.
    fill(sentinel, kDeadPat);

    // Install the pre-op hook.
    ggml_cuda_set_pre_op_hook((void *) &sentinel_pre_op_hook);

    // Compute the graph.
    ggml_status st = ggml_backend_graph_compute(backend, gf);
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr,
            "test_sentinel_poc: ggml_backend_graph_compute -> status=%d\n",
            (int)st);
        fail("compute", "graph compute failed");
    }
    // Unregister the hook (also covers null-safety check).
    ggml_cuda_set_pre_op_hook(nullptr);

    // Property 2: hook actually fired.
    if (!g_state.hook_fired) {
        fail("property-2 hook fired",
             "pre_op_hook was never invoked on the sentinel");
    }
    std::fprintf(stderr,
        "test_sentinel_poc: P2 OK — hook fired (%d calls)\n",
        g_state.hook_call_count);

    // Property 6: capture disabled.
    if (!g_state.capture_status_ok) {
        fail("property-6 capture disabled",
             "stream was in cudaStreamCaptureStatusActive when hook ran");
    }
    std::fprintf(stderr,
        "test_sentinel_poc: P6 OK — stream not in capture\n");

    // Property 4: src[1..3] visible to hook.
    if (!g_state.aux_src1_match || !g_state.aux_src2_match ||
        !g_state.aux_src3_match) {
        std::fprintf(stderr,
            "test_sentinel_poc: aux match (1,2,3) = (%d,%d,%d)\n",
            (int)g_state.aux_src1_match, (int)g_state.aux_src2_match,
            (int)g_state.aux_src3_match);
        fail("property-4 src[1..3]",
             "extra-src fingerprints missing — allocator/topo-sort issue");
    }
    std::fprintf(stderr, "test_sentinel_poc: P4 OK — src[1..3] match\n");

    // Read back the sentinel's dst and B.
    std::vector<float> sentinel_host(ggml_nbytes(sentinel) / sizeof(float));
    std::vector<float> b_host       (ggml_nbytes(b)        / sizeof(float));
    ggml_backend_tensor_get(sentinel, sentinel_host.data(), 0,
                             ggml_nbytes(sentinel));
    ggml_backend_tensor_get(b,        b_host.data(),        0,
                             ggml_nbytes(b));

    // Property 3: default DUP did NOT execute. The hook wrote kHookPat;
    // if the default kernel had run AFTER the hook it would now contain
    // kInPattern, not kHookPat.
    for (float v : sentinel_host) {
        if (std::fabs(v - kHookPat) > 1e-4f) {
            std::fprintf(stderr,
                "test_sentinel_poc: sentinel A = %f (expected %f = kHookPat)\n",
                v, kHookPat);
            fail("property-3 default DUP did not execute",
                 "sentinel value is not the hook-written pattern");
        }
    }
    std::fprintf(stderr,
        "test_sentinel_poc: P3 OK — default DUP did not overwrite dst\n");

    // Property 5: consumer B = A + A reads the executor-written A.
    const float expected_b = 2.0f * kHookPat;
    for (float v : b_host) {
        if (std::fabs(v - expected_b) > 1e-4f) {
            std::fprintf(stderr,
                "test_sentinel_poc: B = %f (expected %f = 2*kHookPat)\n",
                v, expected_b);
            fail("property-5 consumer reads executor-written dst",
                 "consumer did not see the hook-written sentinel value");
        }
    }
    std::fprintf(stderr,
        "test_sentinel_poc: P5 OK — consumer reads %f = 2*kHookPat\n",
        expected_b);

    // Tear down.
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);

    std::fprintf(stderr,
        "test_sentinel_poc: ALL SIX PROPERTIES PASS — "
        "proceed with sentinel design (DUP + pre_op_hook)\n");
    return 0;
}
