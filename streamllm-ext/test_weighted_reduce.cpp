// streamllm-ext — M1 cutover S4: correctness test for
// streamllm_ext::qwen3::launch_weighted_reduce_slots.
//
// Generates synthetic (slot_out, weights), computes a CPU reference,
// runs the GPU kernel, and asserts max_abs_diff < 1e-5.
//
// Shapes are picked to exercise:
//   - boundary case where M % M_TILE != 0
//   - n_used > 1 (the reduce axis)
//   - n_tokens > 1 (independent rows)
// Plus a tiny known-answer block at the end so a regression is
// caught even if the random seed changes.
//
// Build target: streamllm-weighted-reduce-test
//
// Run: CUDA_VISIBLE_DEVICES=7 streamllm-weighted-reduce-test
//
// Exits 0 on success, 1 on first mismatch.

#include "qwen3_moe_fused.h"   // launch_weighted_reduce_slots

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using streamllm_ext::qwen3::launch_weighted_reduce_slots;

namespace {

void check_cuda(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s: %s\n", what,
                     cudaGetErrorString(e));
        std::exit(1);
    }
}

// CPU reference: layer_out[t, m] = sum_u weights[t, u] * slot_out[t, u, m].
void weighted_reduce_cpu(
    const std::vector<float> & slot_out,   // [n_tokens, n_used, M]
    const std::vector<float> & weights,    // [n_tokens, n_used]
    std::vector<float>       & layer_out,  // [n_tokens, M]
    int n_tokens, int n_used, int M)
{
    layer_out.assign((size_t) n_tokens * M, 0.0f);
    for (int t = 0; t < n_tokens; ++t) {
        for (int m = 0; m < M; ++m) {
            float acc = 0.f;
            for (int u = 0; u < n_used; ++u) {
                acc += weights[(size_t) t * n_used + u]
                     * slot_out[((size_t) t * n_used + u) * M + m];
            }
            layer_out[(size_t) t * M + m] = acc;
        }
    }
}

bool run_case(int n_tokens, int n_used, int M, uint32_t seed,
              const char * tag, float tol = 1e-5f)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);

    std::vector<float> slot_out_h((size_t) n_tokens * n_used * M);
    std::vector<float> weights_h ((size_t) n_tokens * n_used);
    for (auto & v : slot_out_h) v = dist(rng);
    // Use sensible weights (positive, roughly summing to 1 per token —
    // matches what the renormalised topk weights look like in real
    // dispatches).
    for (int t = 0; t < n_tokens; ++t) {
        float sum = 0.f;
        for (int u = 0; u < n_used; ++u) {
            float w = std::abs(dist(rng)) + 0.01f;
            weights_h[(size_t) t * n_used + u] = w;
            sum += w;
        }
        for (int u = 0; u < n_used; ++u) {
            weights_h[(size_t) t * n_used + u] /= sum;
        }
    }

    std::vector<float> layer_out_ref;
    weighted_reduce_cpu(slot_out_h, weights_h, layer_out_ref,
                       n_tokens, n_used, M);

    float * slot_out_d = nullptr;
    float * weights_d  = nullptr;
    float * layer_out_d = nullptr;
    check_cuda(cudaMalloc(&slot_out_d, slot_out_h.size() * sizeof(float)),
               "cudaMalloc slot_out");
    check_cuda(cudaMalloc(&weights_d, weights_h.size() * sizeof(float)),
               "cudaMalloc weights");
    check_cuda(cudaMalloc(&layer_out_d, layer_out_ref.size() * sizeof(float)),
               "cudaMalloc layer_out");
    check_cuda(cudaMemcpy(slot_out_d, slot_out_h.data(),
                          slot_out_h.size() * sizeof(float),
                          cudaMemcpyHostToDevice),
               "memcpy slot_out");
    check_cuda(cudaMemcpy(weights_d, weights_h.data(),
                          weights_h.size() * sizeof(float),
                          cudaMemcpyHostToDevice),
               "memcpy weights");
    // Pre-fill layer_out_d with a poison value so a kernel that
    // accidentally skips a row is caught even if the reference
    // would have been 0 there.
    {
        std::vector<float> poison(layer_out_ref.size(), -123.456f);
        check_cuda(cudaMemcpy(layer_out_d, poison.data(),
                              poison.size() * sizeof(float),
                              cudaMemcpyHostToDevice),
                   "memcpy poison");
    }

    cudaStream_t stream = 0;
    launch_weighted_reduce_slots(
        slot_out_d, weights_d, layer_out_d,
        n_tokens, n_used, M,
        (streamllm_ext::StreamHandle) stream);
    check_cuda(cudaStreamSynchronize(stream), "sync");

    std::vector<float> layer_out_gpu(layer_out_ref.size());
    check_cuda(cudaMemcpy(layer_out_gpu.data(), layer_out_d,
                          layer_out_gpu.size() * sizeof(float),
                          cudaMemcpyDeviceToHost),
               "memcpy back");

    float max_abs_diff = 0.f;
    int   first_bad    = -1;
    for (size_t i = 0; i < layer_out_ref.size(); ++i) {
        float d = std::fabs(layer_out_gpu[i] - layer_out_ref[i]);
        if (d > max_abs_diff) {
            max_abs_diff = d;
            if (first_bad < 0 && d > tol) first_bad = (int) i;
        }
    }

    cudaFree(slot_out_d);
    cudaFree(weights_d);
    cudaFree(layer_out_d);

    bool ok = (max_abs_diff <= tol);
    std::fprintf(stderr,
        "test_weighted_reduce[%s]: n_tokens=%d n_used=%d M=%d "
        "max_abs_diff=%.3e tol=%.1e %s\n",
        tag, n_tokens, n_used, M,
        max_abs_diff, tol, ok ? "PASS" : "FAIL");
    if (!ok) {
        int t = first_bad / M;
        int m = first_bad % M;
        std::fprintf(stderr,
            "  first mismatch at flat idx %d (t=%d, m=%d): "
            "ref=%.6f gpu=%.6f\n",
            first_bad, t, m,
            layer_out_ref[first_bad], layer_out_gpu[first_bad]);
    }
    return ok;
}

bool run_known_answer() {
    // 2 tokens, 3 slots, M=4. Easy to hand-check.
    const int n_tokens = 2, n_used = 3, M = 4;
    // weights[t][u]
    std::vector<float> w = {
        0.5f, 0.3f, 0.2f,    // token 0
        0.1f, 0.6f, 0.3f,    // token 1
    };
    // slot_out[t][u][m]
    std::vector<float> s = {
        // token 0
        1.f, 2.f, 3.f, 4.f,   // u=0
        5.f, 6.f, 7.f, 8.f,   // u=1
        9.f,10.f,11.f,12.f,   // u=2
        // token 1
       13.f,14.f,15.f,16.f,   // u=0
       17.f,18.f,19.f,20.f,   // u=1
       21.f,22.f,23.f,24.f,   // u=2
    };
    // Expected:
    //   t=0,m=0: 0.5*1 + 0.3*5 + 0.2*9  = 0.5 + 1.5 + 1.8 = 3.8
    //   t=0,m=1: 0.5*2 + 0.3*6 + 0.2*10 = 1.0 + 1.8 + 2.0 = 4.8
    //   t=0,m=2: 0.5*3 + 0.3*7 + 0.2*11 = 1.5 + 2.1 + 2.2 = 5.8
    //   t=0,m=3: 0.5*4 + 0.3*8 + 0.2*12 = 2.0 + 2.4 + 2.4 = 6.8
    //   t=1,m=0: 0.1*13 + 0.6*17 + 0.3*21 = 1.3 + 10.2 + 6.3 = 17.8
    //   t=1,m=1: 0.1*14 + 0.6*18 + 0.3*22 = 1.4 + 10.8 + 6.6 = 18.8
    //   t=1,m=2: 0.1*15 + 0.6*19 + 0.3*23 = 1.5 + 11.4 + 6.9 = 19.8
    //   t=1,m=3: 0.1*16 + 0.6*20 + 0.3*24 = 1.6 + 12.0 + 7.2 = 20.8
    std::vector<float> expected = {
         3.8f,  4.8f,  5.8f,  6.8f,
        17.8f, 18.8f, 19.8f, 20.8f,
    };

    float * sd = nullptr;
    float * wd = nullptr;
    float * yd = nullptr;
    check_cuda(cudaMalloc(&sd, s.size() * sizeof(float)), "alloc s");
    check_cuda(cudaMalloc(&wd, w.size() * sizeof(float)), "alloc w");
    check_cuda(cudaMalloc(&yd, expected.size() * sizeof(float)), "alloc y");
    check_cuda(cudaMemcpy(sd, s.data(), s.size() * sizeof(float),
                          cudaMemcpyHostToDevice), "memcpy s");
    check_cuda(cudaMemcpy(wd, w.data(), w.size() * sizeof(float),
                          cudaMemcpyHostToDevice), "memcpy w");

    launch_weighted_reduce_slots(
        sd, wd, yd, n_tokens, n_used, M,
        (streamllm_ext::StreamHandle) 0);
    check_cuda(cudaStreamSynchronize(0), "sync");

    std::vector<float> got(expected.size());
    check_cuda(cudaMemcpy(got.data(), yd, expected.size() * sizeof(float),
                          cudaMemcpyDeviceToHost), "memcpy back");

    cudaFree(sd); cudaFree(wd); cudaFree(yd);

    float max_d = 0.f;
    int   bad   = -1;
    for (size_t i = 0; i < expected.size(); ++i) {
        float d = std::fabs(got[i] - expected[i]);
        if (d > max_d) { max_d = d; if (d > 1e-5f && bad < 0) bad = (int) i; }
    }
    bool ok = (max_d <= 1e-5f);
    std::fprintf(stderr,
        "test_weighted_reduce[known-answer]: 2x3x4 hand-computed "
        "max_abs_diff=%.3e %s\n",
        max_d, ok ? "PASS" : "FAIL");
    if (!ok) {
        std::fprintf(stderr,
            "  first mismatch idx %d: ref=%.4f gpu=%.4f\n",
            bad, expected[bad], got[bad]);
    }
    return ok;
}

}  // anon

int main() {
    std::fprintf(stderr, "test_weighted_reduce: starting\n");

    bool all_ok = true;

    // Known answer first — catches gross indexing bugs.
    all_ok &= run_known_answer();

    // Qwen3-30B-A3B shape (M = hidden_dim/2 = ffn intermediate / 2 = 768
    // for the gate/up outputs; n_used = 8). Decode-batch size 1 and 32.
    all_ok &= run_case(/*n_tokens*/ 1,  /*n_used*/ 8, /*M*/ 768,  0xC0FFEEu, "qwen3-30b-a3b decode tg=1");
    all_ok &= run_case(/*n_tokens*/32,  /*n_used*/ 8, /*M*/ 768,  0xDEADu,   "qwen3-30b-a3b decode tg=32");

    // Boundary: M not a multiple of M_TILE=256. 768 IS a multiple, so
    // pick a non-multiple.
    all_ok &= run_case(/*n_tokens*/ 4,  /*n_used*/ 4, /*M*/ 300,  0x1234u,   "M-not-multiple-of-256");

    // n_used = 1 (single-expert edge case)
    all_ok &= run_case(/*n_tokens*/ 8,  /*n_used*/ 1, /*M*/ 256,  0x5678u,   "n_used=1");

    // Larger n_used (post-M1 archs could go higher)
    all_ok &= run_case(/*n_tokens*/ 2,  /*n_used*/16, /*M*/ 768,  0xBEEFu,   "n_used=16");

    std::fprintf(stderr, "test_weighted_reduce: %s\n",
                 all_ok ? "ALL PASS" : "FAILED");
    return all_ok ? 0 : 1;
}
