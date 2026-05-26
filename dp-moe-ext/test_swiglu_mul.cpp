// DPMoE — M1 cutover S6: correctness test for
// dp_moe_ext::qwen3::launch_swiglu_mul.
//
// Generates synthetic (slot_gate, slot_up), computes a CPU reference
// for silu(gate) * up, runs the GPU kernel, and asserts
// max_abs_diff < 1e-5 on random inputs (using __expf in CUDA can be
// a touch noisier than std::exp, so the tolerance is set conservatively).
//
// Also runs a known-answer block (silu(0)=0, silu(large positive)≈input)
// so a regression is caught even if the random seed changes.
//
// Build target: dp_moe-swiglu-mul-test
//
// Run: CUDA_VISIBLE_DEVICES=0 dp_moe-swiglu-mul-test
//
// Exits 0 on success, 1 on first mismatch.

#include "fused_kernels.h"   // launch_swiglu_mul

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using dp_moe_ext::qwen3::launch_swiglu_mul;

namespace {

void check_cuda(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s: %s\n", what,
                     cudaGetErrorString(e));
        std::exit(1);
    }
}

// CPU reference: out[i] = silu(gate[i]) * up[i].
void swiglu_mul_cpu(const std::vector<float> & gate,
                     const std::vector<float> & up,
                     std::vector<float>       & out)
{
    out.resize(gate.size());
    for (std::size_t i = 0; i < gate.size(); ++i) {
        const float g   = gate[i];
        const float sig = 1.0f / (1.0f + std::exp(-g));
        out[i] = g * sig * up[i];
    }
}

bool run_case(std::size_t N, uint32_t seed, const char * tag,
              float tol = 5e-6f)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-3.f, 3.f);

    std::vector<float> gate_h(N), up_h(N);
    for (auto & v : gate_h) v = dist(rng);
    for (auto & v : up_h)   v = dist(rng);

    std::vector<float> ref;
    swiglu_mul_cpu(gate_h, up_h, ref);

    float * gate_d = nullptr;
    float * up_d   = nullptr;
    float * out_d  = nullptr;
    check_cuda(cudaMalloc(&gate_d, N * sizeof(float)), "malloc gate");
    check_cuda(cudaMalloc(&up_d,   N * sizeof(float)), "malloc up");
    check_cuda(cudaMalloc(&out_d,  N * sizeof(float)), "malloc out");
    check_cuda(cudaMemcpy(gate_d, gate_h.data(), N * sizeof(float),
                          cudaMemcpyHostToDevice), "memcpy gate");
    check_cuda(cudaMemcpy(up_d,   up_h.data(),   N * sizeof(float),
                          cudaMemcpyHostToDevice), "memcpy up");
    // Poison so a skipped tail is visible.
    {
        std::vector<float> poison(N, -987.654f);
        check_cuda(cudaMemcpy(out_d, poison.data(), N * sizeof(float),
                              cudaMemcpyHostToDevice), "memcpy poison");
    }

    cudaStream_t stream = 0;
    launch_swiglu_mul(gate_d, up_d, out_d, N,
                      (dp_moe_ext::StreamHandle) stream);
    check_cuda(cudaStreamSynchronize(stream), "sync");

    std::vector<float> gpu(N);
    check_cuda(cudaMemcpy(gpu.data(), out_d, N * sizeof(float),
                          cudaMemcpyDeviceToHost), "memcpy back");

    float max_abs_diff = 0.f;
    int   first_bad    = -1;
    for (std::size_t i = 0; i < N; ++i) {
        float d = std::fabs(gpu[i] - ref[i]);
        if (d > max_abs_diff) {
            max_abs_diff = d;
            if (first_bad < 0 && d > tol) first_bad = (int) i;
        }
    }

    cudaFree(gate_d);
    cudaFree(up_d);
    cudaFree(out_d);

    bool ok = (max_abs_diff <= tol);
    std::fprintf(stderr,
        "test_swiglu_mul[%s]: N=%zu max_abs_diff=%.3e tol=%.1e %s\n",
        tag, N, max_abs_diff, tol, ok ? "PASS" : "FAIL");
    if (!ok && first_bad >= 0) {
        std::fprintf(stderr,
            "  first mismatch at i=%d: gate=%.6f up=%.6f ref=%.6f gpu=%.6f\n",
            first_bad, gate_h[first_bad], up_h[first_bad],
            ref[first_bad], gpu[first_bad]);
    }
    return ok;
}

bool run_known_answer() {
    // Hand-checkable values.
    //  i=0: gate=0  → silu(0)=0       → out = 0 * up = 0
    //  i=1: gate=10 → silu(10)≈10     → out ≈ 10 * up
    //  i=2: gate=-10→ silu(-10)≈0     → out ≈ 0
    //  i=3: gate=1  → silu(1)=0.73106 → out = 0.73106 * up
    const std::size_t N = 4;
    std::vector<float> g = { 0.f, 10.f, -10.f, 1.f };
    std::vector<float> u = { 7.f,  3.f,   5.f, 2.f };

    float * gate_d = nullptr;
    float * up_d   = nullptr;
    float * out_d  = nullptr;
    check_cuda(cudaMalloc(&gate_d, N * sizeof(float)), "malloc gate");
    check_cuda(cudaMalloc(&up_d,   N * sizeof(float)), "malloc up");
    check_cuda(cudaMalloc(&out_d,  N * sizeof(float)), "malloc out");
    check_cuda(cudaMemcpy(gate_d, g.data(), N * sizeof(float),
                          cudaMemcpyHostToDevice), "memcpy g");
    check_cuda(cudaMemcpy(up_d, u.data(), N * sizeof(float),
                          cudaMemcpyHostToDevice), "memcpy u");
    launch_swiglu_mul(gate_d, up_d, out_d, N,
                      (dp_moe_ext::StreamHandle) (cudaStream_t) 0);
    check_cuda(cudaStreamSynchronize(0), "sync");

    std::vector<float> gpu(N);
    check_cuda(cudaMemcpy(gpu.data(), out_d, N * sizeof(float),
                          cudaMemcpyDeviceToHost), "memcpy back");
    cudaFree(gate_d); cudaFree(up_d); cudaFree(out_d);

    // silu(1) * 2 ≈ 0.7310585786 * 2 ≈ 1.4621171
    bool ok = true;
    auto close = [](float a, float b, float t) {
        return std::fabs(a - b) <= t;
    };
    if (!close(gpu[0], 0.f, 1e-6f))                ok = false;
    if (!close(gpu[1], 10.f * 3.f, 5e-3f))         ok = false;  // silu(10)≈10 within 1e-4
    if (!close(gpu[2], 0.f, 5e-3f))                ok = false;  // silu(-10)≈0
    if (!close(gpu[3], 1.4621171f, 5e-5f))         ok = false;
    std::fprintf(stderr,
        "test_swiglu_mul[known-answer]: "
        "gpu = [%.6f, %.6f, %.6f, %.6f]  %s\n",
        gpu[0], gpu[1], gpu[2], gpu[3], ok ? "PASS" : "FAIL");
    return ok;
}

}  // anon

int main() {
    std::fprintf(stderr, "test_swiglu_mul: starting\n");

    bool all_ok = true;

    // Qwen3-30B-A3B decode shapes: n_tokens={1,32}, n_used=8, n_ff=768.
    // N covers per-slot intermediate size.
    all_ok &= run_case(/*N*/ 1 *  8 * 768, 0xC0FFEE, "qwen3-30b-a3b tg=1");
    all_ok &= run_case(/*N*/ 32 * 8 * 768, 0xDEAD,   "qwen3-30b-a3b tg=32");

    // Boundary not multiple of (threads*per_thread = 1024).
    all_ok &= run_case(/*N*/ 1023,         0xBEEF,   "boundary 1023");
    all_ok &= run_case(/*N*/ 1025,         0xCAFE,   "boundary 1025");

    // Tiny edge cases.
    all_ok &= run_case(/*N*/    1,         0x1234,   "single elem");
    all_ok &= run_case(/*N*/    4,         0x5678,   "four elems");

    all_ok &= run_known_answer();

    std::fprintf(stderr, "test_swiglu_mul: %s\n",
                 all_ok ? "ALL PASS" : "FAILED");
    return all_ok ? 0 : 1;
}
