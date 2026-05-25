#include "fused_kernels.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

void check_cuda(cudaError_t err, const char * what) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(err));
        std::exit(1);
    }
}

std::vector<int> first_routed_order(
    const std::vector<int32_t> & ids,
    int n_tokens,
    int n_used,
    int n_expert) {
    std::vector<int> out;
    std::vector<int> seen((size_t) n_expert, 0);
    for (int t = 0; t < n_tokens; ++t) {
        for (int u = 0; u < n_used; ++u) {
            const int e = ids[(size_t) t * n_used + u];
            if (e < 0 || e >= n_expert || seen[(size_t) e]) continue;
            seen[(size_t) e] = 1;
            out.push_back(e);
        }
    }
    return out;
}

std::vector<int> reference_exact_shifted(
    const std::vector<int32_t> & ids,
    const std::vector<float> & weights,
    const std::vector<float> & R,
    int n_tokens,
    int n_used,
    int n_expert,
    int K_min,
    int K_max,
    float kbar) {
    const int n_K = K_max - K_min + 1;
    std::vector<float> sum_g2((size_t) n_expert, 0.0f);
    for (int t = 0; t < n_tokens; ++t) {
        for (int u = 0; u < n_used; ++u) {
            const int e = ids[(size_t) t * n_used + u];
            if (e < 0 || e >= n_expert) continue;
            float g = weights[(size_t) t * n_used + u];
            if (g < 0.0f) g = 0.0f;
            sum_g2[(size_t) e] += g * g;
        }
    }

    const std::vector<int> active =
        first_routed_order(ids, n_tokens, n_used, n_expert);
    std::vector<int> chunks((size_t) n_expert, 0);
    const int n = (int) active.size();
    if (n == 0) return chunks;

    long target_l = (long) std::lrint((double) n * (double) kbar);
    const long lo = (long) n * K_min;
    const long hi = (long) n * K_max;
    if (target_l < lo) target_l = lo;
    if (target_l > hi) target_l = hi;
    const int target_extra = (int) (target_l - lo);
    const int extra_max = K_max - K_min;
    const int B1 = target_extra + 1;
    constexpr float INF = 1.0e30f;

    std::vector<float> dp_prev((size_t) B1, INF);
    std::vector<float> dp_cur((size_t) B1, INF);
    std::vector<uint8_t> trace((size_t) n * B1, 0);
    dp_prev[0] = 0.0f;

    for (int i = 0; i < n; ++i) {
        std::fill(dp_cur.begin(), dp_cur.end(), INF);
        const int e = active[(size_t) i];
        const float g2 = sum_g2[(size_t) e];
        for (int nb = 0; nb <= target_extra; ++nb) {
            float best = INF;
            int best_x = 0;
            const int x_hi = std::min(extra_max, nb);
            for (int x = x_hi; x >= 0; --x) {
                const int prev_b = nb - x;
                const float base = dp_prev[(size_t) prev_b];
                if (base >= INF * 0.5f) continue;
                const float cost = base + g2 * R[(size_t) e * n_K + x];
                if (cost < best) {
                    best = cost;
                    best_x = x;
                }
            }
            dp_cur[(size_t) nb] = best;
            trace[(size_t) i * B1 + (size_t) nb] = (uint8_t) best_x;
        }
        dp_prev.swap(dp_cur);
    }

    int b = target_extra;
    for (int i = n - 1; i >= 0; --i) {
        const int e = active[(size_t) i];
        const int x = (int) trace[(size_t) i * B1 + (size_t) b];
        chunks[(size_t) e] = K_min + x;
        b -= x;
        if (b < 0) b = 0;
    }
    return chunks;
}

std::vector<int> reference_naive_unshifted(
    const std::vector<int32_t> & ids,
    const std::vector<float> & weights,
    const std::vector<float> & R,
    int n_tokens,
    int n_used,
    int n_expert,
    int K_min,
    int K_max,
    float kbar) {
    const int n_K = K_max - K_min + 1;
    std::vector<float> sum_g2((size_t) n_expert, 0.0f);
    for (int t = 0; t < n_tokens; ++t) {
        for (int u = 0; u < n_used; ++u) {
            const int e = ids[(size_t) t * n_used + u];
            if (e < 0 || e >= n_expert) continue;
            float g = weights[(size_t) t * n_used + u];
            if (g < 0.0f) g = 0.0f;
            sum_g2[(size_t) e] += g * g;
        }
    }

    const std::vector<int> active =
        first_routed_order(ids, n_tokens, n_used, n_expert);
    std::vector<int> chunks((size_t) n_expert, 0);
    const int n = (int) active.size();
    if (n == 0) return chunks;

    long target_l = (long) std::lrint((double) n * (double) kbar);
    const long lo = (long) n * K_min;
    const long hi = (long) n * K_max;
    if (target_l < lo) target_l = lo;
    if (target_l > hi) target_l = hi;
    const int target = (int) target_l;
    const int B1 = target + 1;
    constexpr float INF = 1.0e30f;

    std::vector<float> dp_prev((size_t) B1, INF);
    std::vector<float> dp_cur((size_t) B1, INF);
    std::vector<uint8_t> trace((size_t) n * B1, 0);
    dp_prev[0] = 0.0f;

    for (int i = 0; i < n; ++i) {
        std::fill(dp_cur.begin(), dp_cur.end(), INF);
        const int e = active[(size_t) i];
        const float g2 = sum_g2[(size_t) e];
        for (int prev_b = 0; prev_b <= target; ++prev_b) {
            const float base = dp_prev[(size_t) prev_b];
            if (base >= INF * 0.5f) continue;
            for (int k = 0; k < n_K; ++k) {
                const int K = K_min + k;
                const int nb = prev_b + K;
                if (nb > target) break;
                const float cost = base + g2 * R[(size_t) e * n_K + k];
                if (cost < dp_cur[(size_t) nb]) {
                    dp_cur[(size_t) nb] = cost;
                    trace[(size_t) i * B1 + (size_t) nb] = (uint8_t) K;
                }
            }
        }
        dp_prev.swap(dp_cur);
    }

    int b = target;
    for (int i = n - 1; i >= 0; --i) {
        const int e = active[(size_t) i];
        const int K = (int) trace[(size_t) i * B1 + (size_t) b];
        chunks[(size_t) e] = K;
        b -= K;
        if (b < 0) b = 0;
    }
    return chunks;
}

void run_case(
    const char * name,
    const std::vector<int32_t> & ids,
    const std::vector<float> & weights,
    int n_tokens,
    int n_used,
    int n_expert,
    int K_min,
    int K_max,
    float kbar) {
    const int n_K = K_max - K_min + 1;
    std::vector<float> R((size_t) n_expert * n_K, 0.0f);
    for (int e = 0; e < n_expert; ++e) {
        for (int k = 0; k < n_K; ++k) {
            // Monotone-ish but expert-dependent residual table. Exact DP
            // should prefer higher K for high-gate experts while preserving
            // exact-DP semantics.
            R[(size_t) e * n_K + k] =
                (float) (0.05 * (double) (n_K - k) +
                         0.001 * (double) ((e * 17 + k * 11) % 23));
        }
    }

    const std::vector<int> expected =
        reference_naive_unshifted(ids, weights, R, n_tokens, n_used, n_expert,
                                  K_min, K_max, kbar);
    const std::vector<int> shifted =
        reference_exact_shifted(ids, weights, R, n_tokens, n_used, n_expert,
                                K_min, K_max, kbar);
    if (shifted != expected) {
        std::fprintf(stderr, "%s: shifted reference differs from naive DP\n",
                     name);
        std::exit(1);
    }
    const std::vector<int> expected_order =
        first_routed_order(ids, n_tokens, n_used, n_expert);

    int32_t * ids_d = nullptr;
    float * weights_d = nullptr;
    float * R_d = nullptr;
    int * chunks_d = nullptr;
    int * order_d = nullptr;
    int * n_active_d = nullptr;
    float * dp_prev_d = nullptr;
    float * dp_cur_d = nullptr;
    uint8_t * trace_d = nullptr;

    const size_t ids_bytes = ids.size() * sizeof(int32_t);
    const size_t weights_bytes = weights.size() * sizeof(float);
    const size_t R_bytes = R.size() * sizeof(float);
    const size_t chunks_bytes = (size_t) n_expert * sizeof(int);
    const int B1 = n_expert * std::max(0, K_max - K_min) + 1;
    const size_t dp_bytes = (size_t) B1 * sizeof(float);
    const size_t trace_bytes = (size_t) n_expert * (size_t) B1 * sizeof(uint8_t);

    check_cuda(cudaMalloc(&ids_d, ids_bytes), "cudaMalloc ids");
    check_cuda(cudaMalloc(&weights_d, weights_bytes), "cudaMalloc weights");
    check_cuda(cudaMalloc(&R_d, R_bytes), "cudaMalloc R");
    check_cuda(cudaMalloc(&chunks_d, chunks_bytes), "cudaMalloc chunks");
    check_cuda(cudaMalloc(&order_d, chunks_bytes), "cudaMalloc order");
    check_cuda(cudaMalloc(&n_active_d, sizeof(int)), "cudaMalloc n_active");
    check_cuda(cudaMalloc(&dp_prev_d, dp_bytes), "cudaMalloc dp_prev");
    check_cuda(cudaMalloc(&dp_cur_d, dp_bytes), "cudaMalloc dp_cur");
    check_cuda(cudaMalloc(&trace_d, trace_bytes), "cudaMalloc trace");

    check_cuda(cudaMemcpy(ids_d, ids.data(), ids_bytes, cudaMemcpyHostToDevice),
               "copy ids");
    check_cuda(cudaMemcpy(weights_d, weights.data(), weights_bytes,
                          cudaMemcpyHostToDevice), "copy weights");
    check_cuda(cudaMemcpy(R_d, R.data(), R_bytes, cudaMemcpyHostToDevice),
               "copy R");

    dp_moe_ext::qwen3::launch_plan_chunks_kbar_exact_strided(
        ids_d,
        (size_t) n_used * sizeof(int32_t),
        weights_d,
        (size_t) n_used * sizeof(float),
        nullptr,
        0,
        R_d,
        K_min,
        K_max,
        0,
        kbar,
        n_tokens,
        n_used,
        n_expert,
        K_max,
        chunks_d,
        order_d,
        n_active_d,
        dp_prev_d,
        dp_cur_d,
        trace_d,
        (dp_moe_ext::StreamHandle) 0);
    check_cuda(cudaGetLastError(), "planner launch");
    check_cuda(cudaDeviceSynchronize(), "planner sync");

    std::vector<int> got_chunks((size_t) n_expert, -1);
    std::vector<int> got_order((size_t) n_expert, -1);
    int got_n_active = -1;
    check_cuda(cudaMemcpy(got_chunks.data(), chunks_d, chunks_bytes,
                          cudaMemcpyDeviceToHost), "copy chunks");
    check_cuda(cudaMemcpy(got_order.data(), order_d, chunks_bytes,
                          cudaMemcpyDeviceToHost), "copy order");
    check_cuda(cudaMemcpy(&got_n_active, n_active_d, sizeof(int),
                          cudaMemcpyDeviceToHost), "copy n_active");

    int failures = 0;
    if (got_n_active != (int) expected_order.size()) {
        std::fprintf(stderr, "%s: n_active got=%d expected=%zu\n",
                     name, got_n_active, expected_order.size());
        ++failures;
    }
    for (size_t i = 0; i < expected_order.size(); ++i) {
        if (got_order[i] != expected_order[i]) {
            std::fprintf(stderr, "%s: order[%zu] got=%d expected=%d\n",
                         name, i, got_order[i], expected_order[i]);
            ++failures;
        }
    }
    for (int e = 0; e < n_expert; ++e) {
        if (got_chunks[(size_t) e] != expected[(size_t) e]) {
            std::fprintf(stderr, "%s: chunks[%d] got=%d expected=%d\n",
                         name, e, got_chunks[(size_t) e],
                         expected[(size_t) e]);
            ++failures;
        }
    }

    cudaFree(ids_d);
    cudaFree(weights_d);
    cudaFree(R_d);
    cudaFree(chunks_d);
    cudaFree(order_d);
    cudaFree(n_active_d);
    cudaFree(dp_prev_d);
    cudaFree(dp_cur_d);
    cudaFree(trace_d);

    if (failures != 0) {
        std::fprintf(stderr, "%s: %d failure(s)\n", name, failures);
        std::exit(1);
    }
}

void run_bench_case(
    const char * name,
    int n_tokens,
    int n_used,
    int n_expert,
    int K_min,
    int K_max,
    float kbar,
    int warmup,
    int iters) {
    const int n_K = K_max - K_min + 1;
    const int total = n_tokens * n_used;

    std::vector<int32_t> ids((size_t) total);
    std::vector<float> weights((size_t) total);
    std::vector<float> R((size_t) n_expert * n_K, 0.0f);

    std::mt19937 rng(0xD17A110Cu);
    std::uniform_real_distribution<float> wdist(0.02f, 0.85f);
    for (int i = 0; i < total; ++i) {
        // Deterministically route across all experts for large batches.
        ids[(size_t) i] = (int32_t) ((i * 37 + (i / n_used) * 13 + 5) % n_expert);
        weights[(size_t) i] = wdist(rng);
    }
    for (int e = 0; e < n_expert; ++e) {
        for (int k = 0; k < n_K; ++k) {
            R[(size_t) e * n_K + k] =
                (float) (0.05 * (double) (n_K - k) +
                         0.001 * (double) ((e * 17 + k * 11) % 23));
        }
    }

    const std::vector<int> active =
        first_routed_order(ids, n_tokens, n_used, n_expert);
    const int B1 = n_expert * std::max(0, K_max - K_min) + 1;

    int32_t * ids_d = nullptr;
    float * weights_d = nullptr;
    float * R_d = nullptr;
    int * chunks_d = nullptr;
    int * order_d = nullptr;
    int * n_active_d = nullptr;
    float * dp_prev_d = nullptr;
    float * dp_cur_d = nullptr;
    uint8_t * trace_d = nullptr;
    int * chunks_h = nullptr;
    int * order_h = nullptr;
    int * n_active_h = nullptr;

    const size_t ids_bytes = ids.size() * sizeof(int32_t);
    const size_t weights_bytes = weights.size() * sizeof(float);
    const size_t R_bytes = R.size() * sizeof(float);
    const size_t chunks_bytes = (size_t) n_expert * sizeof(int);
    const size_t dp_bytes = (size_t) B1 * sizeof(float);
    const size_t trace_bytes = (size_t) n_expert * (size_t) B1 * sizeof(uint8_t);

    check_cuda(cudaMalloc(&ids_d, ids_bytes), "bench cudaMalloc ids");
    check_cuda(cudaMalloc(&weights_d, weights_bytes), "bench cudaMalloc weights");
    check_cuda(cudaMalloc(&R_d, R_bytes), "bench cudaMalloc R");
    check_cuda(cudaMalloc(&chunks_d, chunks_bytes), "bench cudaMalloc chunks");
    check_cuda(cudaMalloc(&order_d, chunks_bytes), "bench cudaMalloc order");
    check_cuda(cudaMalloc(&n_active_d, sizeof(int)), "bench cudaMalloc n_active");
    check_cuda(cudaMalloc(&dp_prev_d, dp_bytes), "bench cudaMalloc dp_prev");
    check_cuda(cudaMalloc(&dp_cur_d, dp_bytes), "bench cudaMalloc dp_cur");
    check_cuda(cudaMalloc(&trace_d, trace_bytes), "bench cudaMalloc trace");
    check_cuda(cudaMallocHost(&chunks_h, chunks_bytes), "bench cudaMallocHost chunks");
    check_cuda(cudaMallocHost(&order_h, chunks_bytes), "bench cudaMallocHost order");
    check_cuda(cudaMallocHost(&n_active_h, sizeof(int)), "bench cudaMallocHost n_active");

    cudaStream_t stream = nullptr;
    check_cuda(cudaStreamCreate(&stream), "bench stream create");
    check_cuda(cudaMemcpyAsync(ids_d, ids.data(), ids_bytes,
                               cudaMemcpyHostToDevice, stream),
               "bench copy ids");
    check_cuda(cudaMemcpyAsync(weights_d, weights.data(), weights_bytes,
                               cudaMemcpyHostToDevice, stream),
               "bench copy weights");
    check_cuda(cudaMemcpyAsync(R_d, R.data(), R_bytes,
                               cudaMemcpyHostToDevice, stream),
               "bench copy R");
    check_cuda(cudaStreamSynchronize(stream), "bench copy sync");

    auto launch_once = [&]() {
        dp_moe_ext::qwen3::launch_plan_chunks_kbar_exact_strided(
            ids_d,
            (size_t) n_used * sizeof(int32_t),
            weights_d,
            (size_t) n_used * sizeof(float),
            nullptr,
            0,
            R_d,
            K_min,
            K_max,
            0,
            kbar,
            n_tokens,
            n_used,
            n_expert,
            K_max,
            chunks_d,
            order_d,
            n_active_d,
            dp_prev_d,
            dp_cur_d,
            trace_d,
            (dp_moe_ext::StreamHandle) stream);
    };

    for (int i = 0; i < warmup; ++i) {
        launch_once();
    }
    check_cuda(cudaStreamSynchronize(stream), "bench warmup sync");

    cudaEvent_t ev0 = nullptr;
    cudaEvent_t ev1 = nullptr;
    check_cuda(cudaEventCreate(&ev0), "bench event create 0");
    check_cuda(cudaEventCreate(&ev1), "bench event create 1");
    check_cuda(cudaEventRecord(ev0, stream), "bench event record 0");
    for (int i = 0; i < iters; ++i) {
        launch_once();
    }
    check_cuda(cudaEventRecord(ev1, stream), "bench event record 1");
    check_cuda(cudaEventSynchronize(ev1), "bench event sync");
    float kernel_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&kernel_ms, ev0, ev1),
               "bench event elapsed");

    const auto rt0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        launch_once();
        check_cuda(cudaMemcpyAsync(chunks_h, chunks_d, chunks_bytes,
                                   cudaMemcpyDeviceToHost, stream),
                   "bench copy chunks d2h");
        check_cuda(cudaMemcpyAsync(order_h, order_d, chunks_bytes,
                                   cudaMemcpyDeviceToHost, stream),
                   "bench copy order d2h");
        check_cuda(cudaMemcpyAsync(n_active_h, n_active_d, sizeof(int),
                                   cudaMemcpyDeviceToHost, stream),
                   "bench copy n_active d2h");
        check_cuda(cudaStreamSynchronize(stream), "bench roundtrip sync");
    }
    const auto rt1 = std::chrono::steady_clock::now();

    const double kernel_us = (double) kernel_ms * 1000.0 / (double) iters;
    const double roundtrip_us =
        (double) std::chrono::duration_cast<std::chrono::nanoseconds>(
            rt1 - rt0).count() / 1000.0 / (double) iters;
    const double d2h_us = roundtrip_us - kernel_us;

    std::printf(
        "%-18s n_tokens=%d n_used=%d n_expert=%d active=%zu "
        "K=[%d,%d] kbar=%.2f B1=%d kernel_us=%.3f roundtrip_us=%.3f "
        "d2h_sync_us=%.3f\n",
        name, n_tokens, n_used, n_expert, active.size(),
        K_min, K_max, kbar, B1, kernel_us, roundtrip_us, d2h_us);

    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    cudaStreamDestroy(stream);
    cudaFree(ids_d);
    cudaFree(weights_d);
    cudaFree(R_d);
    cudaFree(chunks_d);
    cudaFree(order_d);
    cudaFree(n_active_d);
    cudaFree(dp_prev_d);
    cudaFree(dp_cur_d);
    cudaFree(trace_d);
    cudaFreeHost(chunks_h);
    cudaFreeHost(order_h);
    cudaFreeHost(n_active_h);
}

} // namespace

int main(int argc, char ** argv) {
    if (argc > 1 && std::string(argv[1]) == "--bench") {
        const int iters = argc > 2 ? std::atoi(argv[2]) : 4096;
        const int bench_n_tokens = argc > 3 ? std::atoi(argv[3]) : 128;
        const int warmup = 128;
        run_bench_case("dp_n128_k3",
                       bench_n_tokens, 8, 128, 2, 8, 3.0f, warmup, iters);
        run_bench_case("dp_n128_k4_5",
                       bench_n_tokens, 8, 128, 2, 8, 4.5f, warmup, iters);
        run_bench_case("dp_n128_k6",
                       bench_n_tokens, 8, 128, 2, 8, 6.0f, warmup, iters);
        return 0;
    }

    run_case("decode_repeated",
             {3, 7, 3, 1, 9, 7, 12, 3},
             {0.41f, 0.22f, 0.37f, 0.11f, 0.58f, 0.14f, 0.31f, 0.19f},
             1, 8, 16, 2, 8, 3.0f);

    run_case("batch_two_tokens",
             {2, 5, 8, 5, 1, 2, 7, 8},
             {0.25f, 0.63f, 0.18f, 0.44f, 0.52f, 0.16f, 0.35f, 0.29f},
             2, 4, 16, 2, 8, 4.5f);

    run_case("clamp_to_min",
             {4, 6, 4, 10},
             {0.90f, 0.70f, 0.20f, 0.10f},
             1, 4, 16, 3, 8, 1.0f);

    run_case("fixed_k_range",
             {0, 2, 4, 6},
             {0.12f, 0.34f, 0.56f, 0.78f},
             1, 4, 16, 4, 4, 4.0f);

    run_case("tie_break_first_routed",
             {11, 3, 8, 2, 11, 3},
             {0.30f, 0.30f, 0.30f, 0.30f, 0.30f, 0.30f},
             1, 6, 16, 2, 8, 3.0f);

    std::fprintf(stderr, "dp_moe-kbar-planner-test: OK\n");
    return 0;
}
