// streamllm-ext — launch / sync breakdown counters.
//
// Always-on atomic counters for kernel launches, host syncs, and CPU
// time spent in the sentinel eager MoE path.  Header-only: one inline
// translation unit per .cpp / .cu that includes it bumps the globals
// declared here via the ``inline`` keyword (C++17 inline variables).
//
// These exist to answer "where does each token's decode time go" at the
// granularity the TPS bottleneck investigation needs: kernel launch
// count by kind, cudaStreamSynchronize / cudaStreamWaitEvent counts,
// and accumulated CPU time per dispatch.  All counters are dumped via
// ``launch_diag::dump_to_stderr`` from ``qwen3_runtime_glue::clear``
// when STREAMLLM_STATS is set.
//
// Cost: each bump is one relaxed fetch_add — a few ns.  Cheap enough
// to leave on by default.

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace streamllm_ext { namespace launch_diag {

// Kernel-launch kinds we instrument.  Each is bumped once per kernel
// launch on the compute stream.
enum class Kind : int {
    MoeMatmul = 0,         // anybcq::naver_gemv_moe_launch  → nqmv_bias_planes_moe_fused
    AlphaBetaRefresh,      // qwen3::refresh_alpha_for_anyprec_launch
    QbiasRefresh,          // qwen3::refresh_q_bias_for_anyprec_launch
    SwigluMul,             // qwen3::launch_swiglu_mul
    WeightedReduce,        // qwen3::launch_weighted_reduce_slots
    F32ToF16,              // anybcq::launch_f32_to_f16
    Memcpy2DAsync,         // cudaMemcpy2DAsync issued from the dispatch path
    MemcpyAsync,           // cudaMemcpyAsync issued from the dispatch path
    MemsetAsync,           // cudaMemsetAsync issued from the dispatch path
    _COUNT
};

inline const char * kind_name(Kind k) {
    switch (k) {
        case Kind::MoeMatmul:         return "moe_matmul";
        case Kind::AlphaBetaRefresh:  return "alpha_beta_refresh";
        case Kind::QbiasRefresh:      return "qbias_refresh";
        case Kind::SwigluMul:         return "swiglu_mul";
        case Kind::WeightedReduce:    return "weighted_reduce";
        case Kind::F32ToF16:          return "f32_to_f16";
        case Kind::Memcpy2DAsync:     return "memcpy_2d_async";
        case Kind::MemcpyAsync:       return "memcpy_async";
        case Kind::MemsetAsync:       return "memset_async";
        default:                       return "?";
    }
}

// C++17 inline variables — one definition shared across all TUs that
// include this header.  Default-constructed atomics start at zero.
inline std::atomic<uint64_t> g_launch_count[(int)Kind::_COUNT] = {};

// Host-side sync points.
inline std::atomic<uint64_t> g_stream_sync_count{0};           // cudaStreamSynchronize
inline std::atomic<uint64_t> g_stream_wait_event_count{0};     // cudaStreamWaitEvent submitted
inline std::atomic<uint64_t> g_async_load_wait_count{0};       // wait_async_load_batch calls

// Per-call CPU ns + call counts for the two hot frames.  forward_moe_layer
// is per-(token × layer), dispatch_one_canonical is per-(forward × canonical).
inline std::atomic<uint64_t> g_forward_moe_layer_ns{0};
inline std::atomic<uint64_t> g_forward_moe_layer_calls{0};
inline std::atomic<uint64_t> g_dispatch_one_canonical_ns{0};
inline std::atomic<uint64_t> g_dispatch_one_canonical_calls{0};

// Plan-derived counters.  load_set_size summed across every dispatch;
// useful as a sanity check against the residency counters in vram_pool.
inline std::atomic<uint64_t> g_load_set_size_sum{0};
inline std::atomic<uint64_t> g_load_set_dispatch_count{0};

inline void note_launch(Kind k) {
    const int i = (int) k;
    if (i < 0 || i >= (int) Kind::_COUNT) return;
    g_launch_count[i].fetch_add(1, std::memory_order_relaxed);
}

inline void note_stream_sync()      { g_stream_sync_count.fetch_add(1, std::memory_order_relaxed); }
inline void note_stream_wait_event(){ g_stream_wait_event_count.fetch_add(1, std::memory_order_relaxed); }
inline void note_async_load_wait()  { g_async_load_wait_count.fetch_add(1, std::memory_order_relaxed); }

inline void note_forward_moe_layer(uint64_t ns) {
    g_forward_moe_layer_ns.fetch_add(ns,    std::memory_order_relaxed);
    g_forward_moe_layer_calls.fetch_add(1,   std::memory_order_relaxed);
}
inline void note_dispatch_one_canonical(uint64_t ns) {
    g_dispatch_one_canonical_ns.fetch_add(ns,    std::memory_order_relaxed);
    g_dispatch_one_canonical_calls.fetch_add(1,   std::memory_order_relaxed);
}
inline void note_load_set(size_t n) {
    g_load_set_size_sum.fetch_add((uint64_t) n, std::memory_order_relaxed);
    g_load_set_dispatch_count.fetch_add(1,       std::memory_order_relaxed);
}

// Reset every counter to zero.  Useful between bench phases.
inline void reset() {
    for (int i = 0; i < (int) Kind::_COUNT; ++i) {
        g_launch_count[i].store(0, std::memory_order_relaxed);
    }
    g_stream_sync_count.store(0,            std::memory_order_relaxed);
    g_stream_wait_event_count.store(0,      std::memory_order_relaxed);
    g_async_load_wait_count.store(0,        std::memory_order_relaxed);
    g_forward_moe_layer_ns.store(0,         std::memory_order_relaxed);
    g_forward_moe_layer_calls.store(0,      std::memory_order_relaxed);
    g_dispatch_one_canonical_ns.store(0,    std::memory_order_relaxed);
    g_dispatch_one_canonical_calls.store(0, std::memory_order_relaxed);
    g_load_set_size_sum.store(0,            std::memory_order_relaxed);
    g_load_set_dispatch_count.store(0,      std::memory_order_relaxed);
}

inline void dump_to_stderr() {
    const uint64_t fwd_calls  = g_forward_moe_layer_calls.load(std::memory_order_relaxed);
    const uint64_t fwd_ns     = g_forward_moe_layer_ns.load(std::memory_order_relaxed);
    const uint64_t disp_calls = g_dispatch_one_canonical_calls.load(std::memory_order_relaxed);
    const uint64_t disp_ns    = g_dispatch_one_canonical_ns.load(std::memory_order_relaxed);
    const uint64_t lss_sum    = g_load_set_size_sum.load(std::memory_order_relaxed);
    const uint64_t lss_disp   = g_load_set_dispatch_count.load(std::memory_order_relaxed);

    std::fprintf(stderr,
        "streamllm-ext launch breakdown:\n"
        "  forward_moe_layer:       calls=%lu  cpu_total=%.3f ms  cpu_avg=%.2f us/call\n"
        "  dispatch_one_canonical:  calls=%lu  cpu_total=%.3f ms  cpu_avg=%.2f us/call\n"
        "  stream_sync_count:       %lu\n"
        "  stream_wait_event_count: %lu\n"
        "  async_load_wait_count:   %lu\n"
        "  load_set_size_sum:       %lu over %lu dispatches  (avg %.2f/dispatch)\n",
        (unsigned long) fwd_calls,
        fwd_ns / 1.0e6,
        fwd_calls ? (double)fwd_ns / (double)fwd_calls / 1.0e3 : 0.0,
        (unsigned long) disp_calls,
        disp_ns / 1.0e6,
        disp_calls ? (double)disp_ns / (double)disp_calls / 1.0e3 : 0.0,
        (unsigned long) g_stream_sync_count.load(std::memory_order_relaxed),
        (unsigned long) g_stream_wait_event_count.load(std::memory_order_relaxed),
        (unsigned long) g_async_load_wait_count.load(std::memory_order_relaxed),
        (unsigned long) lss_sum, (unsigned long) lss_disp,
        lss_disp ? (double) lss_sum / (double) lss_disp : 0.0);

    uint64_t total_launches = 0;
    for (int i = 0; i < (int) Kind::_COUNT; ++i) {
        total_launches += g_launch_count[i].load(std::memory_order_relaxed);
    }
    std::fprintf(stderr,
        "  kernel/copy launches by kind  (total=%lu):\n",
        (unsigned long) total_launches);
    for (int i = 0; i < (int) Kind::_COUNT; ++i) {
        const uint64_t v = g_launch_count[i].load(std::memory_order_relaxed);
        if (v == 0) continue;
        const double per_fwd = fwd_calls ? (double) v / (double) fwd_calls : 0.0;
        std::fprintf(stderr,
            "    %-22s = %10lu  (%.2f / forward_moe_layer)\n",
            kind_name((Kind) i), (unsigned long) v, per_fwd);
    }
}

}}  // namespace streamllm_ext::launch_diag
