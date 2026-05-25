// DPMoE — launch / sync breakdown counters.
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
// when DP_MOE_STATS is set.
//
// Cost: each bump is one relaxed fetch_add — a few ns.  Cheap enough
// to leave on by default.

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <chrono>

namespace dp_moe_ext { namespace launch_diag {

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

enum class Phase : int {
    PreInputsD2H = 0,
    Plan,
    Reserve,
    Load,
    Validate,
    Execute,
    Release,
    Pread,
    PoolLoad,
    PointerPatch,
    DirectPrepare,
    DirectIdsD2H,
    DirectLoad,
    DirectD2D,
    _COUNT
};

inline const char * phase_name(Phase p) {
    switch (p) {
        case Phase::PreInputsD2H:  return "pre_inputs_d2h";
        case Phase::Plan:          return "plan";
        case Phase::Reserve:       return "reserve";
        case Phase::Load:          return "load";
        case Phase::Validate:      return "validate";
        case Phase::Execute:       return "execute";
        case Phase::Release:       return "release";
        case Phase::Pread:         return "pread";
        case Phase::PoolLoad:      return "pool_load";
        case Phase::PointerPatch:  return "pointer_patch";
        case Phase::DirectPrepare: return "direct_prepare";
        case Phase::DirectIdsD2H:  return "direct_ids_d2h";
        case Phase::DirectLoad:    return "direct_load";
        case Phase::DirectD2D:     return "direct_d2d";
        default:                   return "?";
    }
}

// Index 0 = prefill / multi-token, index 1 = decode / single-token.
inline std::atomic<uint64_t> g_phase_ns[2][(int)Phase::_COUNT] = {};
inline std::atomic<uint64_t> g_phase_calls[2][(int)Phase::_COUNT] = {};
inline std::atomic<uint64_t> g_pread_logical_bytes[2] = {};
inline std::atomic<uint64_t> g_pread_physical_bytes[2] = {};
inline std::atomic<uint64_t> g_pread_spans[2] = {};
inline std::atomic<uint64_t> g_required_by_chunk[2][8] = {};
inline std::atomic<uint64_t> g_resident_by_chunk[2][8] = {};
inline std::atomic<uint64_t> g_unique_load_by_chunk[2][8] = {};
inline std::atomic<uint64_t> g_evicted_by_chunk[8] = {};
inline thread_local bool g_current_decode_phase = false;

inline void set_current_decode_phase(bool decode) {
    g_current_decode_phase = decode;
}

inline bool current_decode_phase() {
    return g_current_decode_phase;
}

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

inline void note_phase(Phase p, bool decode, uint64_t ns) {
    const int i = (int) p;
    if (i < 0 || i >= (int) Phase::_COUNT) return;
    const int ph = decode ? 1 : 0;
    g_phase_ns[ph][i].fetch_add(ns, std::memory_order_relaxed);
    g_phase_calls[ph][i].fetch_add(1, std::memory_order_relaxed);
}

inline void note_pread_io(bool decode, size_t logical_bytes,
                          size_t physical_bytes, size_t spans) {
    const int ph = decode ? 1 : 0;
    g_pread_logical_bytes[ph].fetch_add(
        (uint64_t) logical_bytes, std::memory_order_relaxed);
    g_pread_physical_bytes[ph].fetch_add(
        (uint64_t) physical_bytes, std::memory_order_relaxed);
    g_pread_spans[ph].fetch_add((uint64_t) spans,
                                std::memory_order_relaxed);
}

inline int chunk_bucket_from_cid(int cid) {
    constexpr int kCidChunkBaseLocal = 100;
    const int p = cid - kCidChunkBaseLocal;
    return (p >= 0 && p < 8) ? p : -1;
}

inline void note_required_chunk(bool decode, int cid, bool resident) {
    const int p = chunk_bucket_from_cid(cid);
    if (p < 0) return;
    const int ph = decode ? 1 : 0;
    g_required_by_chunk[ph][p].fetch_add(1, std::memory_order_relaxed);
    if (resident) {
        g_resident_by_chunk[ph][p].fetch_add(1, std::memory_order_relaxed);
    }
}

inline void note_unique_load_chunk(bool decode, int cid) {
    const int p = chunk_bucket_from_cid(cid);
    if (p < 0) return;
    const int ph = decode ? 1 : 0;
    g_unique_load_by_chunk[ph][p].fetch_add(1, std::memory_order_relaxed);
}

inline void note_evicted_chunk(int cid) {
    const int p = chunk_bucket_from_cid(cid);
    if (p < 0) return;
    g_evicted_by_chunk[p].fetch_add(1, std::memory_order_relaxed);
}

class PhaseTimer {
public:
    PhaseTimer(Phase p, bool decode)
        : p_(p), decode_(decode), t0_(std::chrono::steady_clock::now()) {}
    ~PhaseTimer() {
        const auto dt = std::chrono::steady_clock::now() - t0_;
        note_phase(p_, decode_, (uint64_t) std::chrono::duration_cast<
            std::chrono::nanoseconds>(dt).count());
    }
private:
    Phase p_;
    bool decode_;
    std::chrono::steady_clock::time_point t0_;
};

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
    for (int ph = 0; ph < 2; ++ph) {
        for (int i = 0; i < (int) Phase::_COUNT; ++i) {
            g_phase_ns[ph][i].store(0, std::memory_order_relaxed);
            g_phase_calls[ph][i].store(0, std::memory_order_relaxed);
        }
        g_pread_logical_bytes[ph].store(0, std::memory_order_relaxed);
        g_pread_physical_bytes[ph].store(0, std::memory_order_relaxed);
        g_pread_spans[ph].store(0, std::memory_order_relaxed);
        for (int p = 0; p < 8; ++p) {
            g_required_by_chunk[ph][p].store(0, std::memory_order_relaxed);
            g_resident_by_chunk[ph][p].store(0, std::memory_order_relaxed);
            g_unique_load_by_chunk[ph][p].store(0, std::memory_order_relaxed);
        }
    }
    for (int p = 0; p < 8; ++p) {
        g_evicted_by_chunk[p].store(0, std::memory_order_relaxed);
    }
}

inline void dump_to_stderr() {
    const uint64_t fwd_calls  = g_forward_moe_layer_calls.load(std::memory_order_relaxed);
    const uint64_t fwd_ns     = g_forward_moe_layer_ns.load(std::memory_order_relaxed);
    const uint64_t disp_calls = g_dispatch_one_canonical_calls.load(std::memory_order_relaxed);
    const uint64_t disp_ns    = g_dispatch_one_canonical_ns.load(std::memory_order_relaxed);
    const uint64_t lss_sum    = g_load_set_size_sum.load(std::memory_order_relaxed);
    const uint64_t lss_disp   = g_load_set_dispatch_count.load(std::memory_order_relaxed);

    std::fprintf(stderr,
        "DPMoE launch breakdown:\n"
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

    std::fprintf(stderr, "  phase timings by phase:\n");
    for (int ph = 0; ph < 2; ++ph) {
        const char * ph_name = ph ? "decode" : "prefill";
        uint64_t total_ns = 0;
        for (int i = 0; i < (int) Phase::_COUNT; ++i) {
            total_ns += g_phase_ns[ph][i].load(std::memory_order_relaxed);
        }
        if (total_ns == 0) continue;
        std::fprintf(stderr, "    [%s]\n", ph_name);
        for (int i = 0; i < (int) Phase::_COUNT; ++i) {
            const uint64_t ns = g_phase_ns[ph][i].load(std::memory_order_relaxed);
            const uint64_t calls = g_phase_calls[ph][i].load(std::memory_order_relaxed);
            if (ns == 0 && calls == 0) continue;
            std::fprintf(stderr,
                "      %-16s calls=%8lu total=%10.3f ms avg=%9.2f us\n",
                phase_name((Phase) i),
                (unsigned long) calls,
                ns / 1.0e6,
                calls ? (double) ns / (double) calls / 1.0e3 : 0.0);
        }
        const uint64_t logical =
            g_pread_logical_bytes[ph].load(std::memory_order_relaxed);
        const uint64_t physical =
            g_pread_physical_bytes[ph].load(std::memory_order_relaxed);
        const uint64_t spans =
            g_pread_spans[ph].load(std::memory_order_relaxed);
        if (logical || physical || spans) {
            std::fprintf(stderr,
                "      pread_io         logical=%8.1f MB physical=%8.1f MB "
                "gap=%8.1f MB spans=%8lu avg_span=%8.1f KB\n",
                logical / 1048576.0,
                physical / 1048576.0,
                physical >= logical ? (physical - logical) / 1048576.0 : 0.0,
                (unsigned long) spans,
                spans ? (double) physical / (double) spans / 1024.0 : 0.0);
        }
        bool any_chunk = false;
        for (int p = 0; p < 8; ++p) {
            if (g_required_by_chunk[ph][p].load(std::memory_order_relaxed) ||
                g_unique_load_by_chunk[ph][p].load(std::memory_order_relaxed)) {
                any_chunk = true;
                break;
            }
        }
        if (any_chunk) {
            std::fprintf(stderr,
                "      chunk residency by chunk index:\n"
                "        c | required | resident | hit%% | unique_loads\n");
            for (int p = 0; p < 8; ++p) {
                const uint64_t req =
                    g_required_by_chunk[ph][p].load(std::memory_order_relaxed);
                const uint64_t hit =
                    g_resident_by_chunk[ph][p].load(std::memory_order_relaxed);
                const uint64_t loads =
                    g_unique_load_by_chunk[ph][p].load(std::memory_order_relaxed);
                if (!req && !loads) continue;
                std::fprintf(stderr,
                    "        %d | %8lu | %8lu | %5.1f | %12lu\n",
                    p, (unsigned long) req, (unsigned long) hit,
                    req ? 100.0 * (double) hit / (double) req : 0.0,
                    (unsigned long) loads);
            }
        }
    }
    bool any_evict = false;
    for (int p = 0; p < 8; ++p) {
        if (g_evicted_by_chunk[p].load(std::memory_order_relaxed)) {
            any_evict = true;
            break;
        }
    }
    if (any_evict) {
        std::fprintf(stderr,
            "  evictions by chunk index:\n"
            "    c | evictions\n");
        for (int p = 0; p < 8; ++p) {
            const uint64_t ev =
                g_evicted_by_chunk[p].load(std::memory_order_relaxed);
            if (!ev) continue;
            std::fprintf(stderr, "    %d | %9lu\n",
                         p, (unsigned long) ev);
        }
    }
}

}}  // namespace dp_moe_ext::launch_diag
