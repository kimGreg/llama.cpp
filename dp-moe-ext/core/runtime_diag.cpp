// DPMoE — cache-stack diagnostics implementation.
//
// Compiled into the static lib only when -DDP_MOE_DIAG=ON. The
// header's stubs provide the OFF path. See runtime_diag.h.

#include "runtime_diag.h"

#include "runtime.h"
#include "vram_pool.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace dp_moe_ext { namespace diag {

namespace {

// Cache-miss attribution buckets. Indexed by routing rank u (= ids[t,
// u]'s position in the top-K list) and chunk index p. Per-rank
// reveals which routed positions starve; per-chunk-idx reveals which
// chunk indices miss most (chunk 0 is reused 3× more than higher
// chunks, so its hit rate sets the cache floor).
//
// Lock-free atomic increments: a concurrent eviction may race a probe
// by ±1 per bucket; fine for aggregate signal. The buckets are
// process-wide, not per-Entry, because the LOAD walk classifies
// across all wids in one pass.
constexpr int kMaxRankBuckets  = 16;     // ≥ n_used_per_tok
constexpr int kMaxChunkIdxBuckets = kMaxChunksPerTensor + 1;

// Score-distribution histogram bucket count: 100 bins of width 0.01
// over [0, 1]. Both per-rank gate score and cum-before share these
// buckets; cum can exceed 1.0, in which case the value is clamped
// into the last bucket (over-bucketing tail).
constexpr int kScoreBins = 100;

struct Counters {
    std::atomic<uint64_t> rank_attempt[kMaxRankBuckets]    = {};
    std::atomic<uint64_t> rank_vram_hit[kMaxRankBuckets]   = {};
    std::atomic<uint64_t> rank_dram_hit[kMaxRankBuckets]   = {};
    std::atomic<uint64_t> rank_ssd_miss[kMaxRankBuckets]   = {};
    std::atomic<uint64_t> chunkidx_attempt[kMaxChunkIdxBuckets]  = {};
    std::atomic<uint64_t> chunkidx_vram_hit[kMaxChunkIdxBuckets] = {};
    std::atomic<uint64_t> chunkidx_dram_hit[kMaxChunkIdxBuckets] = {};
    std::atomic<uint64_t> chunkidx_ssd_miss[kMaxChunkIdxBuckets] = {};

    // Per-site move_chunk timing buckets. Indexed by MoveSite enum.
    std::atomic<uint64_t> move_ns   [(int)MoveSite ::_COUNT] = {};
    std::atomic<uint64_t> move_calls[(int)MoveSite ::_COUNT] = {};

    // Per-chunk classification counters (move_chunk events).
    std::atomic<uint64_t> move_evt[(int)MoveEvent::_COUNT] = {};

    // Score-distribution histograms, per-rank.
    //   gate_score_hist [r][b] : count of (rank=r, gate_score in bucket b)
    //   cum_before_hist [r][b] : count of (rank=r, cum_before in bucket b)
    std::atomic<uint64_t> gate_score_hist [kMaxRankBuckets][kScoreBins] = {};
    std::atomic<uint64_t> cum_before_hist [kMaxRankBuckets][kScoreBins] = {};
    // Sum of desired_chunks emitted per (rank), for avg-chunks-per-rank.
    std::atomic<uint64_t> desired_sum     [kMaxRankBuckets] = {};
    std::atomic<uint64_t> desired_count   [kMaxRankBuckets] = {};
};
Counters g_counters;

// Optional per-event CSV trace. Open at install via DP_MOE_GATE_TRACE_FILE.
// Format: layer,token_idx,rank,expert_id,gate_score,cum_before,desired_chunks.
// Lets an offline analyzer replay any rank_cum policy against the
// recorded score distribution.
FILE *      g_gate_trace_fp = nullptr;
std::mutex  g_gate_trace_mu;

const char * site_name(MoveSite s) {
    switch (s) {
        case MoveSite::Pread:             return "pread";
        case MoveSite::Xform:             return "xform";
        case MoveSite::HostCacheInsert:   return "host_cache_insert";
        case MoveSite::HostCacheSnapshot: return "host_cache_snapshot";
        case MoveSite::PoolLoadEnqueue:   return "pool_load_enq";
        case MoveSite::PtrUpdate:         return "ptr_update";
        case MoveSite::MakeRoom:          return "make_room";
        case MoveSite::Alloc:             return "alloc";
        default:                           return "?";
    }
}

const char * event_name(MoveEvent e) {
    switch (e) {
        case MoveEvent::SsdMiss:            return "ssd_miss";
        case MoveEvent::DramHit:            return "dram_hit";
        case MoveEvent::VramIdempotentHit:  return "vram_idempotent_hit";
        case MoveEvent::CacheInsert:        return "cache_insert";
        case MoveEvent::CacheSnapshot:      return "cache_snapshot";
        default:                             return "?";
    }
}

// Belady headroom: optional plain-text chunk-request trace. Open at
// install when DP_MOE_TRACE_FILE is set; closed at clear. Format
// is "<wid>\t<cid>\n" per request, in the order the LOAD walk emits
// them. The mutex serialises concurrent writes from prefetch workers
// + the hook thread; cheap because this path only runs when the user
// opted in to recording.
FILE *      g_trace_fp = nullptr;
std::mutex  g_trace_mu;

} // anon

void install_open_trace() {
    if (const char * tf = std::getenv("DP_MOE_TRACE_FILE")) {
        if (tf[0] && g_trace_fp == nullptr) {
            FILE * fp = std::fopen(tf, "w");
            if (fp) {
                std::setvbuf(fp, nullptr, _IOFBF, 1 << 20);
                g_trace_fp = fp;
                std::fprintf(stderr,
                    "DPMoE: chunk trace recording → %s\n", tf);
            } else {
                std::fprintf(stderr,
                    "DPMoE: failed to open trace file %s\n", tf);
            }
        }
    }
    if (const char * gtf = std::getenv("DP_MOE_GATE_TRACE_FILE")) {
        if (gtf[0] && g_gate_trace_fp == nullptr) {
            FILE * fp = std::fopen(gtf, "w");
            if (fp) {
                std::setvbuf(fp, nullptr, _IOFBF, 1 << 20);
                std::fprintf(fp,
                    "layer,token,rank,expert,gate_score,cum_before,desired\n");
                g_gate_trace_fp = fp;
                std::fprintf(stderr,
                    "DPMoE: gate-event trace recording → %s\n", gtf);
            } else {
                std::fprintf(stderr,
                    "DPMoE: failed to open gate trace file %s\n", gtf);
            }
        }
    }
}

void record_move_ns(MoveSite site, uint64_t ns) {
    const int i = (int)site;
    if (i < 0 || i >= (int)MoveSite::_COUNT) return;
    g_counters.move_ns   [i].fetch_add(ns, std::memory_order_relaxed);
    g_counters.move_calls[i].fetch_add(1,  std::memory_order_relaxed);
}

void record_move_event(MoveEvent ev) {
    const int i = (int)ev;
    if (i < 0 || i >= (int)MoveEvent::_COUNT) return;
    g_counters.move_evt[i].fetch_add(1, std::memory_order_relaxed);
}

void record_gate_event(int layer, int token_idx, int rank,
                       int expert_id, float gate_score,
                       float cum_before, int desired_chunks)
{
    const int rb = std::min(std::max(rank, 0), kMaxRankBuckets - 1);
    auto bucket = [](float v) {
        int b = (int)(v * (float)kScoreBins);
        if (b < 0) b = 0;
        if (b >= kScoreBins) b = kScoreBins - 1;
        return b;
    };
    g_counters.gate_score_hist[rb][bucket(gate_score)].fetch_add(
        1, std::memory_order_relaxed);
    g_counters.cum_before_hist[rb][bucket(cum_before)].fetch_add(
        1, std::memory_order_relaxed);
    g_counters.desired_sum[rb].fetch_add(
        (uint64_t)desired_chunks, std::memory_order_relaxed);
    g_counters.desired_count[rb].fetch_add(1, std::memory_order_relaxed);

    if (g_gate_trace_fp) {
        std::lock_guard<std::mutex> lk(g_gate_trace_mu);
        std::fprintf(g_gate_trace_fp,
            "%d,%d,%d,%d,%.6f,%.6f,%d\n",
            layer, token_idx, rank, expert_id,
            gate_score, cum_before, desired_chunks);
    }
}

void record_chunk(const DPMoERuntime & rt,
                  const std::string &      synthetic_wid,
                  int                      rank_u,
                  int                      chunk_idx,
                  int                      cid)
{
    const int rank_b  = std::min(rank_u,  kMaxRankBuckets  - 1);
    const int chunk_b = std::min(chunk_idx, kMaxChunkIdxBuckets - 1);
    g_counters.rank_attempt [rank_b ].fetch_add(1, std::memory_order_relaxed);
    g_counters.chunkidx_attempt[chunk_b].fetch_add(1, std::memory_order_relaxed);
    if (rt.pool().is_resident(synthetic_wid, cid)) {
        g_counters.rank_vram_hit [rank_b ].fetch_add(1, std::memory_order_relaxed);
        g_counters.chunkidx_vram_hit[chunk_b].fetch_add(1, std::memory_order_relaxed);
    } else if (rt.host_resident(synthetic_wid, cid)) {
        g_counters.rank_dram_hit [rank_b ].fetch_add(1, std::memory_order_relaxed);
        g_counters.chunkidx_dram_hit[chunk_b].fetch_add(1, std::memory_order_relaxed);
    } else {
        g_counters.rank_ssd_miss [rank_b ].fetch_add(1, std::memory_order_relaxed);
        g_counters.chunkidx_ssd_miss[chunk_b].fetch_add(1, std::memory_order_relaxed);
    }
    if (g_trace_fp) {
        std::lock_guard<std::mutex> lk(g_trace_mu);
        std::fprintf(g_trace_fp, "%s\t%d\n",
                     synthetic_wid.c_str(), cid);
    }
}

void clear_close_trace_and_dump() {
    // Close the trace files first so a partial flush from the inner
    // record_chunk / record_gate_event path can't race the dump.
    if (g_trace_fp) {
        std::lock_guard<std::mutex> lk(g_trace_mu);
        std::fclose(g_trace_fp);
        g_trace_fp = nullptr;
    }
    if (g_gate_trace_fp) {
        std::lock_guard<std::mutex> lk(g_gate_trace_mu);
        std::fclose(g_gate_trace_fp);
        g_gate_trace_fp = nullptr;
    }

    // The dump only fires when the user set DP_MOE_STATS or
    // DP_MOE_PROFILE. (The same env conventions used by the rest
    // of the runtime's diagnostic output.) Saves spamming stderr on
    // ordinary runs even when DIAG was compiled in.
    const bool want_stats   = std::getenv("DP_MOE_STATS")   != nullptr;
    const bool want_profile = std::getenv("DP_MOE_PROFILE") != nullptr;
    if (!want_stats && !want_profile) return;

    auto & m = g_counters;
    if (want_stats) {
        // Aggregate tier-hit summary. Sum the per-rank buckets — gives
        // a single cache-stack health number.
        uint64_t tot_att = 0, tot_vram = 0, tot_dram = 0, tot_ssd = 0;
        for (int r = 0; r < kMaxRankBuckets; ++r) {
            tot_att  += m.rank_attempt[r].load();
            tot_vram += m.rank_vram_hit[r].load();
            tot_dram += m.rank_dram_hit[r].load();
            tot_ssd  += m.rank_ssd_miss[r].load();
        }
        if (tot_att > 0) {
            const double pct = 100.0 / (double)tot_att;
            std::fprintf(stderr,
                "DPMoE tier hits: attempts=%lu  VRAM=%.1f%% (%lu)  "
                "DRAM=%.1f%% (%lu)  SSD=%.1f%% (%lu)\n",
                (unsigned long)tot_att,
                tot_vram * pct, (unsigned long)tot_vram,
                tot_dram * pct, (unsigned long)tot_dram,
                tot_ssd  * pct, (unsigned long)tot_ssd);
            // Expert-chunk allocation summary. Surfaces the headline
            // numbers the policy operator cares about: how many chunks
            // got requested per token, how many actually had to be
            // SSD-streamed, and the avg chunks per (rank, chunk-idx)
            // bucket. Per-expert distribution requires the trace file
            // (DP_MOE_TRACE_FILE) + experiments/analyze_expert_chunks.py.
            uint64_t avg_chunkidx_weighted = 0;
            uint64_t att_with_idx = 0;
            for (int pp = 0; pp < kMaxChunkIdxBuckets; ++pp) {
                const uint64_t a = m.chunkidx_attempt[pp].load();
                avg_chunkidx_weighted += a * (uint64_t)pp;
                att_with_idx += a;
            }
            const double avg_chunk_idx = att_with_idx
                ? (double)avg_chunkidx_weighted / (double)att_with_idx : 0.0;
            std::fprintf(stderr,
                "DPMoE chunk pattern: requests=%lu  ssd_loaded=%lu "
                "(%.1f%%)  avg_chunk_idx=%.2f  active_chunks/req=%.2f\n",
                (unsigned long)tot_att,
                (unsigned long)tot_ssd,
                tot_ssd * pct,
                avg_chunk_idx,
                tot_att > 0 ? (double)att_with_idx / (double)tot_att : 0.0);
        }
    }

    if (!want_profile) return;

    // Per-rank breakdown — where in the routing top-K do misses
    // concentrate? Useful for tuning rank_cum thresholds.
    std::fprintf(stderr,
        "  per-rank misses (rank u = ids[t,u] position):\n"
        "    rank | attempts |  VRAM%% |  DRAM%% |   SSD%% |  ssd_count\n");
    for (int r = 0; r < kMaxRankBuckets; ++r) {
        const uint64_t a = m.rank_attempt[r].load();
        if (a == 0) continue;
        const uint64_t vh = m.rank_vram_hit[r].load();
        const uint64_t dh = m.rank_dram_hit[r].load();
        const uint64_t sm = m.rank_ssd_miss[r].load();
        const double pa = 100.0 / (double)a;
        std::fprintf(stderr,
            "      %2d | %8lu | %5.1f%% | %5.1f%% | %5.1f%% | %10lu\n",
            r, (unsigned long)a, vh*pa, dh*pa, sm*pa,
            (unsigned long)sm);
    }
    // Per-chunk-idx breakdown — chunk 0 is the universal base (every
    // routed expert uses it), so its miss rate sets the cache floor;
    // higher chunks can afford more eviction.
    std::fprintf(stderr,
        "  per-chunk-idx misses (chunk 0 = base, kMax-1 = top refinement):\n"
        "    chunk | attempts |  VRAM%% |  DRAM%% |   SSD%% |  ssd_count\n");
    for (int pp = 0; pp < kMaxChunkIdxBuckets; ++pp) {
        const uint64_t a = m.chunkidx_attempt[pp].load();
        if (a == 0) continue;
        const uint64_t vh = m.chunkidx_vram_hit[pp].load();
        const uint64_t dh = m.chunkidx_dram_hit[pp].load();
        const uint64_t sm = m.chunkidx_ssd_miss[pp].load();
        const double pa = 100.0 / (double)a;
        std::fprintf(stderr,
            "      %3d | %8lu | %5.1f%% | %5.1f%% | %5.1f%% | %10lu\n",
            pp, (unsigned long)a, vh*pa, dh*pa, sm*pa,
            (unsigned long)sm);
    }

    // Per-call-site move_chunk overhead breakdown. Splits the
    // software-overhead term in the refined roofline equation
    // (T_observed - T_bw_lower_bound) into its named summands so we
    // can see which fix to do first.
    uint64_t total_move_ns = 0;
    for (int i = 0; i < (int)MoveSite::_COUNT; ++i) {
        total_move_ns += m.move_ns[i].load();
    }
    if (total_move_ns > 0) {
        std::fprintf(stderr,
            "  move_chunk overhead (total = %.3f s):\n"
            "    site                  |     calls |  total_ms |  avg_us | %% of total\n",
            total_move_ns / 1e9);
        for (int i = 0; i < (int)MoveSite::_COUNT; ++i) {
            const uint64_t ns    = m.move_ns   [i].load();
            const uint64_t calls = m.move_calls[i].load();
            const double total_ms = ns / 1e6;
            const double avg_us   = calls
                ? ((double)ns / (double)calls / 1e3) : 0.0;
            const double pct      = 100.0 * (double)ns / (double)total_move_ns;
            std::fprintf(stderr,
                "    %-21s | %9lu | %9.2f | %7.2f | %5.1f%%\n",
                site_name((MoveSite)i),
                (unsigned long)calls, total_ms, avg_us, pct);
        }
    }
    bool any_evt = false;
    for (int i = 0; i < (int)MoveEvent::_COUNT; ++i) {
        if (m.move_evt[i].load() > 0) { any_evt = true; break; }
    }
    if (any_evt) {
        std::fprintf(stderr,
            "  move_chunk events:\n");
        for (int i = 0; i < (int)MoveEvent::_COUNT; ++i) {
            const uint64_t v = m.move_evt[i].load();
            if (v == 0) continue;
            std::fprintf(stderr,
                "    %-21s = %lu\n",
                event_name((MoveEvent)i), (unsigned long)v);
        }
    }

    // Score-distribution summary. Per-rank avg gate / cum_before /
    // desired-chunks. The histograms themselves are written to the
    // gate trace CSV (when DP_MOE_GATE_TRACE_FILE is set) for
    // offline replay; here we emit a 1-line summary per rank that
    // shows the policy's effective shape.
    bool any_gate = false;
    for (int r = 0; r < kMaxRankBuckets; ++r) {
        if (m.desired_count[r].load() > 0) { any_gate = true; break; }
    }
    if (any_gate) {
        std::fprintf(stderr,
            "  gate-event score distribution (per rank u):\n"
            "    rank | events |  mean_gate |  mean_cum |  mean_desired\n");
        for (int r = 0; r < kMaxRankBuckets; ++r) {
            const uint64_t n = m.desired_count[r].load();
            if (n == 0) continue;
            // Recover means from per-bucket histograms (bucket center
            // = (b + 0.5) / kScoreBins).
            double sum_g = 0.0, sum_c = 0.0;
            uint64_t n_g = 0, n_c = 0;
            for (int b = 0; b < kScoreBins; ++b) {
                const uint64_t gh = m.gate_score_hist[r][b].load();
                const uint64_t ch = m.cum_before_hist[r][b].load();
                const double center = ((double)b + 0.5) / (double)kScoreBins;
                sum_g += (double)gh * center;
                sum_c += (double)ch * center;
                n_g += gh;
                n_c += ch;
            }
            const double mean_gate = n_g ? sum_g / (double)n_g : 0.0;
            const double mean_cum  = n_c ? sum_c / (double)n_c : 0.0;
            const double mean_des  =
                (double)m.desired_sum[r].load() / (double)n;
            std::fprintf(stderr,
                "    %4d | %6lu |   %7.3f |  %7.3f |       %5.2f\n",
                r, (unsigned long)n, mean_gate, mean_cum, mean_des);
        }
    }
}

}} // namespace dp_moe_ext::diag

// Aggregated tier-hit accessors (sum across all rank buckets). Single
// atomic-load per bucket, no contention. Used by /dp_moe/stats so
// the panel can show a live cache-hit-rate readout.
extern "C" {
unsigned long long dp_moe_stat_tier_attempts(void) {
    using namespace dp_moe_ext::diag;
    uint64_t t = 0;
    for (int r = 0; r < kMaxRankBuckets; ++r)
        t += g_counters.rank_attempt[r].load(std::memory_order_relaxed);
    return (unsigned long long) t;
}
unsigned long long dp_moe_stat_tier_vram_hits(void) {
    using namespace dp_moe_ext::diag;
    uint64_t t = 0;
    for (int r = 0; r < kMaxRankBuckets; ++r)
        t += g_counters.rank_vram_hit[r].load(std::memory_order_relaxed);
    return (unsigned long long) t;
}
unsigned long long dp_moe_stat_tier_dram_hits(void) {
    using namespace dp_moe_ext::diag;
    uint64_t t = 0;
    for (int r = 0; r < kMaxRankBuckets; ++r)
        t += g_counters.rank_dram_hit[r].load(std::memory_order_relaxed);
    return (unsigned long long) t;
}
unsigned long long dp_moe_stat_tier_ssd_misses(void) {
    using namespace dp_moe_ext::diag;
    uint64_t t = 0;
    for (int r = 0; r < kMaxRankBuckets; ++r)
        t += g_counters.rank_ssd_miss[r].load(std::memory_order_relaxed);
    return (unsigned long long) t;
}
int dp_moe_stat_diag_enabled(void) { return 1; }
} // extern "C"
