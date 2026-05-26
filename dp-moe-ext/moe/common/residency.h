// DPMoE — MoE residency tracker.
//
// Per-(wid, cid) bookkeeping for the on-demand chunks the MoE
// scheduler loads above the pinned MIN floor. Used by
// `MoEScheduler::make_room_for` when the VramChunkPool runs out of
// arena space — the tracker picks the lowest-priority resident
// chunk to evict.
//
// One LRU list per AnyBCQ plane index (front = oldest within plane).
// `touch` moves an entry to the back of its plane's list and stamps
// a monotonically-increasing global counter so the eviction policy
// can compare staleness across planes.
//
// Default eviction policy is a weighted score:
//
//     score = plane_w · plane_idx
//           + age_w   · touches_since_last_seen          ← LRU
//           + freq_w  / (touch_count + 1)                ← LFU
//
// Default weights:
//   plane_w = 8  — keep the monotonic-by-plane invariant. Within
//                  one expert, higher-index planes always evict
//                  before lower-index ones.
//   age_w   = 1  — break ties by recency.
//   freq_w  = 0  — disabled by default (no-op LFU). Set via
//                  DP_MOE_MOE_EVICT_FREQ_WEIGHT to enable.
//
// Alternative diagnostic policy:
//
//     DP_MOE_MOE_EVICT_AGE_PLANE_TIE_WINDOW >= 0
//
//     Pick the oldest candidate first; if candidates are within the
//     configured age window, evict the higher plane first.  A window
//     of 0 is literal "pure age with high-plane exact tie-break".
//
// Reservations: the chunks the *current* dispatch needs are kept in
// `reserved_` for the load+compute window; `make_room` skips them.
//
// Thread-safe: `touch`, `reserve`, `release`, `make_room` are
// callable from any thread (hook thread runs touch/reserve/release;
// prefetch workers call make_room on pool-load failure).

#pragma once

#include "runtime.h"
#include "vram_pool.h"
#include "launch_diag.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <list>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace dp_moe_ext {

class MoEResidencyTracker {
public:
    MoEResidencyTracker(double plane_w, double age_w, double freq_w,
                         int64_t age_plane_tie_window)
        : plane_weight_(plane_w),
          age_weight_(age_w),
          freq_weight_(freq_w),
          age_plane_tie_window_(age_plane_tie_window) {}

    void touch(const std::string & wid, int cid, int plane) {
        if (plane < 0 || plane >= kMaxChunksPerTensor) return;
        std::lock_guard<std::mutex> lk(mu_);
        const uint64_t ts = ++touch_ts_;
        touch_locked_(wid, cid, plane, ts);
    }

    void touch_at(const std::string & wid, int cid, int plane, uint64_t ts) {
        if (plane < 0 || plane >= kMaxChunksPerTensor) return;
        std::lock_guard<std::mutex> lk(mu_);
        if (ts > touch_ts_) touch_ts_ = ts;
        touch_locked_(wid, cid, plane, ts);
    }

private:
    void touch_locked_(const std::string & wid, int cid, int plane, uint64_t ts) {
        ChunkKey k{wid, cid};
        auto it = index_.find(k);
        if (it != index_.end()) {
            buckets_[it->second.plane].erase(it->second.it);
            buckets_[plane].push_back(k);
            it->second.plane         = plane;
            it->second.it            = std::prev(buckets_[plane].end());
            it->second.last_touch_ts = ts;
            it->second.touch_count  += 1;
            return;
        }
        buckets_[plane].push_back(k);
        index_.emplace(std::move(k),
                        Entry{plane, std::prev(buckets_[plane].end()),
                              ts, /*touch_count=*/1});
    }

public:
    void reserve(const std::string & wid, int cid) {
        std::lock_guard<std::mutex> lk(mu_);
        reserved_.insert(ChunkKey{wid, cid});
    }
    void release(const std::string & wid, int cid) {
        std::lock_guard<std::mutex> lk(mu_);
        reserved_.erase(ChunkKey{wid, cid});
    }

    // Combined-score eviction. For each plane bucket find the most
    // evictable entry (front of LRU list, skipping reserved); compute
    // its score; pop the global maximum.
    //
    // Two reservation sources are honoured: this tracker's local
    // ``reserved_`` set (legacy hook-window dispatch reservation) AND
    // the scheduler's replay-scoped ``is_replay_reserved`` set
    // (P2★: moved off the runtime onto the scheduler).  Replay
    // reservations protect the in-flight dispatch's freshly-loaded
    // chunks across the rest of the captured replay, so a later
    // layer's planner can't evict chunks the earlier layer's kernel
    // is still about to read.
    //
    // Eviction MUST clear the entry's per-plane device-pointer slot
    // (dev.chunk_qw_ptrs[p] / chunk_alpha_ptrs[p]) so the kernel
    // doesn't read stale memory after the freed slot gets reused by
    // another chunk's load. The runtime exposes this via
    // ``rt->clear_chunk_device_ptr(wid, cid)`` — without it, kernel
    // outputs go subtly wrong under heavy eviction (the model still
    // generates tokens but at degraded / corrupted quality).
    bool make_room(VramChunkPool & pool,
                    DPMoERuntime & rt,
                    const Scheduler & sched) {
        std::lock_guard<std::mutex> lk(mu_);
        const uint64_t now = touch_ts_;
        int    best_p   = -1;
        double best_sc  = -1.0;
        uint64_t best_age = 0;
        std::list<ChunkKey>::iterator best_it;
        for (int p = 0; p < kMaxChunksPerTensor; ++p) {
            auto & bucket = buckets_[p];
            for (auto it = bucket.begin(); it != bucket.end(); ++it) {
                if (reserved_.count(*it)) continue;
                if (sched.is_replay_reserved(it->wid, it->cid)) continue;
                auto ix = index_.find(*it);
                if (ix == index_.end()) continue;  // defensive
                const uint64_t age = now - ix->second.last_touch_ts;
                const uint64_t cnt = ix->second.touch_count;
                if (age_plane_tie_window_ >= 0) {
                    const uint64_t win = (uint64_t)age_plane_tie_window_;
                    bool take = best_p < 0;
                    if (!take) {
                        if (age > best_age + win) {
                            take = true;
                        } else if (best_age <= age + win) {
                            take = p > best_p;
                        }
                    }
                    if (take) {
                        best_p = p;
                        best_age = age;
                        best_it = it;
                    }
                    break;
                }
                const double sc = plane_weight_ * (double)p
                                + age_weight_   * (double)age
                                + freq_weight_  / (double)(cnt + 1);
                if (sc > best_sc) {
                    best_sc  = sc;
                    best_p   = p;
                    best_it  = it;
                }
                break;  // only the bucket's front (= oldest within plane)
            }
        }
        if (best_p < 0) return false;
        ChunkKey victim = *best_it;
        buckets_[best_p].erase(best_it);
        index_.erase(victim);
        // Eviction-debug counters (constraint-violation gate).
        g_evictions_total_.fetch_add(1, std::memory_order_relaxed);
        // Dispatch-active counter: any chunk pinned in reserved_
        // means a dispatch is in flight. (Crude proxy — if reserved_
        // is non-empty we're inside someone's reserve+release window.)
        bool dispatch_active = false;
        size_t reserved_count;
        {
            // We're already holding mu_ from the caller's lock_guard
            // — these are unlocked reads inside that scope.
            reserved_count = reserved_.size();
            dispatch_active = reserved_count > 0;
        }
        if (dispatch_active) {
            g_evictions_during_dispatch_.fetch_add(
                1, std::memory_order_relaxed);
        }
        launch_diag::note_evicted_chunk(victim.cid);
        pool.evict(victim.wid, victim.cid);
        // Clear the per-plane device pointer-table slot async on the
        // pool's copy_stream (SSOT §6.1.6 step 6).  The worker is
        // holding ``io_stream_mu_`` so this emission is FIFO-ordered
        // against subsequent loads on the same stream — the next
        // chunk that lands in this slot writes a fresh non-null
        // pointer before any kernel reads through it.
        rt.clear_chunk_device_ptr(victim.wid, victim.cid,
                                   pool.copy_stream());
        return true;
    }

    // Debug counters for T5/cap-pressure investigation.
    static unsigned long long evictions_total() {
        return g_evictions_total_.load(std::memory_order_relaxed);
    }
    static unsigned long long evictions_during_dispatch() {
        return g_evictions_during_dispatch_.load(std::memory_order_relaxed);
    }
    static void reset_eviction_counters() {
        g_evictions_total_.store(0, std::memory_order_relaxed);
        g_evictions_during_dispatch_.store(0, std::memory_order_relaxed);
    }

    size_t size() const {
        std::lock_guard<std::mutex> lk(mu_);
        return index_.size();
    }

private:
    struct Entry {
        int plane;
        std::list<ChunkKey>::iterator it;
        uint64_t last_touch_ts;
        uint64_t touch_count;
    };
    std::array<std::list<ChunkKey>, kMaxChunksPerTensor> buckets_;
    std::unordered_map<ChunkKey, Entry, ChunkKeyHash> index_;
    std::unordered_set<ChunkKey, ChunkKeyHash> reserved_;
    uint64_t touch_ts_ = 0;
    double   plane_weight_;
    double   age_weight_;
    double   freq_weight_;
    int64_t  age_plane_tie_window_;
    mutable std::mutex mu_;

    // Static atomic counters (process-wide, all trackers share).
    static inline std::atomic<unsigned long long> g_evictions_total_{0};
    static inline std::atomic<unsigned long long> g_evictions_during_dispatch_{0};
};

}  // namespace dp_moe_ext
