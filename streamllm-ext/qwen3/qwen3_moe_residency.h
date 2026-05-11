// streamllm-ext — MoE residency tracker.
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
// Eviction policy is a weighted score:
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
//                  STREAMLLM_MOE_EVICT_FREQ_WEIGHT to enable.
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

#include <array>
#include <cstdint>
#include <list>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace streamllm_ext {

class MoEResidencyTracker {
public:
    MoEResidencyTracker(double plane_w, double age_w, double freq_w)
        : plane_weight_(plane_w), age_weight_(age_w), freq_weight_(freq_w) {}

    void touch(const std::string & wid, int cid, int plane) {
        if (plane < 0 || plane >= kMaxChunksPerTensor) return;
        std::lock_guard<std::mutex> lk(mu_);
        const uint64_t ts = ++touch_ts_;
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
                    StreamllmRuntime & rt,
                    const Scheduler & sched) {
        std::lock_guard<std::mutex> lk(mu_);
        const uint64_t now = touch_ts_;
        int    best_p   = -1;
        double best_sc  = -1.0;
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
        pool.evict(victim.wid, victim.cid);
        rt.clear_chunk_device_ptr(victim.wid, victim.cid);
        return true;
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
    mutable std::mutex mu_;
};

}  // namespace streamllm_ext
