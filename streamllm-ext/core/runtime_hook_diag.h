// streamllm-ext — cache-stack diagnostics.
//
// Per-rank / per-plane tier-hit buckets + STREAMLLM_TRACE_FILE
// chunk-request recorder. Pure measurement infrastructure: not load-
// bearing for runtime correctness or hot-path TPS.
//
// Gated by the STREAMLLM_DIAG cmake option. When OFF (default), the
// header below provides inline empty stubs and runtime_hook_diag.cpp
// is excluded from the build; callers compile down to no-ops.
// When ON, runtime_hook_diag.cpp provides the real bodies, the
// counter struct lives there, and the LOAD walk's classify/record
// site forwards each chunk request to ``record_chunk``.

#pragma once

#include <cstdint>
#include <string>

namespace streamllm_ext {

class StreamllmRuntime;     // forward decl

namespace diag {

// Per-call-site labels for move_chunk timing buckets. Used by
// record_move_ns to attribute the software-overhead term that
// ROOFLINE.md reports (T_observed − T_bw_lower_bound).
enum class MoveSite : int {
    Pread = 0,            // ::pread into ring slot
    Xform,                // plane_disk_to_kernel
    HostCacheInsert,      // host_mu lock + e.host.chunks[p] = ...
    HostCacheSnapshot,    // cache-hit branch's xform_scratch.assign
    PoolLoadEnqueue,      // pool_->load wall (mutex + cudaMemcpyAsync)
    PtrUpdate,            // anybcq::update_per_plane_after_load_async
    MakeRoom,             // scheduler_->make_room_for retry loop
    Alloc,                // xform_scratch.assign cost
    _COUNT
};

// Per-chunk classification counters. Bumped in record_move_event.
enum class MoveEvent : int {
    SsdMiss = 0,
    DramHit,
    VramIdempotentHit,    // pool.load returned an already-resident slot
    CacheInsert,
    CacheSnapshot,
    _COUNT
};

#ifdef STREAMLLM_DIAG

void install_open_trace();
void clear_close_trace_and_dump();
void record_chunk(const StreamllmRuntime & rt,
                  const std::string &      synthetic_wid,
                  int                      rank_u,
                  int                      chunk_idx,
                  int                      cid);

// Add ``ns`` to the named per-site timing bucket. Cheap atomic add;
// no locks. Wrapped at every move_chunk call site.
void record_move_ns(MoveSite site, uint64_t ns);

// Increment a per-chunk classification counter.
void record_move_event(MoveEvent ev);

// Record a per-(layer, token, rank) gate event from the LOAD walk.
// Bumps the gate-score and cum-before histograms in the diag dump
// AND, when STREAMLLM_GATE_TRACE_FILE is set, appends a CSV row
// (layer, token_idx, rank, expert_id, gate_score, cum_before,
// desired_chunks). Lets an offline analyzer replay any rank_cum
// policy against the recorded score distribution.
void record_gate_event(int layer, int token_idx, int rank,
                       int expert_id, float gate_score,
                       float cum_before, int desired_chunks);

#else

inline void install_open_trace() {}
inline void clear_close_trace_and_dump() {}
inline void record_chunk(const StreamllmRuntime &,
                         const std::string &,
                         int, int, int) {}
inline void record_move_ns(MoveSite, uint64_t) {}
inline void record_move_event(MoveEvent) {}
inline void record_gate_event(int, int, int, int, float, float, int) {}

#endif

}} // namespace streamllm_ext::diag
