// streamllm-ext — MoE scheduler.
//
// MoEScheduler: per-expert dynamic precision allocation driven by a
// gate-score threshold table. on_install() pins the [0, MIN) planes
// per expert via rt.move_chunk(); plan_for_expert() computes the
// desired chunk count for each (canonical, expert_id, gate_score)
// the runtime hook fires.

#include "scheduler.h"
#include "runtime.h"
#include "anybcq_gemv.h"
#include "moe_fused.h"   // MoeExpertTable + qwen3::alloc/free_moe_expert_table
#include "qwen3_moe_residency.h"
#include "qwen3_moe_dispatch.h"

#include <ggml.h>          // ggml_tensor field access for claims_tensor
#include <cuda_runtime.h>

#include <array>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <list>
#include <stdexcept>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <string>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

namespace streamllm_ext {

namespace {

// Helpers to build the "all chunks of a tensor" list. Each managed
// tensor is 1 q_bias chunk + P data chunks (a data chunk holds signs
// + α for that plane, packed into one VRAM slot).
std::vector<int> all_chunks_for(int P) {
    std::vector<int> out;
    out.reserve(P + 1);
    out.push_back(kCidQBias);
    for (int p = 0; p < P; ++p) out.push_back(cid_chunk(p));
    return out;
}

std::vector<MoveOp> all_moves_for(const std::string & wid, int P) {
    std::vector<MoveOp> out;
    out.reserve(P + 1);
    out.push_back({wid, kCidQBias, Tier::RAM, Tier::VRAM});
    for (int p = 0; p < P; ++p)
        out.push_back({wid, cid_chunk(p), Tier::RAM, Tier::VRAM});
    return out;
}

// Read [offset, offset+nbytes) from path into a host buffer.
//
// Uses pread(2) directly + posix_fadvise(DONTNEED) so the kernel
// drops the just-read pages from its page cache immediately. The
// install path reads tens of GB of chunk bytes sequentially and
// without DONTNEED the kernel caches all of it, evicting the user's
// working set and triggering swap thrashing.
std::vector<uint8_t> pread_range(const std::string & path,
                                 int64_t offset, int64_t nbytes) {
    int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        throw std::runtime_error(
            "scheduler: cannot open " + path + ": " + std::strerror(errno));
    }
    std::vector<uint8_t> buf((size_t)nbytes);
    int64_t total = 0;
    while (total < nbytes) {
        ssize_t got = ::pread(fd, buf.data() + total,
                              (size_t)(nbytes - total),
                              (off_t)(offset + total));
        if (got < 0) {
            int err = errno;
            ::close(fd);
            throw std::runtime_error(
                "scheduler: pread failed on " + path + ": " + std::strerror(err));
        }
        if (got == 0) {
            ::close(fd);
            throw std::runtime_error(
                "scheduler: short read on " + path +
                " (wanted " + std::to_string(nbytes) +
                ", got " + std::to_string(total) + ")");
        }
        total += got;
    }
    // Tell the kernel we won't re-read these bytes. POSIX_FADV_DONTNEED
    // schedules the page cache to drop these pages, freeing system RAM.
    // No-op on filesystems / kernels that don't honour it.
    (void)::posix_fadvise(fd, (off_t)offset, (off_t)nbytes, POSIX_FADV_DONTNEED);
    ::close(fd);
    return buf;
}

// Parse one managed tensor's GGUF byte region into an UpstreamLayoutHost
// and register a zero-filled device layout with the runtime. The
// device pointers are back-filled by rt.move_chunk() on each upload.
UpstreamLayoutHost read_one(const StreamReader & reader,
                            const std::string & gguf_path,
                            const std::string & name) {
    const TensorLayout * layout = reader.layout(name);
    if (layout == nullptr) {
        throw std::runtime_error("scheduler: layout missing for " + name);
    }
    auto raw = pread_range(gguf_path, layout->tensor_offset, layout->tensor_nbytes);
    try {
        return build_upstream_layout_host(
            *layout, raw.data(), reader.global().group_size);
    } catch (const std::exception & e) {
        throw std::runtime_error(
            "scheduler: build failed for " + name + ": " + e.what());
    }
}

// Build per-chunk file offset / size vectors from a TensorLayout and
// hand them to the runtime so move_chunk can pread on demand when
// host bytes have been released. Pure metadata copy — doesn't touch
// the file.
void register_chunk_io_from_layout(StreamllmRuntime & rt,
                                   const StreamReader & reader,
                                   const std::string & name) {
    const TensorLayout * layout = reader.layout(name);
    if (layout == nullptr) return;
    std::vector<int64_t> offs;
    std::vector<int64_t> sizes;
    offs.reserve(layout->chunk_bytes.size());
    sizes.reserve(layout->chunk_bytes.size());
    int64_t cursor = layout->tensor_offset + (int64_t)layout->fixed_bytes;
    for (uint32_t bytes : layout->chunk_bytes) {
        offs.push_back(cursor);
        sizes.push_back((int64_t)bytes);
        cursor += (int64_t)bytes;
    }
    rt.register_chunk_io(name, std::move(offs), std::move(sizes));
}


// ---------------------------------------------------------------------
// MoEScheduler
// ---------------------------------------------------------------------
//
// Per-expert dynamic precision policy:
//
//   At install (per synthetic wid <canonical>:e<X>):
//     - Pin q_bias + planes [0, MIN). Quality floor; never evicted.
//     - Planes [MIN, MAX) start unloaded; loaded on demand by
//       plan_for_expert when gate score warrants.
//
//   At dispatch time (mul_mat_id hook calls plan_for_expert per
//   selected expert):
//     - desired = clamp(precision_from_gate(score), MIN, MAX)
//     - on-demand planes ride the VramChunkPool LRU so cap pressure
//       auto-evicts least-recently-used on-demand chunks (pinned
//       MIN planes are never evictable).
//
// Two precision policies (selected via STREAMLLM_MOE_POLICY):
//   "static"  : every routed expert runs at uniform precision = max.
//               Loads the same chunks for every expert regardless of gate
//               score. Use this for original-quality prefill.
//   "dynamic" : per-expert precision derived from gate score. Two ladder
//               modes:
//               (a) Linear (default): precision = floor(gate/step)+1
//                   clamped to [min, max].
//               (b) Threshold table (set STREAMLLM_MOE_GATE_THRESHOLDS):
//                   precision = max{n : gate >= t_n} clamped to
//                   [min, max]. Lets the operator place each
//                   precision step at an arbitrary gate value.
//               Planes [0, min) are pinned at install (never evicted);
//               planes [min, max) are loaded on demand and rotate via
//               LRU. Same code at decode (single-token gate) and at
//               batched prefill (max-gate aggregated per expert).
//
// Env vars:
//   STREAMLLM_MOE_POLICY           "static" | "dynamic" | "score"
//                                  (default static)
//   STREAMLLM_MOE_GATE_STEP        dynamic linear ladder step (default 0.1)
//   STREAMLLM_MOE_GATE_THRESHOLDS  dynamic threshold table: CSV
//                                  "t_{MIN+1},...,t_MAX". Overrides linear
//                                  ladder. Each threshold is a raw gate
//                                  score; precision per expert =
//                                  max{n: gate_score >= t_n}.
//   STREAMLLM_MOE_SCORE_THRESHOLDS score-policy threshold table: CSV
//                                  DESCENDING raw gate-score cutoffs;
//                                  e.g. "0.4,0.3,0.1,0.05".
//   STREAMLLM_MOE_SCORE_CHUNKS     score-policy precision-per-bucket: CSV
//                                  same length as SCORE_THRESHOLDS.
//                                  Per-(t, u) precision = chunks[k] for
//                                  the first k where g >= thresholds[k];
//                                  below the smallest threshold falls back
//                                  to chunks.back(). Per-expert aggregation
//                                  takes max(g) over (t, u) hitting the
//                                  expert and re-runs the same lookup, so
//                                  every (t, u) routed to expert e shares
//                                  the same precision derived from e's
//                                  highest-scoring token in the minibatch.
//                                  Default "8,6,4,2" if thresholds set.
class MoEScheduler : public Scheduler {
public:
    void on_install(StreamllmRuntime & rt,
                    const StreamReader & reader,
                    const std::string & gguf_path) override {
        rt_ = &rt;
        // No install-time plane pinning: every chunk loads on demand.
        // Precision per (token, rank) is set entirely by the policy
        // tables (score_chunks / gate_thresholds). Only the q_bias
        // chunks are uploaded at install (always needed by the kernel;
        // ~3 % of total chunk bytes).

        // Policy: "static" (default), "dynamic" (gate-threshold
        // ladder), or "score" (per-expert max gate score → threshold
        // ladder; the hook computes precision and calls
        // plan_for_expert_with_precision directly).
        policy_ = "static";
        if (const char * pol = getenv("STREAMLLM_MOE_POLICY")) {
            if (std::strcmp(pol, "dynamic") == 0 ||
                std::strcmp(pol, "DYNAMIC") == 0) {
                policy_ = "dynamic";
            } else if (std::strcmp(pol, "score") == 0 ||
                       std::strcmp(pol, "SCORE") == 0) {
                policy_ = "score";
            }
        }
        dynamic_policy_ = (policy_ == "dynamic");

        gate_step_ = read_float_env("STREAMLLM_MOE_GATE_STEP", 0.1f);
        if (gate_step_ <= 0.0f) gate_step_ = 0.1f;

        auto parse_csv = [](const char * csv,
                             std::vector<float> & out) {
            const char * p = csv;
            while (*p) {
                char * end = nullptr;
                float v = std::strtof(p, &end);
                if (end == p) break;
                out.push_back(v);
                p = end;
                while (*p == ',' || *p == ' ') ++p;
            }
        };

        // Optional CSV threshold table for "dynamic" policy:
        // t_{MIN+1},t_{MIN+2},...,t_MAX. Empty = linear ladder.
        if (const char * csv = getenv("STREAMLLM_MOE_GATE_THRESHOLDS")) {
            parse_csv(csv, gate_thresholds_);
        }
        // Score policy tables. Hook reads these to decide precision
        // per expert via lookup(max_g_for_expert).
        if (const char * csv = getenv("STREAMLLM_MOE_SCORE_THRESHOLDS")) {
            parse_csv(csv, score_thresholds_);
        }
        if (const char * csv = getenv("STREAMLLM_MOE_SCORE_CHUNKS")) {
            std::vector<float> tmp;
            parse_csv(csv, tmp);
            score_chunks_.clear();
            score_chunks_.reserve(tmp.size());
            for (float v : tmp) score_chunks_.push_back((int)v);
        }
        // Default chunks list to "8,6,4,2" if user only set thresholds.
        if (!score_thresholds_.empty() && score_chunks_.empty()) {
            score_chunks_ = {8, 6, 4, 2};
        }

        size_t n_total = 0;
        // DRAM cache cap (host tier). 0 = disabled; the runtime still
        // populates host.chunks[p] on SSD-stream as a side-effect cache,
        // but the scheduler doesn't evict — so total host bytes grow
        // without bound (limited only by total managed bytes ≈ 22-35 GB
        // for Qwen3-30B-A3B). When > 0, the scheduler maintains a
        // process-wide LRU of (wid, cid) host-resident chunks and
        // calls rt.release_chunk_host once total bytes exceed the cap.
        host_max_bytes_ = 0;
        if (const char * s = std::getenv("STREAMLLM_HOST_CAP_MB")) {
            long mb = std::atol(s);
            if (mb > 0) host_max_bytes_ = (size_t)mb * 1024ULL * 1024ULL;
        }

        size_t total_pinned = 0, total_on_demand = 0;
        for (const auto & name : reader.chunked_tensor_names()) {
            UpstreamLayoutHost host = read_one(reader, gguf_path, name);
            const int P = host.n_chunks;
            const bool any_prec = host.any_precision;
            // Snapshot before std::move(host) so the host LRU has a
            // consistent per-tensor byte size for cap accounting.
            // Shortcut: every chunk shares ``bytes_per_chunk`` (this is
            // the per-chunk size, no max-vs-min ambiguity).
            // Any-prec: per-chunk sizes differ (chunk 0 holds base_p
            // planes, later chunks hold 1) so snapshot the per-chunk
            // ``kernel_chunk_bytes`` array for accurate LRU accounting.
            const size_t bytes_per_chunk = host.bytes_per_chunk;
            std::vector<size_t> per_chunk_bytes_anyprec;
            if (any_prec) {
                per_chunk_bytes_anyprec.reserve(host.chunk_planes.size());
                for (const auto & cp : host.chunk_planes) {
                    per_chunk_bytes_anyprec.push_back(cp.kernel_chunk_bytes);
                }
            }

            rt.register_layout(name, std::move(host), UpstreamLayoutDevice{});
            // Stash per-chunk file offsets so move_chunk can SSD-stream
            // on demand after release_host_bytes.
            register_chunk_io_from_layout(rt, reader, name);

            if (!any_prec) {
                // Shortcut layout: upload q_bias once at install (small,
                // hot — every kernel call reads it). Plane chunks load on
                // demand. Plan = q_bias only at steady state.
                rt.move_chunk(name, kCidQBias, Tier::RAM, Tier::VRAM);
                base_plans_.emplace(name, Plan{
                    /*chunks=*/{kCidQBias},
                    /*moves =*/{}
                });
                total_pinned += 1;
            } else {
                // Any-prec layout: each data chunk carries its own
                // α^(p)/β^(p). No install-time pin — chunks stream on
                // demand under the same VRAM/host cache policy the
                // shortcut path uses, with the precision tier driven by
                // the score policy at dispatch time.
                base_plans_.emplace(name, Plan{
                    /*chunks=*/{},
                    /*moves =*/{}
                });
            }

            P_of_[name]     = P;
            bytes_per_chunk_of_[name] = bytes_per_chunk;
            if (any_prec) {
                any_prec_wids_.insert(name);
                per_chunk_bytes_anyprec_[name] =
                    std::move(per_chunk_bytes_anyprec);
            }

            total_on_demand += (size_t)P;
            n_total         += 1;

            // Default: drop host chunk buffers — on-demand chunks
            // (above MIN) are SSD-streamed by move_chunk via the
            // chunk-io registration above. Saves O(46 GB) heap on
            // Qwen3-30B-A3B and lets on-demand loading work on
            // commodity (30 GB) hosts. q_bias is retained by
            // release_host_bytes (small, hot path).
            //
            // Override: STREAMLLM_KEEP_HOST_BYTES=1 keeps every chunk
            // in RAM. On a 1 TB-class host this turns each move_chunk
            // into a host→pinned memcpy + cudaMemcpyAsync, skipping
            // the pread which becomes the bottleneck under heavy
            // concurrency (~933 µs/call at 8 workers, scales to
            // 4 ms/call at 32 workers as the SSD path saturates).
            const char * keep_env = std::getenv("STREAMLLM_KEEP_HOST_BYTES");
            const bool keep_host_bytes =
                keep_env != nullptr && std::atoi(keep_env) != 0;
            if (!keep_host_bytes) {
                rt.release_host_bytes(name);
            }
        }
        std::fprintf(stderr,
            "streamllm-scheduler[moe]: %zu experts | policy=%s",
            n_total, policy_.c_str());
        if (policy_ == "dynamic") {
            if (gate_thresholds_.empty()) {
                std::fprintf(stderr, " | gate_step=%.3f", gate_step_);
            } else {
                std::fprintf(stderr, " | thresholds=[");
                for (size_t i = 0; i < gate_thresholds_.size(); ++i) {
                    std::fprintf(stderr, "%s%.3f",
                        i == 0 ? "" : ",", gate_thresholds_[i]);
                }
                std::fprintf(stderr, "]");
            }
        } else if (policy_ == "score") {
            // No swap can race here — set_score_table is only callable
            // post-install via the runtime hook. Still, take the lock
            // to be uniformly safe.
            std::lock_guard<std::mutex> lk(score_table_mu_);
            std::fprintf(stderr, " | score_thresholds=[");
            for (size_t i = 0; i < score_thresholds_.size(); ++i) {
                std::fprintf(stderr, "%s%.3f",
                    i == 0 ? "" : ",", score_thresholds_[i]);
            }
            std::fprintf(stderr, "] chunks=[");
            for (size_t i = 0; i < score_chunks_.size(); ++i) {
                std::fprintf(stderr, "%s%d",
                    i == 0 ? "" : ",", score_chunks_[i]);
            }
            std::fprintf(stderr, "]");
        }
        std::fprintf(stderr,
            " | %zu pinned chunks resident, %zu on-demand chunks available\n",
            total_pinned, total_on_demand);

        // Eagerly build all per-canonical MoeExpertTable instances now,
        // so the first mul_mat_id hook call doesn't pay the build cost
        // (~3 H2Ds × n_canonicals). Tables are slab-allocated via
        // small_alloc — cheap.
        std::unordered_set<std::string> canonical_seen;
        for (const auto & name : reader.chunked_tensor_names()) {
            // synthetic wid form: "<canonical>:e<N>"
            auto colon = name.rfind(":e");
            if (colon == std::string::npos) continue;
            const std::string canonical = name.substr(0, colon);
            if (!canonical_seen.insert(canonical).second) continue;
            (void) moe_expert_table(canonical);
        }
        std::fprintf(stderr,
            "streamllm-scheduler[moe]: built %zu expert tables eagerly\n",
            canonical_seen.size());
    }

    const Plan * plan_for(const std::string & tensor_name,
                          StreamHandle /*compute_stream*/) override {
        // Synthetic wids return their base plan. Canonical MoE names
        // never reach here — they're placeholder-only and excluded from
        // chunked_tensor_names(). The hook calls plan_for_expert
        // instead, which picks the right chunk count per gate_score.
        auto it = base_plans_.find(tensor_name);
        if (it == base_plans_.end()) return nullptr;
        return &it->second;
    }

    const Plan * plan_for_expert(const std::string & canonical_wid,
                                  int expert_id,
                                  float gate_score,
                                  int /*rank*/,
                                  StreamHandle /*compute_stream*/) {
        const std::string synthetic =
            canonical_wid + ":e" + std::to_string(expert_id);
        if (any_prec_wids_.count(synthetic)) {
            // Any-prec MoE — pick target precision (planes) from the
            // gate score using the same ladder the dense path uses, then
            // translate to a chunk count via base_precision.
            auto it_pa = P_of_.find(synthetic);
            const int Pa = it_pa == P_of_.end() ? 0 : it_pa->second;
            const auto * dev = rt_->layout(synthetic);
            const int base_p = dev ? (int)dev->base_precision : 1;
            // Reuse anyprec_desired_from_gate to map gate→planes; the
            // helper clamps to Pa so for any-prec it can over-budget when
            // P_target_planes > Pa, which the chunk-count translation
            // below re-clamps. Pass kMaxChunksPerTensor as the planes
            // ceiling to avoid double-clamping at the chunk level.
            int planes = anyprec_desired_from_gate(
                kMaxChunksPerTensor, gate_score);
            int n_chunks = planes - base_p + 1;
            if (n_chunks < 1)  n_chunks = 1;
            if (n_chunks > Pa) n_chunks = Pa;
            return build_plan_for_anyprec(synthetic, n_chunks);
        }
        auto it_p = P_of_.find(synthetic);
        if (it_p == P_of_.end()) return nullptr;
        const int P = it_p->second;

        // Gate-score → desired chunk count.
        //
        //   static  : every routed expert pulls all P planes.
        //   dynamic : (a) linear ladder  precision = floor(gate/step)+1
        //             (b) threshold table (if STREAMLLM_MOE_GATE_THRESHOLDS
        //                 set):  precision = max{k : gate >= t_k}.
        int desired;
        if (dynamic_policy_) {
            int n;
            if (!gate_thresholds_.empty()) {
                // Scan from highest threshold down; first hit yields k.
                n = 0;
                for (int i = (int)gate_thresholds_.size() - 1; i >= 0; --i) {
                    if (gate_score >= gate_thresholds_[i]) {
                        n = i + 1;
                        break;
                    }
                }
            } else {
                n = (int)std::floor(gate_score / gate_step_) + 1;
            }
            desired = std::max(1, std::min(n, P));
        } else {
            desired = P;
        }
        return build_plan_for(synthetic, desired);
    }

    // Direct-precision entry point used by the score policy: hook
    // already decided desired_precision for this (canonical, expert).
    const Plan * plan_for_expert_with_precision(
        const std::string & canonical_wid,
        int expert_id,
        int desired_precision,
        StreamHandle /*compute_stream*/) {
        const std::string synthetic =
            canonical_wid + ":e" + std::to_string(expert_id);
        if (any_prec_wids_.count(synthetic)) {
            // Any-prec MoE — score policy supplies ``desired_precision``
            // in PLANES (bits). For any-prec the planes loaded by the
            // first n_chunks chunks is base_precision + (n_chunks − 1),
            // so reaching a target plane count P_t needs
            //   n_chunks = max(1, P_t − base_precision + 1)
            // clamped to the encoded chunk count.
            auto it_pa = P_of_.find(synthetic);
            const int Pa = it_pa == P_of_.end() ? 0 : it_pa->second;
            const auto * dev = rt_->layout(synthetic);
            const int base_p = dev ? (int)dev->base_precision : 1;
            int n_chunks = desired_precision - base_p + 1;
            if (n_chunks < 1)  n_chunks = 1;
            if (n_chunks > Pa) n_chunks = Pa;
            return build_plan_for_anyprec(synthetic, n_chunks);
        }
        auto it_p = P_of_.find(synthetic);
        if (it_p == P_of_.end()) return nullptr;
        const int P = it_p->second;
        const int desired = std::max(1, std::min(desired_precision, P));
        return build_plan_for(synthetic, desired);
    }

  private:
    // Mirrors the dense gate-score → desired chunk-count map but applied
    // to any-prec wids.  Same env semantics as the shortcut path
    // (gate_thresholds_ table, otherwise linear ladder via gate_step_),
    // so STREAMLLM_MOE_GATE_THRESHOLDS / STREAMLLM_MOE_SCORE_THRESHOLDS
    // affect both layouts identically.
    int anyprec_desired_from_gate(int Pa, float gate_score) const {
        if (Pa <= 0) return 0;
        if (!dynamic_policy_) return Pa;
        int n;
        if (!gate_thresholds_.empty()) {
            n = 0;
            for (int i = (int)gate_thresholds_.size() - 1; i >= 0; --i) {
                if (gate_score >= gate_thresholds_[i]) { n = i + 1; break; }
            }
        } else {
            n = (int)std::floor(gate_score / gate_step_) + 1;
        }
        if (n < 1)  n = 1;
        if (n > Pa) n = Pa;
        return n;
    }

    // Streaming any-prec plan builder.  Emits chunks [0..desired);
    // chunk 0 holds the base_p jointly-fit planes (so desired=1 means
    // base precision), each subsequent chunk adds one extension plane
    // plus its own α^(p) / β^(p).  The runtime's move_chunk path handles
    // disk pread + kernel-format transform + per-plane pointer activation.
    const Plan * build_plan_for_anyprec(const std::string & synthetic,
                                         int desired) {
        auto it_p = P_of_.find(synthetic);
        if (it_p == P_of_.end()) return nullptr;
        const int P = it_p->second;
        if (desired < 1)  desired = 1;
        if (desired > P)  desired = P;

        for (int p = 0; p < desired; ++p) {
            tracker_.touch(synthetic, cid_chunk(p), p);
        }
        // Host LRU touch — uses the per-chunk byte sizes captured at
        // install (per_chunk_bytes_anyprec_) so cap accounting reflects
        // the actual variable chunk size, not the dense path's
        // single-bytes-per-chunk assumption.
        if (host_max_bytes_ > 0) {
            std::vector<HostKey> to_evict;
            {
                std::lock_guard<std::mutex> lk(host_lru_mu_);
                for (int p = 0; p < desired; ++p) {
                    HostKey k{synthetic, cid_chunk(p)};
                    auto pos_it = host_pos_.find(k);
                    if (pos_it != host_pos_.end()) {
                        host_lru_.splice(host_lru_.begin(), host_lru_,
                                         pos_it->second);
                    } else {
                        const size_t per_chunk =
                            bytes_for_chunk(synthetic, cid_chunk(p));
                        host_lru_.push_front(k);
                        host_pos_.emplace(k, host_lru_.begin());
                        host_used_bytes_ += per_chunk;
                    }
                }
                while (host_used_bytes_ > host_max_bytes_ &&
                       !host_lru_.empty()) {
                    HostKey victim = host_lru_.back();
                    host_lru_.pop_back();
                    host_pos_.erase(victim);
                    const size_t vb = bytes_for_chunk(victim.wid, victim.cid);
                    host_used_bytes_ = (vb > host_used_bytes_)
                                       ? 0 : host_used_bytes_ - vb;
                    to_evict.push_back(victim);
                }
            }
            for (const auto & v : to_evict) {
                rt_->release_chunk_host(v.wid, v.cid);
            }
        }

        const uint64_t key = ((uint64_t)reinterpret_cast<uintptr_t>(&it_p->second))
                              ^ 0xA17C0DECu
                              ^ ((uint64_t)desired << 16);
        auto it_plan = expert_plan_cache_.find(key);
        if (it_plan != expert_plan_cache_.end() &&
            it_plan->second.synthetic_wid == synthetic &&
            it_plan->second.desired == desired) {
            return &it_plan->second.plan;
        }

        // chunks[] follows the same convention as the dense path: a
        // kCidQBias sentinel up front, then the data chunks. Dispatch
        // computes precision as ``chunks.size() − 1`` so the sentinel
        // keeps that math correct without a per-encoder branch. For
        // any-prec the sentinel is just a placeholder — the runtime's
        // move_chunk path keeps q_bias pointers alive via per-chunk β.
        Plan plan;
        plan.chunks.reserve((size_t)desired + 1);
        plan.moves.reserve((size_t)desired);
        plan.chunks.push_back(kCidQBias);
        for (int p = 0; p < desired; ++p) {
            plan.chunks.push_back(cid_chunk(p));
            plan.moves.push_back({synthetic, cid_chunk(p),
                                   Tier::RAM, Tier::VRAM});
        }
        auto [it_new, _] = expert_plan_cache_.emplace(key,
            CachedExpertPlan{synthetic, desired, std::move(plan)});
        return &it_new->second.plan;
    }

    const Plan * build_plan_for(const std::string & synthetic,
                                 int desired) {
        auto it_p = P_of_.find(synthetic);
        if (it_p == P_of_.end()) return nullptr;
        const int P = it_p->second;
        desired = std::min(desired, P);

        // LRU touch on every call — must happen BEFORE the plan cache
        // hit branch, otherwise cached plans (the >99% post-warmup case)
        // would never refresh tracker positions.
        for (int p = 0; p < desired; ++p) {
            tracker_.touch(synthetic, cid_chunk(p), p);
        }
        // Host (DRAM) LRU touch — only when a cap is active. Each
        // (synthetic, cid) admitted to the LRU contributes
        // bytes_per_chunk to host_used_bytes_; eviction happens in
        // the same lock once the cap is breached.
        if (host_max_bytes_ > 0) {
            auto bp_it = bytes_per_chunk_of_.find(synthetic);
            const size_t per_chunk =
                bp_it == bytes_per_chunk_of_.end() ? 0 : bp_it->second;
            std::vector<HostKey> to_evict;
            {
                std::lock_guard<std::mutex> lk(host_lru_mu_);
                for (int p = 0; p < desired; ++p) {
                    HostKey k{synthetic, cid_chunk(p)};
                    auto pos_it = host_pos_.find(k);
                    if (pos_it != host_pos_.end()) {
                        host_lru_.splice(host_lru_.begin(), host_lru_,
                                         pos_it->second);
                    } else {
                        host_lru_.push_front(k);
                        host_pos_.emplace(k, host_lru_.begin());
                        host_used_bytes_ += per_chunk;
                    }
                }
                while (host_used_bytes_ > host_max_bytes_ &&
                       !host_lru_.empty()) {
                    HostKey victim = host_lru_.back();
                    host_lru_.pop_back();
                    host_pos_.erase(victim);
                    // ``bytes_for_chunk`` covers both shortcut and
                    // any-prec — the shared LRU may contain victims
                    // from either layout.
                    const size_t vb = bytes_for_chunk(victim.wid, victim.cid);
                    host_used_bytes_ = (vb > host_used_bytes_)
                                       ? 0 : host_used_bytes_ - vb;
                    to_evict.push_back(victim);
                }
            }
            // Drop the LRU mutex before calling into rt_: the runtime
            // takes its own per-Entry mutex, no risk of deadlock.
            for (const auto & v : to_evict) {
                rt_->release_chunk_host(v.wid, v.cid);
            }
        }

        // Cache: per (synthetic_wid, desired) → Plan.
        const uint64_t key = ((uint64_t)reinterpret_cast<uintptr_t>(&it_p->second))
                              ^ (uint64_t)desired;
        auto it_plan = expert_plan_cache_.find(key);
        if (it_plan != expert_plan_cache_.end() &&
            it_plan->second.synthetic_wid == synthetic &&
            it_plan->second.desired == desired) {
            return &it_plan->second.plan;
        }

        // Build plan: chunks [0, desired), moves for every plane.
        // q_bias is uploaded at install and stays resident.
        Plan plan;
        plan.chunks = build_chunks(desired);
        for (int p = 0; p < desired; ++p) {
            plan.moves.push_back({synthetic, cid_chunk(p),
                                  Tier::RAM, Tier::VRAM});
        }
        auto [it_new, _] = expert_plan_cache_.emplace(key,
            CachedExpertPlan{synthetic, desired, std::move(plan)});
        return &it_new->second.plan;
    }

  public:
    bool make_room_for(VramChunkPool & pool,
                        size_t /*nbytes_needed*/) override {
        // Single victim per call — runtime retries until the load
        // succeeds or this returns false. Passing rt_ lets the tracker
        // also clear the entry's per-plane device-pointer slot so the
        // kernel doesn't read stale memory after the freed VRAM slot
        // gets reused.
        return tracker_.make_room(pool, *rt_);
    }

    void reserve_for_dispatch(const std::string & wid, int cid) {
        tracker_.reserve(wid, cid);
    }
    void release_from_dispatch(const std::string & wid, int cid) {
        tracker_.release(wid, cid);
    }

    const MoeExpertTable * moe_expert_table(
        const std::string & canonical_wid) {
        // Cached per canonical_wid. The table just stores pointers to
        // each expert's pre-allocated d_chunk_qw_ptrs etc. — those
        // mutate in place when a chunk lands, so the table itself never
        // needs rebuilding.
        std::lock_guard<std::mutex> lk(moe_tables_mu_);
        auto it = moe_tables_.find(canonical_wid);
        if (it != moe_tables_.end()) return &it->second;

        std::vector<void *> qw, alpha, qbias;
        std::vector<void *> qbias_slot_addrs;  // any-prec only
        bool any_any_prec = false;
        for (int e = 0; ; ++e) {
            std::string synth = canonical_wid + ":e" + std::to_string(e);
            const auto * dev = rt_->layout(synth);
            if (dev == nullptr) break;  // first missing expert — done
            void ** d_qw    = rt_->anybcq_d_qw_ptrs(synth);
            void ** d_alpha = rt_->anybcq_d_alpha_ptrs(synth);
            void ** d_qbias = rt_->anybcq_d_qbias_slot(synth);
            const auto * host = rt_->host_layout(synth);
            if (d_qw == nullptr || d_alpha == nullptr) {
                std::fprintf(stderr,
                    "streamllm-scheduler[moe]: anybcq_d_*_ptrs missing for %s\n",
                    synth.c_str());
                return nullptr;
            }
            qw.push_back(d_qw);
            alpha.push_back(d_alpha);
            qbias.push_back(const_cast<void *>(dev->q_bias_fp16));
            // For any-prec experts, the install-time q_bias is null
            // (β lives per-chunk and is set by move_chunk on demand).
            // Capture each expert's d_qbias_slot so the dispatch path
            // can refresh ``d_q_bias_per_expert`` per kernel call.
            qbias_slot_addrs.push_back(d_qbias);
            if (host != nullptr && host->any_precision) any_any_prec = true;
        }
        if (qw.empty()) return nullptr;

        // Slab-allocate the three device arrays via small_alloc — one
        // bulk cudaMalloc at install instead of ~3 ms × 144 canonicals.
        const int n_experts = (int)qw.size();
        const size_t bytes = (size_t)n_experts * sizeof(void *);
        MoeExpertTable t;
        t.n_experts = n_experts;
        t.d_qw_planes_per_expert =
            (void ***) rt_->small_alloc(bytes);
        t.d_alpha_planes_per_expert =
            (void ***) rt_->small_alloc(bytes);
        t.d_q_bias_per_expert =
            (void **) rt_->small_alloc(bytes);
        if (t.d_qw_planes_per_expert == nullptr ||
            t.d_alpha_planes_per_expert == nullptr ||
            t.d_q_bias_per_expert == nullptr) {
            std::fprintf(stderr,
                "streamllm-scheduler[moe]: small_alloc exhausted for %s\n",
                canonical_wid.c_str());
            return nullptr;
        }
        // Three small H2Ds — much cheaper than three cudaMallocs.
        cudaMemcpy(t.d_qw_planes_per_expert,    qw.data(),
                   bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(t.d_alpha_planes_per_expert, alpha.data(),
                   bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(t.d_q_bias_per_expert,       qbias.data(),
                   bytes, cudaMemcpyHostToDevice);

        // For any-prec canonicals, also populate d_qbias_slot_per_expert
        // so the dispatch can refresh the table per kernel call from
        // each expert's current β buffer.
        if (any_any_prec) {
            t.d_qbias_slot_per_expert = (void ***) rt_->small_alloc(bytes);
            if (t.d_qbias_slot_per_expert == nullptr) {
                std::fprintf(stderr,
                    "streamllm-scheduler[moe]: small_alloc exhausted for "
                    "qbias_slot table on %s\n", canonical_wid.c_str());
                return nullptr;
            }
            cudaMemcpy(t.d_qbias_slot_per_expert,
                       qbias_slot_addrs.data(),
                       bytes, cudaMemcpyHostToDevice);
            t.needs_qbias_refresh = true;
        }

        auto [it_new, _] = moe_tables_.emplace(canonical_wid, t);
        return &it_new->second;
    }

    const char * name() const override { return "moe"; }

    // ── ggml-cuda hook overrides ─────────────────────────────────
    // The runtime registers thin extern-C shims with ggml-cuda; those
    // shims call ``g_runtime->scheduler().handle_*()``. Dispatch bodies
    // live in qwen3/qwen3_moe_dispatch.cpp.
    bool handle_mul_mat(StreamHandle stream,
                         const struct ggml_tensor * src0,
                         const struct ggml_tensor * src1,
                         struct ggml_tensor *       dst) override {
        return moe_dispatch::handle_mul_mat_impl(
            (cudaStream_t)stream, src0, src1, dst);
    }
    bool handle_mul_mat_id(StreamHandle stream,
                            const struct ggml_tensor * src0,
                            const struct ggml_tensor * src1,
                            const struct ggml_tensor * ids,
                            struct ggml_tensor *       dst) override {
        return moe_dispatch::handle_mul_mat_id_impl(
            (cudaStream_t)stream, src0, src1, ids, dst);
    }
    void on_topk_moe_observed(StreamHandle stream,
                               const struct ggml_tensor * logits,
                               struct ggml_tensor *       weights,
                               struct ggml_tensor *       ids) override {
        moe_dispatch::on_topk_moe_observed_impl(
            (cudaStream_t)stream, logits, weights, ids);
    }

    // Accessors the dispatch reads to look up the score-policy tables.
    // Hook computes per-expert precision = lookup(max_g_for_expert) and
    // sets every (t, u) routed to that expert to the same value.
    bool is_score_policy() const override {
        return policy_ == "score";
    }
    std::vector<float> score_thresholds_snapshot() const override {
        std::lock_guard<std::mutex> lk(score_table_mu_);
        return score_thresholds_;
    }
    std::vector<int> score_chunks_snapshot() const override {
        std::lock_guard<std::mutex> lk(score_table_mu_);
        return score_chunks_;
    }
    bool set_score_table(const std::vector<float> & th,
                          const std::vector<int>   & ch) override {
        // Accept the same shapes the env-var path accepts:
        //   ch.size() == th.size()       (each threshold pairs with a count)
        //   ch.size() == th.size() + 1   (last entry = tail, below all)
        // The dispatch lookup walks min(th, ch) descending; if no
        // threshold is met it falls back to ch.back(), so the +1 tail
        // form is the natural way to express a default chunk count.
        if (ch.empty()) return false;
        if (!(ch.size() == th.size() || ch.size() == th.size() + 1)) {
            return false;
        }
        for (size_t k = 1; k < th.size(); ++k) {
            if (th[k] > th[k - 1]) return false;  // must be descending
        }
        for (int n : ch) {
            if (n < 1 || n > kMaxChunksPerTensor) return false;
        }
        std::lock_guard<std::mutex> lk(score_table_mu_);
        score_thresholds_ = th;
        score_chunks_     = ch;
        // Promote the policy in case the user booted in static mode and
        // is now driving a score table over the wire — the dispatch
        // gates on policy_=="score".
        policy_ = "score";
        return true;
    }

private:
    static int read_int_env(const char * name, int fallback) {
        const char * s = getenv(name);
        if (!s || !s[0]) return fallback;
        return std::atoi(s);
    }
    static float read_float_env(const char * name, float fallback) {
        const char * s = getenv(name);
        if (!s || !s[0]) return fallback;
        return (float)std::atof(s);
    }
    static std::vector<int> build_chunks(int n) {
        std::vector<int> out;
        out.reserve(n + 1);
        out.push_back(kCidQBias);
        for (int p = 0; p < n; ++p) out.push_back(cid_chunk(p));
        return out;
    }

    StreamllmRuntime * rt_ = nullptr;

    // Static (uniform) vs dynamic (gate-threshold ladder) precision
    // policy. Selected at install via STREAMLLM_MOE_POLICY.
    bool  dynamic_policy_ = false;
    float gate_step_      = 0.1f;
    std::vector<float> gate_thresholds_;

    // Policy mode: "static" | "dynamic" | "score".
    // dynamic_policy_ caches policy_ == "dynamic" for the hot path.
    std::string policy_ = "static";

    // Score policy tables. score_thresholds_ is descending; for a given
    // gate score g, precision = score_chunks_[k] where k is the first
    // index with g >= score_thresholds_[k]. Below the smallest threshold
    // the lookup returns score_chunks_.back().
    //
    // Mutable across requests: set_score_table() swaps both vectors
    // under score_table_mu_. Hot-path readers (the LOAD walk) take a
    // snapshot via score_*_snapshot() and iterate locally so a swap
    // mid-generation never tears.
    mutable std::mutex score_table_mu_;
    std::vector<float> score_thresholds_;
    std::vector<int>   score_chunks_;

    std::unordered_map<std::string, int> P_of_;
    // Synthetic wids whose host layout flag had any_precision=true.
    // Used by plan_for_expert / plan_for_expert_with_precision to route
    // through the any-prec planning path (planes ↔ chunks translation +
    // build_plan_for_anyprec).
    std::unordered_set<std::string> any_prec_wids_;
    std::unordered_map<std::string, size_t> bytes_per_chunk_of_;
    // Any-prec only: per-(synthetic_wid) array of per-chunk byte sizes
    // (chunk 0 holds base_p planes worth, later chunks hold 1 plane).
    // Used by the host LRU so the cap reflects actual chunk sizes
    // instead of bytes_per_chunk_of_'s max-across-chunks figure.
    std::unordered_map<std::string, std::vector<size_t>>
        per_chunk_bytes_anyprec_;

    // Unified per-(wid, cid) byte size lookup. Returns 0 if unknown so
    // the LRU treats it as "weightless" (no eviction trigger from it).
    size_t bytes_for_chunk(const std::string & wid, int cid) const {
        if (!cid_is_chunk(cid)) return 0;
        auto it_any = per_chunk_bytes_anyprec_.find(wid);
        if (it_any != per_chunk_bytes_anyprec_.end()) {
            const int p = cid_chunk_index(cid);
            if (p >= 0 && p < (int)it_any->second.size()) {
                return it_any->second[p];
            }
            return 0;
        }
        auto it = bytes_per_chunk_of_.find(wid);
        return it == bytes_per_chunk_of_.end() ? 0 : it->second;
    }
    std::unordered_map<std::string, Plan> base_plans_;

    // Host (DRAM) tier LRU. host_max_bytes_ = 0 disables the cap and
    // skips the LRU bookkeeping entirely (the runtime still populates
    // host.chunks[p] as a side-effect cache; bytes accumulate without
    // bound). When > 0, build_plan_for touches the LRU on every plan
    // emit and evicts the tail past cap.
    struct HostKey {
        std::string wid;
        int         cid = 0;
        bool operator==(const HostKey & o) const {
            return cid == o.cid && wid == o.wid;
        }
    };
    struct HostKeyHash {
        size_t operator()(const HostKey & k) const {
            return std::hash<std::string>{}(k.wid)
                 ^ ((size_t)k.cid * 0x9E3779B97F4A7C15ULL);
        }
    };
    std::mutex                  host_lru_mu_;
    size_t                      host_used_bytes_ = 0;
    size_t                      host_max_bytes_  = 0;
    std::list<HostKey>          host_lru_;
    std::unordered_map<HostKey, std::list<HostKey>::iterator,
                       HostKeyHash> host_pos_;

    // Plane-priority residency tracker for on-demand chunks. Pinned
    // [0, MIN) planes are not tracked (never evicted by this scheduler);
    // make_room_for pops victims here when pool.load() runs out of
    // arena. Eviction score weights are env-tunable (defaults below).
    // freq_weight defaults to 0 (= recency-only behaviour). Streaming
    // bench at cap=8 GB showed TPS/quality were unmoved by freq_w
    // values in {1, 4} — the existing recency score already
    // approximates LFU because hot MoE experts get touched every layer
    // (most-recent ≈ most-frequent). Knob retained for experimentation
    // under different routing distributions.
    MoEResidencyTracker tracker_{
        read_float_env("STREAMLLM_MOE_EVICT_PLANE_WEIGHT", 8.0f),
        read_float_env("STREAMLLM_MOE_EVICT_AGE_WEIGHT",   1.0f),
        read_float_env("STREAMLLM_MOE_EVICT_FREQ_WEIGHT",  0.0f),
    };

    struct CachedExpertPlan {
        std::string synthetic_wid;
        int         desired;
        Plan        plan;
    };
    std::unordered_map<uint64_t, CachedExpertPlan> expert_plan_cache_;

    // Per-canonical-wid expert table for the fused MoE kernel. Built
    // lazily on first moe_expert_table() call; cached thereafter.
    std::unordered_map<std::string, MoeExpertTable> moe_tables_;
    std::mutex moe_tables_mu_;
};

} // anonymous


std::unique_ptr<Scheduler> make_scheduler(const char * which) {
    // Only the MoE scheduler is in production use. Empty/null defaults
    // to it; any other name errors out so users can't silently land on
    // a non-existent fallback.
    if (which == nullptr || which[0] == '\0' ||
        std::strcmp(which, "moe") == 0) {
        return std::make_unique<MoEScheduler>();
    }
    std::fprintf(stderr,
        "streamllm-scheduler: unknown scheduler '%s'; "
        "only 'moe' is supported\n",
        which);
    return nullptr;
}

// ─── Free-function entry points (qwen3_moe_scheduler.h) ───────────
//
// Concrete MoEScheduler is anonymously namespaced above; the dispatch
// glue calls into it via these typed helpers.  Static-cast is safe
// because make_scheduler currently returns exactly one concrete type;
// extending to multiple model schedulers would replace the cast with
// a typed accessor on the runtime + name() check.

namespace qwen3 {

namespace {
inline MoEScheduler & as_moe(Scheduler & s) {
    return static_cast<MoEScheduler &>(s);
}
}  // anon

const Plan * scheduler_plan_dense(
    Scheduler &         sched,
    const std::string & wid,
    StreamHandle        compute_stream)
{
    return as_moe(sched).plan_for(wid, compute_stream);
}

const Plan * scheduler_plan_for_expert(
    Scheduler &         sched,
    const std::string & canonical_wid,
    int                 expert_id,
    float               gate_score,
    int                 rank,
    StreamHandle        compute_stream)
{
    return as_moe(sched).plan_for_expert(
        canonical_wid, expert_id, gate_score, rank, compute_stream);
}

const Plan * scheduler_plan_for_expert_with_precision(
    Scheduler &         sched,
    const std::string & canonical_wid,
    int                 expert_id,
    int                 desired_precision,
    StreamHandle        compute_stream)
{
    return as_moe(sched).plan_for_expert_with_precision(
        canonical_wid, expert_id, desired_precision, compute_stream);
}

const MoeExpertTable * scheduler_moe_expert_table(
    Scheduler &         sched,
    const std::string & canonical_wid)
{
    return as_moe(sched).moe_expert_table(canonical_wid);
}

void scheduler_reserve_for_dispatch(
    Scheduler &         sched,
    const std::string & wid,
    int                 cid)
{
    as_moe(sched).reserve_for_dispatch(wid, cid);
}

void scheduler_release_from_dispatch(
    Scheduler &         sched,
    const std::string & wid,
    int                 cid)
{
    as_moe(sched).release_from_dispatch(wid, cid);
}

void scheduler_after_compute(
    Scheduler &         /*sched*/,
    const std::string & /*wid*/,
    StreamHandle        /*compute_stream*/)
{
    // MoEScheduler has no after_compute behaviour — the residency
    // tracker handles eviction lazily under pool pressure rather than
    // synchronously after each dispatch. Kept as a no-op so the
    // dispatch glue's call site stays untouched while the post-compute
    // hook is in flux (Step 4 graph instrumenter will fold this into
    // the marker callback).
}

}  // namespace qwen3

} // namespace streamllm_ext
