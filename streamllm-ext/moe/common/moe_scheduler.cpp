// streamllm-ext — MoE scheduler.
//
// MoEScheduler: per-expert dynamic precision allocation driven by a
// gate-score threshold table. on_install() pins the [0, MIN) planes
// per expert via rt.move_chunk(); plan_for_expert() computes the
// desired chunk count for each (canonical, expert_id, gate_score)
// the runtime hook fires.

#include "moe_scheduler.h"
#include "runtime.h"
#include "anybcq_gemv.h"
#include "fused_kernels.h"   // MoeExpertTable + qwen3::alloc/free_moe_expert_table
#include "matmul_comp.h"  // MoEMatMulComp (registered at install)
#include "residency.h"
#include "dispatch.h"
// S8 (Mode A): qwen3_graph_instrumenter retired with the legacy
// MUL_MAT_ID dispatch surface; the include is gone.
#include "tensor.h"   // anybcq::AnyBCQFamilyTensor (per-tensor pointer-table accessors)
#include "decoder/direct_matrix/tensor.h"

// (using-decl for qwen3::GraphInstrumenter retired in S8.)

#include <ggml.h>          // ggml_tensor field access for claims_tensor
#include <cuda_runtime.h>

#include <array>
#include <cassert>
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
#include <queue>
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
    if (reader.global().encoder == "direct_matrix") {
        return direct_matrix::build_upstream_layout_direct_matrix(
            *layout, raw.data());
    }
    if (reader.global().encoder == "direct_expert_block") {
        return direct_matrix::build_upstream_layout_direct_expert_block(
            *layout, raw.data());
    }
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
    std::vector<uint8_t> kernel_ready;
    std::vector<std::string> bundles;
    offs.reserve(layout->chunk_bytes.size());
    sizes.reserve(layout->chunk_bytes.size());
    kernel_ready.reserve(layout->chunk_bytes.size());
    bundles.reserve(layout->chunk_bytes.size());
    int64_t cursor = layout->tensor_offset + (int64_t)layout->fixed_bytes;
    for (size_t p = 0; p < layout->chunk_bytes.size(); ++p) {
        const uint32_t bytes = layout->chunk_bytes[p];
        const auto * slice = reader.bundle_slice(name, (uint32_t)cid_chunk((int)p));
        if (slice != nullptr && slice->kernel_ready) {
            offs.push_back(slice->offset);
            sizes.push_back(slice->size);
            kernel_ready.push_back(1);
            bundles.push_back(slice->bundle);
        } else {
            offs.push_back(cursor);
            sizes.push_back((int64_t)bytes);
            kernel_ready.push_back(0);
            bundles.emplace_back();
        }
        cursor += (int64_t)bytes;
    }
    rt.register_chunk_io(name, std::move(offs), std::move(sizes),
                         std::move(kernel_ready), std::move(bundles));
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
// Precision policy: one global KBar dial. Profile mode uses the
// residual table from STREAMLLM_KBAR_PROFILE_FILE; uniform mode rounds
// KBar to the same chunk count for every active expert.
class MoEScheduler : public Scheduler {
public:
    ~MoEScheduler() override {
        if (dynamic_R_d_ != nullptr) {
            cudaFree(dynamic_R_d_);
            dynamic_R_d_ = nullptr;
        }
    }

    void on_install(StreamllmRuntime & rt,
                    const StreamReader & reader,
                    const std::string & gguf_path) override {
        rt_ = &rt;
        // Benchmark/score-mode opt-in: when set, claims_node() returns
        // false so ggml-cuda is free to capture the cgraph. Safe only
        // when every chunk is pinned in VRAM (no SSD streaming, no
        // eviction). The fully-pinned check at the end of on_install
        // aborts loudly if pool capacity is too small.
        if (const char * s = std::getenv("STREAMLLM_ALLOW_CAPTURE")) {
            if (s[0] && s[0] != '0') allow_capture_ = true;
        }
        // Snapshot every managed name (chunked synthetic wids AND
        // placeholder-only canonicals).  claims_tensor / claims_node /
        // dispatch_node consult this set.  Used to live as
        // StreamllmRuntime::managed_names_ but moved here in P2★
        // because "which tensors does this scheduler claim" is a
        // scheduler-policy fact, not a runtime fact.
        for (const auto & name : reader.managed_tensor_names()) {
            managed_names_.insert(name);
        }
        size_t n_total = 0;
        // DRAM cache cap (host tier). STREAMLLM_HOST_CACHE=0 disables
        // the transformed-chunk DRAM cache entirely for embedded-style
        // pure VRAM↔SSD streaming. Otherwise, 0 = uncapped cache and
        // >0 makes the scheduler maintain a process-wide host LRU.
        host_cache_enabled_ = true;
        if (const char * s = std::getenv("STREAMLLM_HOST_CACHE")) {
            host_cache_enabled_ = !(s[0] == '0');
        }
        if (const char * s = std::getenv("STREAMLLM_DISABLE_HOST_CACHE")) {
            if (s[0] && s[0] != '0') host_cache_enabled_ = false;
        }
        host_max_bytes_ = 0;
        if (host_cache_enabled_ &&
            (std::getenv("STREAMLLM_HOST_CAP_MB") != nullptr)) {
            const char * s = std::getenv("STREAMLLM_HOST_CAP_MB");
            long mb = std::atol(s);
            if (mb > 0) host_max_bytes_ = (size_t)mb * 1024ULL * 1024ULL;
        }

        size_t total_pinned = 0, total_on_demand = 0;
        int max_n_chunks_ = 0;
        for (const auto & name : reader.chunked_tensor_names()) {
            UpstreamLayoutHost host = read_one(reader, gguf_path, name);
            const int P = host.n_chunks;
            const bool any_prec = host.any_precision;
            const bool direct_matrix = host.direct_matrix;
            // Snapshot before std::move(host) so the host LRU has a
            // consistent per-tensor byte size for cap accounting.
            // SsAnybcq: every chunk shares ``bytes_per_chunk`` (this is
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

            if (direct_matrix) {
                // Direct-matrix baseline: one native GGML expert matrix
                // per chunk, no q_bias/fixed-meta upload. The active expert
                // chunk streams on demand through the same pool path.
                base_plans_.emplace(name, Plan{
                    /*chunks=*/{},
                    /*moves =*/{}
                });
            } else if (!any_prec) {
                // SsAnybcq layout: upload q_bias once at install (small,
                // hot — every kernel call reads it). Plane chunks load on
                // demand. Plan = q_bias only at steady state.
                //
                // Benchmark/capture mode (STREAMLLM_ALLOW_CAPTURE=1):
                // pre-pin every chunk at install — the load_set is
                // permanently empty and the dispatch's residency checks
                // all short-circuit on first probe.
                rt.move_chunk(name, kCidQBias, Tier::RAM, Tier::VRAM);
                if (allow_capture_) {
                    for (int p = 0; p < P; ++p) {
                        rt.move_chunk(name, cid_chunk(p), Tier::RAM, Tier::VRAM);
                    }
                    std::vector<int> all_chunks;
                    all_chunks.reserve(P + 1);
                    all_chunks.push_back(kCidQBias);
                    for (int p = 0; p < P; ++p) all_chunks.push_back(cid_chunk(p));
                    base_plans_.emplace(name, Plan{ all_chunks, {} });
                    total_pinned += (size_t)(P + 1);
                } else {
                    base_plans_.emplace(name, Plan{
                        /*chunks=*/{kCidQBias},
                        /*moves =*/{}
                    });
                    total_pinned += 1;
                }
            } else {
                // Any-prec layout: each data chunk carries its own
                // α^(p)/β^(p). No install-time pin — chunks stream on
                // demand under the same VRAM/host cache policy the
                // ss_anybcq path uses, with the precision tier driven by
                // the score policy at dispatch time.
                //
                // Benchmark/capture mode pins every plane chunk eagerly
                // for the same reason as ss_anybcq above.
                if (allow_capture_) {
                    for (int p = 0; p < P; ++p) {
                        rt.move_chunk(name, cid_chunk(p), Tier::RAM, Tier::VRAM);
                    }
                    std::vector<int> all_chunks;
                    all_chunks.reserve(P);
                    for (int p = 0; p < P; ++p) all_chunks.push_back(cid_chunk(p));
                    base_plans_.emplace(name, Plan{ all_chunks, {} });
                    total_pinned += (size_t)P;
                } else {
                    base_plans_.emplace(name, Plan{
                        /*chunks=*/{},
                        /*moves =*/{}
                    });
                }
            }

            if (P > max_n_chunks_) max_n_chunks_ = P;
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
            "streamllm-scheduler[moe]: %zu experts | max_chunks=%d",
            n_total, max_n_chunks_);
        std::fprintf(stderr,
            " | %zu pinned chunks resident, %zu on-demand chunks available\n",
            total_pinned, total_on_demand);

        // Optional profile-backed dispatch residuals.
        // overrides everything when present (mostly for dev / when the
        // GGUF didn't carry residuals).  Binary format:
        //   uint32_t n_layers
        //   uint32_t n_experts
        //   uint32_t K_min
        //   uint32_t K_max
        //   float32  R[n_layers * n_experts * (K_max - K_min + 1)]
        if (const char * path = std::getenv("STREAMLLM_KBAR_PROFILE_FILE")) {
            std::FILE * f = std::fopen(path, "rb");
            if (f == nullptr) {
                std::fprintf(stderr,
                    "streamllm-scheduler[moe]: STREAMLLM_KBAR_PROFILE_FILE=%s "
                    "fopen failed: %s\n", path, std::strerror(errno));
            } else {
                uint32_t hdr[4] = {0, 0, 0, 0};
                if (std::fread(hdr, sizeof(uint32_t), 4, f) != 4) {
                    std::fprintf(stderr,
                        "streamllm-scheduler[moe]: residuals file header short\n");
                    std::fclose(f);
                } else {
                    const int n_lay = (int) hdr[0];
                    const int n_exp = (int) hdr[1];
                    const int Kmn   = (int) hdr[2];
                    const int Kmx   = (int) hdr[3];
                    const int nK    = Kmx - Kmn + 1;
                    if (n_lay > 0 && n_exp > 0 && Kmn >= 0 && nK > 0 &&
                        n_lay <= 256 && n_exp <= 4096 && nK <= 16) {
                        const size_t n = (size_t) n_lay * n_exp * nK;
                        std::vector<float> R(n);
                        if (std::fread(R.data(), sizeof(float), n, f) == n) {
                            this->set_dynamic_residuals(R, n_lay, n_exp, Kmn, Kmx);
                        } else {
                            std::fprintf(stderr,
                                "streamllm-scheduler[moe]: short read on residuals "
                                "(want %zu floats)\n", n);
                        }
                    } else {
                        std::fprintf(stderr,
                            "streamllm-scheduler[moe]: bad residuals header "
                            "L=%d E=%d K=[%d, %d]\n", n_lay, n_exp, Kmn, Kmx);
                    }
                    std::fclose(f);
                }
            }
        }
        if (const char * mode = std::getenv("STREAMLLM_KBAR_ALLOCATOR")) {
            if (std::strcmp(mode, "uniform") == 0) {
                kbar_allocator_mode_ = qwen3::KBarAllocatorMode::Uniform;
            } else {
                kbar_allocator_mode_ = qwen3::KBarAllocatorMode::Profile;
            }
        }
        // STREAMLLM_KBAR is the single global dial. In profile mode it
        // is the per-dispatch average chunk budget; in uniform mode it
        // rounds to a fixed chunk count for every active expert.
        if (const char * s = std::getenv("STREAMLLM_KBAR")) {
            dynamic_kbar_.store((float)std::atof(s),
                                   std::memory_order_relaxed);
        }

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

        // Build one MoEMatMulComp per managed canonical.  The dispatch
        // shim hands these off to ``StreamllmRuntime::run`` (Step 2 of
        // the unified chunked-load + compute lifecycle).  We size each
        // comp's pinned input buffers off STREAMLLM_BATCH_N_MAX
        // (default 2048; same env knob that sizes the MoE scratch).
        int max_n_tokens = 2048;
        if (const char * s = std::getenv("STREAMLLM_BATCH_N_MAX")) {
            int v = std::atoi(s); if (v > 0) max_n_tokens = v;
        }
        size_t comps_built = 0;
        for (const auto & canonical : canonical_seen) {
            const MoeExpertTable * table = nullptr;
            {
                std::lock_guard<std::mutex> lk(moe_tables_mu_);
                auto it = moe_tables_.find(canonical);
                if (it != moe_tables_.end()) table = &it->second;
            }
            const auto * any_layout = rt_->layout(canonical + ":e0");
            if (table == nullptr || any_layout == nullptr) continue;
            const int n_experts = table->n_experts;
            // n_used_per_tok is at most n_experts. Some warmup graphs
            // capture an "all-experts" pass (ids->ne[0] == n_experts),
            // so the comp's pinned buffers must accommodate that worst
            // case.  Floor at 32 to keep small-expert canonicals' tu
            // count predictable for the kernel-arg fast path.
            const int max_n_used = std::max(32, n_experts);

            auto comp = std::make_unique<qwen3::MoEMatMulComp>(
                rt, *this, canonical, table, any_layout,
                n_experts, max_n_tokens, max_n_used);
            moe_comps_.emplace(canonical, std::move(comp));
            ++comps_built;
        }
        std::fprintf(stderr,
            "streamllm-scheduler[moe]: built %zu MoEMatMulComp instances\n",
            comps_built);

        // Fully-pinned guard for the capture opt-in (benchmark/score
        // mode). Sum every managed chunk's on-disk byte size and
        // compare to the VRAM pool capacity. If the pool can't hold
        // the whole model, capture is unsafe (eviction would
        // invalidate captured chunk pointers mid-replay) — abort
        // rather than silently corrupt graphs.
        if (allow_capture_) {
            size_t total_managed_bytes = 0;
            for (const auto & kv : P_of_) {
                const auto & nm = kv.first;
                const int    P  = kv.second;
                auto it_any = per_chunk_bytes_anyprec_.find(nm);
                if (it_any != per_chunk_bytes_anyprec_.end()) {
                    for (size_t b : it_any->second) total_managed_bytes += b;
                } else {
                    auto it_b = bytes_per_chunk_of_.find(nm);
                    if (it_b != bytes_per_chunk_of_.end()) {
                        // ss_anybcq: q_bias + P plane chunks all
                        // share bytes_per_chunk.
                        total_managed_bytes += it_b->second * (size_t)(P + 1);
                    }
                }
            }
            const size_t cap = rt.pool().capacity_bytes();
            if (cap < total_managed_bytes) {
                std::fprintf(stderr,
                    "streamllm-scheduler[moe]: STREAMLLM_ALLOW_CAPTURE=1 but "
                    "pool capacity %.2f GB < total managed %.2f GB. "
                    "Capture mode requires fully-pinned VRAM. Set "
                    "STREAMLLM_VRAM_CAP_MB high enough to fit the whole "
                    "model, or unset STREAMLLM_ALLOW_CAPTURE.\n",
                    (double)cap / 1e9, (double)total_managed_bytes / 1e9);
                GGML_ABORT("streamllm-scheduler: capture mode misconfigured");
            }
            std::fprintf(stderr,
                "streamllm-scheduler[moe]: capture mode ENABLED — "
                "pool %.2f GB >= managed %.2f GB (version-keyed graph cache)\n",
                (double)cap / 1e9, (double)total_managed_bytes / 1e9);

            // Device-side mirror of the dynamic-residuals table — used
            // by the K̄-knapsack plan kernel (capture-mode rung 2).
            // Allocated lazily here; the H2D in set_dynamic_residuals
            // does the actual upload when the residuals first arrive.
            // If residuals are already loaded by this point (env-var
            // path), upload them now.
            if (!dynamic_R_.empty()) {
                upload_dynamic_R_to_device_unlocked();
            }
        }
    }

    qwen3::MoEMatMulComp * lookup_moe_comp(const std::string & canonical) {
        auto it = moe_comps_.find(canonical);
        if (it == moe_comps_.end()) return nullptr;
        return it->second.get();
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

    // Direct chunk-count entry point.  The model layer
    // (MoEMatMulComp::plan) has already decided how many chunks this
    // expert needs from the score-table dial; we just emit a Plan
    // whose chunks list matches.  Plane semantics never appear here;
    // any plane↔chunk translation lives in decoder/anybcq.
    const Plan * plan_for_expert_with_chunks(
        const std::string & canonical_wid,
        int expert_id,
        int n_chunks_requested,
        StreamHandle /*compute_stream*/) {
        const std::string synthetic =
            canonical_wid + ":e" + std::to_string(expert_id);
        if (any_prec_wids_.count(synthetic)) {
            auto it_pa = P_of_.find(synthetic);
            const int Pa = it_pa == P_of_.end() ? 0 : it_pa->second;
            int n_chunks = n_chunks_requested;
            if (n_chunks < 1)  n_chunks = 1;
            if (n_chunks > Pa) n_chunks = Pa;
            return build_plan_for_anyprec(synthetic, n_chunks);
        }
        auto it_p = P_of_.find(synthetic);
        if (it_p == P_of_.end()) return nullptr;
        const int P = it_p->second;
        const int n_chunks = std::max(1, std::min(n_chunks_requested, P));
        return build_plan_for(synthetic, n_chunks);
    }

  private:
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
        if (host_cache_enabled_ && host_max_bytes_ > 0) {
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

        const std::string key = synthetic + "#" + std::to_string(desired);
        auto it_plan = expert_plan_cache_.find(key);
        if (it_plan != expert_plan_cache_.end()) {
            // Cache-content sanity check: cannot fail with a string
            // key, but keep the assert so a future refactor (e.g.
            // resurrecting the integer key) trips loudly instead of
            // silently returning the wrong plan.
            assert(it_plan->second.synthetic_wid == synthetic);
            assert(it_plan->second.desired       == desired);
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
        if (host_cache_enabled_ && host_max_bytes_ > 0) {
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
                    // ``bytes_for_chunk`` covers both ss_anybcq and
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
        // String key — see comment on expert_plan_cache_ for the
        // uint64-key collision bug this avoids.
        const std::string key = synthetic + "#" + std::to_string(desired);
        auto it_plan = expert_plan_cache_.find(key);
        if (it_plan != expert_plan_cache_.end()) {
            assert(it_plan->second.synthetic_wid == synthetic);
            assert(it_plan->second.desired       == desired);
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
        // gets reused. ``*this`` carries the replay-reservation set
        // the tracker consults to skip chunks still in-flight.
        return tracker_.make_room(pool, *rt_, *this);
    }

    void reserve_for_dispatch(const std::string & wid, int cid) {
        tracker_.reserve(wid, cid);
    }
    void release_from_dispatch(const std::string & wid, int cid) {
        tracker_.release(wid, cid);
    }
    void touch_resident(const std::string & wid, int cid, int plane) {
        tracker_.touch(wid, cid, plane);
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
            anybcq::AnyBCQFamilyTensor * tens = rt_->tensor_anybcq(synth);
            if (tens == nullptr) break;
            void ** d_qw    = tens->d_qw_ptrs();
            void ** d_alpha = tens->d_alpha_ptrs();
            void ** d_qbias = tens->d_qbias_slot();
            const UpstreamLayoutHost & host = tens->host();
            if (d_qw == nullptr || d_alpha == nullptr) {
                std::fprintf(stderr,
                    "streamllm-scheduler[moe]: per-plane device pointer "
                    "tables missing for %s\n",
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
            if (host.any_precision) any_any_prec = true;
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

    // ── ggml-cuda hook handlers (Scheduler virtuals) ──────────────
    // Per-node dispatch is routed through the Scheduler base via
    // claims_node + dispatch_node; this scheduler claims managed
    // MUL_MAT / MUL_MAT_ID nodes and dispatches them through the
    // existing per-op implementations in dispatch.cpp.

    bool claims_node(const struct ggml_tensor * node) const override {
        // streamllm always dispatches managed mul_mat / mul_mat_id
        // ops eagerly — the LOAD walk needs full CUDA API access on
        // every invocation, which cuda-graph capture forbids. ggml-
        // cuda consults this predicate per cgraph node and disables
        // capture for any cgraph where it returns true.
        //
        // Benchmark/score-mode opt-in (STREAMLLM_ALLOW_CAPTURE=1):
        // operator has guaranteed full VRAM pin → LOAD is a no-op,
        // capture is safe.  The score-table version hook
        // (qwen3_runtime_glue) invalidates the captured graph when
        // the dial swaps, so phase-aware C_phase works too.
        if (allow_capture_) return false;
        if (node == nullptr) return false;
        if (node->op != GGML_OP_MUL_MAT &&
            node->op != GGML_OP_MUL_MAT_ID) {
            return false;
        }
        const ggml_tensor * w = node->src[0];
        if (w == nullptr || w->name[0] == '\0') return false;
        return managed_names_.count(w->name) != 0;
    }

    bool claims_tensor(const struct ggml_tensor * w) const override {
        // S10 dense-managed clear-fail predicate: distinguish managed
        // canonicals from unmanaged dense weights at the
        // ``streamllm_try_cuda_mul_mat`` boundary so the abort
        // message names the leaked tensor.
        if (w == nullptr || w->name[0] == '\0') return false;
        return managed_names_.count(w->name) != 0;
    }

    // The legacy ``dispatch_node`` and ``observe_topk_moe``
    // overrides were retired in the post-M1 cleanup pass — the
    // base-class virtuals no longer exist.

    // ── Per-replay state (Scheduler virtuals) ─────────────────────
    // Replay reservations protect chunks against eviction across the
    // current cgraph_compute pass. Make_room consults
    // is_replay_reserved before picking a victim.
    void add_replay_reservations(
        const std::vector<ChunkKey> & v) override {
        if (v.empty()) return;
        std::lock_guard<std::mutex> lk(replay_reservations_mu_);
        for (const auto & k : v) {
            replay_reservations_.emplace(
                k.wid + "#" + std::to_string(k.cid));
        }
    }
    bool is_replay_reserved(const std::string & wid,
                             int cid) const override {
        std::lock_guard<std::mutex> lk(replay_reservations_mu_);
        return replay_reservations_.count(
            wid + "#" + std::to_string(cid)) != 0;
    }
    void clear_replay_reservations() override {
        std::lock_guard<std::mutex> lk(replay_reservations_mu_);
        replay_reservations_.clear();
    }

    // ── Graph-compute prewalk ────────────────────────────────────
    // Fired by the ggml-cuda graph_compute_begin hook (added in
    // step 4a). Walks the cgraph, primes the layer-instrumenter's
    // node→layer map, and fires a GraphBegin marker. The dispatch
    // glue subsequently calls scheduler_on_managed_node_visit per
    // managed node; the instrumenter detects layer transitions and
    // fires LayerBegin/LayerEnd markers.
    //
    // Optional diagnostic: set STREAMLLM_DEBUG_GRAPH_WALK=1 to log
    // managed-op counts on each prewalk.
    void on_graph_compute_begin(StreamHandle compute_stream,
                                 const struct ggml_cgraph * cgraph) override {
        if (rt_ == nullptr || cgraph == nullptr) return;

        // Snapshot KBar + version once per cgraph_compute. The version
        // is read by ggml-cuda's graph-cache predicate so a dial swap
        // forces re-capture.
        {
            const float snap = dynamic_kbar_.load(std::memory_order_relaxed);
            const uint64_t version =
                dial_version_.load(std::memory_order_relaxed);
            rt_->set_replay_kbar(snap);
            rt_->set_replay_dial_version(version);
        }

        // S8 (Mode A): cgraph audit. STREAMLLM_CGRAPH_AUDIT=1 walks
        // the cgraph and proves the two post-cutover invariants:
        //   (a) every node named ``"streamllm.moe_layer_<N>"`` is an
        //       op with all four srcs wired (cb() didn't rename it
        //       away — that bug was the cause of the first
        //       end-to-end failure during S5+S6+S7 verification);
        //   (b) zero managed MUL_MAT_ID nodes remain in the cgraph.
        // Violations abort with a concrete report; the audit runs
        // each replay so a regression that introduces managed ops
        // mid-decode is caught immediately.
        if (const char * a = getenv("STREAMLLM_CGRAPH_AUDIT")) {
            if (a[0] && a[0] != '0') {
                const int n_nodes = ggml_graph_n_nodes(
                    const_cast<ggml_cgraph *>(cgraph));
                int n_sentinels      = 0;
                int n_managed_mmid   = 0;
                int n_managed_mm     = 0;
                int n_renamed        = 0;
                for (int i = 0; i < n_nodes; ++i) {
                    const ggml_tensor * node = ggml_graph_node(
                        const_cast<ggml_cgraph *>(cgraph), i);
                    if (node == nullptr) continue;
                    const bool is_sentinel = (std::strncmp(
                        node->name, "streamllm.moe_layer_", 20) == 0);
                    if (is_sentinel) {
                        ++n_sentinels;
                        if (node->src[0] == nullptr ||
                            node->src[1] == nullptr ||
                            node->src[2] == nullptr ||
                            node->src[3] == nullptr) {
                            ++n_renamed; // structurally broken sentinel
                            std::fprintf(stderr,
                                "[streamllm-audit] sentinel '%s' missing "
                                "src[0..3]: [%p, %p, %p, %p]\n",
                                node->name,
                                (void*)node->src[0], (void*)node->src[1],
                                (void*)node->src[2], (void*)node->src[3]);
                        }
                    }
                    if (node->op == GGML_OP_MUL_MAT_ID) {
                        const ggml_tensor * w = node->src[0];
                        if (w && w->name[0] && managed_names_.count(w->name)) {
                            ++n_managed_mmid;
                            std::fprintf(stderr,
                                "[streamllm-audit] managed MUL_MAT_ID "
                                "leaked into cgraph: src0='%s' node='%s'\n",
                                w->name, node->name);
                        }
                    }
                    if (node->op == GGML_OP_MUL_MAT) {
                        const ggml_tensor * w = node->src[0];
                        if (w && w->name[0] && managed_names_.count(w->name)) {
                            ++n_managed_mm;
                            std::fprintf(stderr,
                                "[streamllm-audit] managed MUL_MAT (dense) "
                                "leaked into cgraph: src0='%s' node='%s'\n",
                                w->name, node->name);
                        }
                    }
                }
                std::fprintf(stderr,
                    "[streamllm-audit] nodes=%d sentinels=%d "
                    "managed_mul_mat_id=%d managed_mul_mat=%d "
                    "broken_sentinels=%d\n",
                    n_nodes, n_sentinels,
                    n_managed_mmid, n_managed_mm, n_renamed);
                if (n_managed_mmid > 0 || n_managed_mm > 0 || n_renamed > 0) {
                    GGML_ABORT(
                        "streamllm-ext: cgraph audit failed — "
                        "managed_mul_mat_id=%d managed_mul_mat=%d "
                        "broken_sentinels=%d. Mode A cutover invariants "
                        "violated.",
                        n_managed_mmid, n_managed_mm, n_renamed);
                }
            }
        }

        MarkerEvent ev;
        ev.kind           = MarkerKind::GraphBegin;
        ev.compute_stream = compute_stream;
        on_marker(ev);
    }

    // Symmetric end-of-graph callback. Fires GraphEnd. The
    // qwen3_graph_instrumenter prewalk (S5+) and its per-layer
    // marker emission were retired in S8 — sentinel-based dispatch
    // has no need for the per-canonical layer_begin / layer_end
    // tracking.
    void on_graph_compute_end(StreamHandle compute_stream,
                               const struct ggml_cgraph * /*cgraph*/) override {
        MarkerEvent ev;
        ev.kind           = MarkerKind::GraphEnd;
        ev.compute_stream = compute_stream;
        on_marker(ev);
    }

    // Marker callback. Currently used only for diagnostics (gated by
    // STREAMLLM_DEBUG_MARKERS=1). Future work folds prefetch /
    // residency policy decisions in here once the instrumenter
    // scope expands.
    void on_marker(const MarkerEvent & ev) override {
        const char * dbg = getenv("STREAMLLM_DEBUG_MARKERS");
        if (dbg == nullptr || dbg[0] == '\0' || dbg[0] == '0') return;
        const char * kind_name = "?";
        switch (ev.kind) {
            case MarkerKind::GraphBegin:  kind_name = "GraphBegin";  break;
            case MarkerKind::GraphEnd:    kind_name = "GraphEnd";    break;
            case MarkerKind::LayerBegin:  kind_name = "LayerBegin";  break;
            case MarkerKind::LayerEnd:    kind_name = "LayerEnd";    break;
            case MarkerKind::AttentionIn: kind_name = "AttentionIn"; break;
            case MarkerKind::KvWrite:     kind_name = "KvWrite";     break;
            case MarkerKind::MoeDispatch: kind_name = "MoeDispatch"; break;
            case MarkerKind::Custom:      kind_name = "Custom";      break;
        }
        std::fprintf(stderr,
            "streamllm-marker: kind=%s layer=%d\n",
            kind_name, ev.layer_index);
    }

    uint64_t dial_version() const {
        return dial_version_.load(std::memory_order_relaxed);
    }

    bool          allow_capture()           const { return allow_capture_; }

    // ── Profile-backed per-dispatch K decision ─────────────────────
    // Residual-aware per-dispatch K via the local-FFN-output-L2² gain
    // criterion.  See moe_scheduler.h for the rule + header comment.
    bool set_dynamic_residuals(const std::vector<float> & R_flat,
                                 int n_layers, int n_experts,
                                 int K_min, int K_max) {
        if (n_layers <= 0 || n_experts <= 0) return false;
        const int n_K = K_max - K_min + 1;
        if (n_K <= 0) return false;
        if ((int)R_flat.size() != n_layers * n_experts * n_K) return false;
        {
            std::lock_guard<std::mutex> lk(dial_mu_);
            dynamic_R_ = R_flat;
            dynamic_n_layers_  = n_layers;
            dynamic_n_experts_ = n_experts;
            dynamic_K_min_     = K_min;
            dynamic_K_max_     = K_max;
            // Bump the same version atomic so any captured cgraph
            // invalidates (the dispatch decision encodes into the
            // grid/launch shape).
            dial_version_.fetch_add(1, std::memory_order_relaxed);
            upload_dynamic_R_to_device_unlocked();
        }
        std::fprintf(stderr,
            "streamllm-scheduler[moe]: dynamic_residuals SET — n_layers=%d "
            "n_experts=%d K=[%d, %d]\n",
            n_layers, n_experts, K_min, K_max);
        return true;
    }
    // Upload the host dynamic_R_ table to the device mirror.  Caller
    // must hold dial_mu_.  Re-allocates dynamic_R_d_ when the
    // shape changes (rare — set once at install).
    void upload_dynamic_R_to_device_unlocked() {
        const size_t want = dynamic_R_.size();
        if (want == 0) return;
        if (dynamic_R_d_ != nullptr && dynamic_R_d_count_ != want) {
            cudaFree(dynamic_R_d_);
            dynamic_R_d_ = nullptr;
            dynamic_R_d_count_ = 0;
        }
        if (dynamic_R_d_ == nullptr) {
            const size_t bytes = want * sizeof(float);
            cudaError_t err = cudaMalloc(&dynamic_R_d_, bytes);
            if (err != cudaSuccess) {
                std::fprintf(stderr,
                    "streamllm-scheduler[moe]: cudaMalloc(dynamic_R_d, "
                    "%zu B) failed: %s — falling back to host plan\n",
                    bytes, cudaGetErrorString(err));
                dynamic_R_d_ = nullptr;
                return;
            }
            dynamic_R_d_count_ = want;
        }
        cudaMemcpy(dynamic_R_d_, dynamic_R_.data(),
                    want * sizeof(float), cudaMemcpyHostToDevice);
    }
    const float * dynamic_R_device() const { return dynamic_R_d_; }
    int dynamic_K_min() const { return dynamic_K_min_; }
    int dynamic_K_max() const { return dynamic_K_max_; }
    void clear_dynamic_residuals() {
        std::lock_guard<std::mutex> lk(dial_mu_);
        dynamic_R_.clear();
        dynamic_n_layers_  = 0;
        dynamic_n_experts_ = 0;
        dynamic_K_min_     = 0;
        dynamic_K_max_     = 0;
        dial_version_.fetch_add(1, std::memory_order_relaxed);
    }
    bool has_dynamic_residuals() const {
        return dynamic_n_layers_ > 0 && !dynamic_R_.empty();
    }
    float dynamic_kbar() const {
        return dynamic_kbar_.load(std::memory_order_relaxed);
    }
    qwen3::KBarAllocatorMode kbar_allocator_mode() const {
        return kbar_allocator_mode_;
    }
    void set_dynamic_kbar(float kbar) {
        dynamic_kbar_.store(kbar, std::memory_order_relaxed);
        dial_version_.fetch_add(1, std::memory_order_relaxed);
    }
    bool set_kbar(float kbar, qwen3::KBarAllocatorMode mode) {
        if (!(kbar >= 0.0f)) return false;
        kbar_allocator_mode_ = mode;
        dynamic_kbar_.store(kbar, std::memory_order_relaxed);
        dial_version_.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    // Per-dispatch K̄-budget allocator (PROBLEM.md §5).
    //
    //   n        = experts.size() = number of unique active experts
    //   B_local  = round(n × kbar)   chunks total to spend this dispatch
    //
    // Solves the multi-choice 0/1 knapsack
    //
    //   minimise   Σᵢ gᵢ² · R[L, eᵢ, Kᵢ]
    //   over       Kᵢ ∈ {K_min, …, K_max}    for i = 1..n
    //   s.t.       Σᵢ Kᵢ  =  B_local                (hard equality)
    //
    // via O(n × B × K_range) dynamic programming.  Optimal regardless
    // of whether R[L, e, K] is monotone-non-increasing in K (greedy
    // would lose on non-monotone curves — see commit msg / PROBLEM.md
    // §3.2 caveat).
    //
    // Implementation: two rolling float buffers (dp_prev, dp_cur) +
    // a (n × (B+1)) trace table of int8 K-choices.  All thread-local
    // so no allocation per dispatch after warm-up.
    //
    // Cost analysis at production scale:
    //   decode b=1, Qwen3 (top_k=8, K_range=7, B≤64):
    //     ~8 × 64 × 7 = 3.6 k float ops/dispatch ≈ 2 μs
    //     × 144 dispatches/token = ≈ 300 μs/token
    //     → ~1 % of a 30 t/s decode budget.
    bool allocate_dispatch_budget(int layer,
                                    const std::vector<int>   & experts,
                                    const std::vector<float> & gates,
                                    std::vector<int>         & K_out) const {
        if (dynamic_n_layers_ == 0) return false;
        const float kbar = dynamic_kbar_.load(std::memory_order_relaxed);
        if (!(kbar > 0.0f)) return false;
        const int n = (int) experts.size();
        if (n == 0) return false;
        if ((int) gates.size() != n) return false;
        K_out.assign(n, dynamic_K_min_);
        // Invalid layer → caller falls back to gate-threshold path.
        if (layer < 0 || layer >= dynamic_n_layers_) return false;

        const int K_min = dynamic_K_min_;
        const int K_max = dynamic_K_max_;
        // Guard: K range must be sane before DP tables are allocated.
        if (K_min < 0 || K_max < K_min) {
            GGML_ABORT("streamllm-allocator: invalid K range [%d, %d]",
                       K_min, K_max);
        }
        if (K_max > 255) {
            GGML_ABORT("streamllm-allocator: K_max=%d exceeds uint8_t trace capacity",
                       K_max);
        }
        const int n_K   = K_max - K_min + 1;
        if (n_K <= 1) return true;

        const int Bhi = n * K_max;
        const int B1  = Bhi + 1;

        long t = (long) std::lrint((double) n * (double) kbar);
        if (t < (long) n * K_min) t = (long) n * K_min;
        if (t > (long) n * K_max) t = (long) n * K_max;
        const int target = (int) t;

        // Pre-compute the cost table  C[i, K] = gᵢ² · R[L, eᵢ, K]
        // for K ∈ [K_min, K_max].  Hot-loop access is contiguous over K.
        thread_local std::vector<float> tls_C;
        tls_C.assign((size_t) n * n_K, 0.0f);
        for (int i = 0; i < n; ++i) {
            const int e = experts[i];
            if (e < 0 || e >= dynamic_n_experts_) {
                // Unknown expert — caller should fall back; return false so
                // the gate-threshold path takes over rather than silently
                // treating cost as 0 and biasing the budget.
                return false;
            }
            const size_t base = ((size_t) layer * dynamic_n_experts_
                                  + (size_t) e) * (size_t) n_K;
            const float g  = gates[i];
            const float gg = g * g;
            for (int k = 0; k < n_K; ++k) {
                const size_t pos = base + (size_t) k;
                if (pos >= dynamic_R_.size()) {
                    // R table shape mismatch — same fallback as above.
                    return false;
                }
                tls_C[(size_t) i * n_K + (size_t) k] = gg * dynamic_R_[pos];
            }
        }

        // DP.  dp[b] = min Σ g²·R over first i experts spending b chunks.
        // We use INF = 1e30f as "unreachable".
        constexpr float INF = 1e30f;
        thread_local std::vector<float>   dp_prev, dp_cur;
        thread_local std::vector<uint8_t> trace;   // K choice per (i, b)
        dp_prev.assign((size_t) B1, INF);
        dp_cur .resize((size_t) B1);
        trace  .assign((size_t) n * B1, 0);
        dp_prev[0] = 0.0f;

        for (int i = 0; i < n; ++i) {
            std::fill(dp_cur.begin(), dp_cur.end(), INF);
            const float * Ci = &tls_C[(size_t) i * n_K];
            uint8_t * trace_i = &trace[(size_t) i * B1];
            for (int b_prev = 0; b_prev < B1; ++b_prev) {
                const float dp_p = dp_prev[b_prev];
                if (dp_p >= INF) continue;
                for (int k = 0; k < n_K; ++k) {
                    const int K_val = K_min + k;
                    const int b_new = b_prev + K_val;
                    if (b_new >= B1) break;
                    const float cost = dp_p + Ci[k];
                    if (cost < dp_cur[b_new]) {
                        dp_cur[b_new] = cost;
                        trace_i[b_new] = (uint8_t) K_val;
                    }
                }
            }
            dp_prev.swap(dp_cur);
        }

        const int target_b = target;

        // Backtrack from dp_prev[target_b].
        if (dp_prev[target_b] < INF) {
            int b = target_b;
            for (int i = n - 1; i >= 0; --i) {
                const int K_i = (int) trace[(size_t) i * B1 + (size_t) b];
                K_out[i] = K_i;
                b -= K_i;
            }
        }
        // else: DP couldn't reach target (shouldn't happen given the
        // [n·K_min, n·K_max] clamp). Sanity check below catches it.

        // ── Runtime sanity check — PROBLEM.md §10 invariants ──
        // Hard-enforced because the rest of the dispatch pipeline
        // (kernel grid dims, per-expert chunk pointer tables) assumes
        // these without re-checking; a silent budget violation here
        // produces wrong reconstructions, not a crash.  Cost: one
        // pass + 2 ints — negligible vs the kernel launch that
        // follows.
        long check_sum = 0;
        for (int i = 0; i < n; ++i) {
            const int K_i = K_out[i];
            if (K_i < K_min || K_i > K_max) {
                std::fprintf(stderr,
                    "streamllm-allocator: invariant 2 violated — "
                    "K[%d]=%d outside [%d, %d]\n",
                    i, K_i, K_min, K_max);
                GGML_ABORT("streamllm-allocator: per-expert K out of range");
            }
            check_sum += K_i;
        }
        // DP commits to the picked budget cell — hard equality.
        // In K̄-mode this equals the configured target; in ε-mode it
        // is the smallest reachable B satisfying Σ g²·R[K] ≤ ε.
        if (check_sum != (long) target_b) {
            std::fprintf(stderr,
                "streamllm-allocator: invariant 1 violated — "
                "sum K[e]=%ld, expected target_b=%d "
                "(mode=kbar, KBar=%g n=%d)\n",
                check_sum, target_b, (double) kbar, n);
            GGML_ABORT("streamllm-allocator: budget not enforced");
        }

        // ── Optional per-layer first-dispatch dump for diagnosis.
        // STREAMLLM_LOG_ALLOC=1 emits one log line per (layer × first
        // dispatch encountered) with the gate distribution, the
        // chosen K[], the greedy total cost Σ g²·R[K], and the
        // uniform-baseline cost Σ g²·R[K̄] for comparison.  Used to
        // pinpoint where greedy diverges from uniform.
        static std::atomic<bool> alloc_log_layer_seen[256] = {};
        const char * log_env = std::getenv("STREAMLLM_LOG_ALLOC");
        if (log_env && log_env[0] && log_env[0] != '0' &&
            layer >= 0 && layer < 256 &&
            !alloc_log_layer_seen[layer].exchange(true)) {
            // Sort experts by gate desc for readability.
            std::vector<int> order(n);
            for (int i = 0; i < n; ++i) order[i] = i;
            std::sort(order.begin(), order.end(),
                       [&](int a, int b) { return gates[a] > gates[b]; });
            // Uniform-K reference for the diagnostic ratio.  Pick the
            // K that matches the current effective K̄ so the comparison
            // is fair across modes.
            const double Kbar_eff = (double) target_b / (double) n;
            const int K_unif = (int) std::lrint(Kbar_eff);
            const int K_unif_c =
                std::max(K_min, std::min(K_max, K_unif));
            double err_dp = 0.0, err_uniform = 0.0;
            for (int i = 0; i < n; ++i) {
                const int e = experts[i];
                if (e < 0 || e >= dynamic_n_experts_) continue;
                const size_t base = ((size_t) layer * dynamic_n_experts_
                                       + (size_t) e) * (size_t) n_K;
                const float g = gates[i];
                const float gg = g * g;
                const int idx_g = K_out[i]   - K_min;
                const int idx_u = K_unif_c   - K_min;
                if (base + (size_t) idx_g < dynamic_R_.size())
                    err_dp      += gg * (double) dynamic_R_[base + (size_t) idx_g];
                if (base + (size_t) idx_u < dynamic_R_.size())
                    err_uniform += gg * (double) dynamic_R_[base + (size_t) idx_u];
            }
            std::fprintf(stderr,
                "streamllm-alloc L=%d n=%d mode=kbar  KBar_eff=%.3f  B=%d  "
                "err_dp=%.6g  err_uniform(K=%d)=%.6g  ratio=%.3f\n",
                layer, n, Kbar_eff, target_b,
                err_dp, K_unif_c, err_uniform,
                err_uniform > 0 ? err_dp / err_uniform : 0.0);
            std::fprintf(stderr, "  experts↓gate ");
            for (int j = 0; j < n && j < 16; ++j) {
                const int i = order[j];
                std::fprintf(stderr, " e%d:g=%.3f→K=%d", experts[i], gates[i], K_out[i]);
            }
            std::fprintf(stderr, "\n");
        }
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

    // Score-threshold table.  Length n_score_tiers_ = max n_chunks
    // across managed tensors; score_thresholds_[k] is the lower-edge
    mutable std::mutex    dial_mu_;
    std::atomic<uint64_t> dial_version_{0};

    // Profile residuals loaded from STREAMLLM_KBAR_PROFILE_FILE.
    std::vector<float>   dynamic_R_;
    int                  dynamic_n_layers_  = 0;
    int                  dynamic_n_experts_ = 0;
    int                  dynamic_K_min_     = 0;
    int                  dynamic_K_max_     = 0;
    std::atomic<float>   dynamic_kbar_{0.0f};
    qwen3::KBarAllocatorMode    kbar_allocator_mode_{qwen3::KBarAllocatorMode::Profile};

    // STREAMLLM_ALLOW_CAPTURE=1: opt-in to CUDA-graph capture for the
    // managed cgraph. Only legal when the operator has set
    // STREAMLLM_VRAM_CAP_MB large enough to keep every chunk pinned.
    // The fully-pinned check at the end of on_install aborts loudly
    // when the assumption is broken.
    bool allow_capture_ = false;
    // Device mirror of dynamic_R_ for KBar planners.
    float *               dynamic_R_d_ = nullptr;
    size_t                dynamic_R_d_count_ = 0;

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
    bool                        host_cache_enabled_ = true;
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

    // Per-(synthetic_wid, desired_chunks) plan cache.
    //
    // History: this map was previously keyed on a uint64_t built by
    // XOR-folding the address of P_of_[synthetic] with the desired
    // count.  Two distinct (synthetic, desired) pairs can land on the
    // same uint64_t, and ``emplace(key, …)`` is a no-op on key
    // collision — so the function would silently return the colliding
    // entry's plan (built for a different expert), giving the
    // dispatch a load_set that doesn't match its required_set.  That
    // was the root cause of the T5 residency violations after the
    // chunk/plane refactor.  The cache now uses a string key
    // (synthetic + "#" + desired) so collisions cannot occur.
    struct CachedExpertPlan {
        std::string synthetic_wid;
        int         desired;
        Plan        plan;
    };
    std::unordered_map<std::string, CachedExpertPlan> expert_plan_cache_;

    // Per-canonical-wid expert table for the fused MoE kernel. Built
    // lazily on first moe_expert_table() call; cached thereafter.
    std::unordered_map<std::string, MoeExpertTable> moe_tables_;
    std::mutex moe_tables_mu_;

    // Tensor-name set populated at on_install — every managed
    // tensor (chunked synthetic wids AND placeholder-only
    // canonicals).  Read by claims_tensor / claims_node /
    // dispatch_node.  Replaces the moved-off `is_managed_name`
    // surface that used to live on StreamllmRuntime.
    std::unordered_set<std::string> managed_names_;

    // Replay-scoped chunk reservations. Populated by the dispatch
    // path as it commits to loading chunks; consulted by make_room
    // via the tracker's is_replay_reserved check; cleared at
    // graph_compute_end.  Belongs to the scheduler (not the
    // runtime) because the routing-decision semantics are MoE-
    // specific.
    std::unordered_set<std::string>          replay_reservations_;
    mutable std::mutex                       replay_reservations_mu_;

    // Per-canonical MoEMatMulComp.  Built eagerly at install once the
    // per-canonical MoeExpertTable + e0 device layout are known.  The
    // dispatch shim calls ``scheduler_lookup_moe_comp(canonical)`` to
    // hand off to ``StreamllmRuntime::run``.  Owned by the scheduler;
    // destroyed in the dtor (after the runtime, since comps hold pinned
    // buffers freed via cudaFreeHost).
    std::unordered_map<std::string, std::unique_ptr<qwen3::MoEMatMulComp>>
        moe_comps_;

    // S5+S6+S7+S8 (Mode A): the per-canonical graph instrumenter was
    // retired alongside the legacy MUL_MAT_ID dispatch surface. The
    // sentinel rail carries its own per-layer identity via the
    // ``"streamllm.moe_layer_<L>"`` name; the cgraph audit
    // (STREAMLLM_CGRAPH_AUDIT=1) is the only walk that still touches
    // every node.
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

// ─── Free-function entry points (scheduler.h) ───────────
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

const Plan * scheduler_plan_for_expert_with_chunks(
    Scheduler &         sched,
    const std::string & canonical_wid,
    int                 expert_id,
    int                 n_chunks_requested,
    StreamHandle        compute_stream)
{
    return as_moe(sched).plan_for_expert_with_chunks(
        canonical_wid, expert_id, n_chunks_requested, compute_stream);
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

void scheduler_touch_resident(
    Scheduler &         sched,
    const std::string & wid,
    int                 cid,
    int                 plane)
{
    as_moe(sched).touch_resident(wid, cid, plane);
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

// scheduler_on_managed_node_visit was retired with the
// qwen3_graph_instrumenter prewalk; sentinel naming carries the
// per-layer index directly.

float scheduler_kbar(const Scheduler & sched) {
    return static_cast<const MoEScheduler &>(sched).dynamic_kbar();
}

KBarAllocatorMode scheduler_kbar_allocator_mode(const Scheduler & sched) {
    return static_cast<const MoEScheduler &>(sched).kbar_allocator_mode();
}

bool scheduler_set_kbar(Scheduler & sched, float kbar, KBarAllocatorMode mode) {
    return as_moe(sched).set_kbar(kbar, mode);
}

uint64_t scheduler_dial_version(const Scheduler & sched) {
    return static_cast<const MoEScheduler &>(sched).dial_version();
}

bool scheduler_allow_capture(const Scheduler & sched) {
    return static_cast<const MoEScheduler &>(sched).allow_capture();
}

// ─── Dynamic per-dispatch residuals (rung 2) ───────────────────────
bool scheduler_set_dynamic_residuals(
    Scheduler &                sched,
    const std::vector<float> & R_flat,
    int                        n_layers,
    int                        n_experts,
    int                        K_min,
    int                        K_max)
{
    return as_moe(sched).set_dynamic_residuals(
        R_flat, n_layers, n_experts, K_min, K_max);
}
void scheduler_clear_dynamic_residuals(Scheduler & sched) {
    as_moe(sched).clear_dynamic_residuals();
}
bool scheduler_has_dynamic_residuals(const Scheduler & sched) {
    return static_cast<const MoEScheduler &>(sched).has_dynamic_residuals();
}
bool scheduler_allocate_dispatch_budget(const Scheduler &        sched,
                                          int                      layer,
                                          const std::vector<int>   & experts,
                                          const std::vector<float> & gates,
                                          std::vector<int>         & K_out) {
    return static_cast<const MoEScheduler &>(sched)
        .allocate_dispatch_budget(layer, experts, gates, K_out);
}
const float * scheduler_dynamic_R_device(const Scheduler & sched) {
    return static_cast<const MoEScheduler &>(sched).dynamic_R_device();
}
int scheduler_dynamic_K_min(const Scheduler & sched) {
    return static_cast<const MoEScheduler &>(sched).dynamic_K_min();
}
int scheduler_dynamic_K_max(const Scheduler & sched) {
    return static_cast<const MoEScheduler &>(sched).dynamic_K_max();
}

MoEMatMulComp * scheduler_lookup_moe_comp(
    Scheduler &         sched,
    const std::string & canonical_wid)
{
    return as_moe(sched).lookup_moe_comp(canonical_wid);
}

}  // namespace qwen3

} // namespace streamllm_ext
