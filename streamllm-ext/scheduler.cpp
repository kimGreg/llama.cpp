// streamllm-ext — scheduler implementations.
//
// Each scheduler is a thin policy over the runtime's two primitives:
//
//   rt.move_chunk(wid, cid, src, dst)     — async H2D
//   rt.chunk_matmul(wid, chunks, X)       — launch nqmv_bias
//
// on_install() builds the host-side upstream layout for every managed
// tensor, registers it with the runtime, and optionally pre-uploads to
// VRAM. plan() returns {chunks, moves} the hook applies.

#include "scheduler.h"
#include "runtime.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

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

std::vector<uint8_t> pread_range(const std::string & path,
                                 int64_t offset, int64_t nbytes) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("scheduler: cannot open " + path);
    f.seekg(offset);
    std::vector<uint8_t> buf((size_t)nbytes);
    f.read((char *)buf.data(), nbytes);
    if (f.gcount() != nbytes) {
        throw std::runtime_error(
            "scheduler: short read on " + path +
            " (wanted " + std::to_string(nbytes) +
            ", got " + std::to_string(f.gcount()) + ")");
    }
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

void upload_and_pin_all(StreamllmRuntime & rt, const std::string & name, int P) {
    // Eager / Lazy / Prefetch keep every chunk resident for the run —
    // upload then pin explicitly (move_chunk itself no longer auto-pins).
    rt.move_chunk(name, kCidQBias, Tier::RAM, Tier::VRAM);
    rt.pool().pin(name, kCidQBias);
    for (int p = 0; p < P; ++p) {
        rt.move_chunk(name, cid_chunk(p), Tier::RAM, Tier::VRAM);
        rt.pool().pin(name, cid_chunk(p));
    }
}

// Extract the integer layer index from a tensor name. Works for:
//   * Qwen3 dense:   "blk.<N>.<kind>.weight"
//   * Qwen3-MoE:     "blk.<N>.ffn_{up,gate,down}_exps.<X>.weight"
// We scan digits immediately after the first "blk." and stop, so the
// outer block index N is returned regardless of what follows — the
// expert index X in the MoE case is correctly ignored. Returns -1 for
// non-layer tensors (token_embd, output_norm, ...).
int parse_layer(const std::string & name) {
    const auto pos = name.find("blk.");
    if (pos == std::string::npos) return -1;
    size_t i = pos + 4;
    int n = 0;
    bool any = false;
    while (i < name.size() && std::isdigit((unsigned char)name[i])) {
        n = n * 10 + (name[i] - '0');
        any = true;
        ++i;
    }
    return any ? n : -1;
}


// ---- Eager -----------------------------------------------------------

class EagerScheduler : public Scheduler {
public:
    void on_install(StreamllmRuntime & rt,
                    const StreamReader & reader,
                    const std::string & gguf_path) override {
        for (const auto & name : reader.managed_tensor_names()) {
            UpstreamLayoutHost host = read_one(reader, gguf_path, name);
            int P = host.n_chunks;
            rt.register_layout(name, std::move(host), UpstreamLayoutDevice{});
            upload_and_pin_all(rt, name, P);
            plans_.emplace(name, Plan{all_chunks_for(P), {}});
        }
        std::fprintf(stderr,
            "streamllm-scheduler[eager]: pre-loaded %zu managed tensors\n",
            plans_.size());
    }

    const Plan * plan(const std::string & tensor_name,
                      StreamHandle /*compute_stream*/) override {
        auto it = plans_.find(tensor_name);
        if (it == plans_.end()) return nullptr;
        return &it->second;
    }

    const char * name() const override { return "eager"; }

private:
    std::unordered_map<std::string, Plan> plans_;
};


// ---- Lazy ------------------------------------------------------------

class LazyScheduler : public Scheduler {
public:
    void on_install(StreamllmRuntime & rt,
                    const StreamReader & reader,
                    const std::string & gguf_path) override {
        rt_ = &rt;
        for (const auto & name : reader.managed_tensor_names()) {
            UpstreamLayoutHost host = read_one(reader, gguf_path, name);
            int P = host.n_chunks;
            rt.register_layout(name, std::move(host), UpstreamLayoutDevice{});
            // First-touch plan carries the full move list; steady-state
            // plan shares the same chunks but with no moves.
            first_touch_plans_.emplace(name, Plan{
                all_chunks_for(P), all_moves_for(name, P)
            });
            steady_plans_.emplace(name, Plan{all_chunks_for(P), {}});
        }
        std::fprintf(stderr,
            "streamllm-scheduler[lazy]: parsed %zu host layouts "
            "(no VRAM uploads yet)\n",
            first_touch_plans_.size());
    }

    const Plan * plan(const std::string & tensor_name,
                      StreamHandle /*compute_stream*/) override {
        // first_touched_ tracks which wids have had their moves
        // returned already. Plans themselves live in steady_plans_ for
        // the whole process lifetime; first_touch_plans_ just carries
        // the move list we want to attach on the first visit.
        auto st = steady_plans_.find(tensor_name);
        if (st == steady_plans_.end()) return nullptr;
        if (first_touched_.insert(tensor_name).second) {
            auto ft = first_touch_plans_.find(tensor_name);
            if (ft == first_touch_plans_.end()) return &st->second;
            ++n_uploaded_;
            if (n_uploaded_ <= 4 || n_uploaded_ % 32 == 0) {
                std::fprintf(stderr,
                    "streamllm-scheduler[lazy]: uploading %s "
                    "(%zu / %zu managed)\n",
                    tensor_name.c_str(), n_uploaded_,
                    first_touch_plans_.size());
            }
            return &ft->second;
        }
        return &st->second;
    }


    const char * name() const override { return "lazy"; }

private:
    StreamllmRuntime * rt_ = nullptr;
    std::unordered_map<std::string, Plan> first_touch_plans_;
    std::unordered_map<std::string, Plan> steady_plans_;
    std::unordered_set<std::string>       first_touched_;
    size_t n_uploaded_ = 0;
};


// ---- LayerPrefetch ---------------------------------------------------
//
// For each mul_mat, fire own-layer moves (if not resident) and also
// prefetch moves for the weights in layer N+lookahead. Moves run on
// the pool's copy stream, so the kernel's ``wait_on_stream`` only
// blocks if the H2D hasn't completed yet — i.e., copy and compute
// overlap.
//
// Lookahead is tunable via STREAMLLM_LOOKAHEAD (default 1).

class LayerPrefetchScheduler : public Scheduler {
public:
    void on_install(StreamllmRuntime & rt,
                    const StreamReader & reader,
                    const std::string & gguf_path) override {
        rt_ = &rt;
        if (const char * s = std::getenv("STREAMLLM_LOOKAHEAD")) {
            lookahead_ = std::max(0, std::atoi(s));
        }

        int max_layer = -1;
        for (const auto & name : reader.managed_tensor_names()) {
            UpstreamLayoutHost host = read_one(reader, gguf_path, name);
            int P = host.n_chunks;
            P_of_.emplace(name, P);
            rt.register_layout(name, std::move(host), UpstreamLayoutDevice{});
            int l = parse_layer(name);
            layer_of_.emplace(name, l);
            if (l > max_layer) max_layer = l;
            plans_.emplace(name, Plan{all_chunks_for(P), {}});
        }
        if (max_layer >= 0) {
            layer_tensors_.resize(max_layer + 1);
            for (const auto & [name, l] : layer_of_) {
                if (l >= 0 && l < (int)layer_tensors_.size()) {
                    layer_tensors_[l].push_back(name);
                }
            }
            for (auto & v : layer_tensors_) std::sort(v.begin(), v.end());
        }
        std::fprintf(stderr,
            "streamllm-scheduler[prefetch]: %zu layers indexed, "
            "lookahead=%d (no uploads at install)\n",
            layer_tensors_.size(), lookahead_);
    }

    const Plan * plan(const std::string & tensor_name,
                      StreamHandle /*compute_stream*/) override {
        auto it = plans_.find(tensor_name);
        if (it == plans_.end()) return nullptr;
        Plan & p = it->second;
        p.moves.clear();

        auto append_moves_for = [&](const std::string & t) {
            auto pit = P_of_.find(t);
            if (pit == P_of_.end()) return;
            int P = pit->second;
            // "Resident" probe uses q-plane 0 + q-bias as a cheap proxy
            // for "full set already in flight / uploaded".
            if (rt_->pool().is_resident(t, cid_chunk(0)) &&
                rt_->pool().is_resident(t, kCidQBias)) {
                return;
            }
            auto mv = all_moves_for(t, P);
            p.moves.insert(p.moves.end(), mv.begin(), mv.end());
        };

        auto ensure_layer = [&](int L) {
            if (L < 0 || L >= (int)layer_tensors_.size()) return;
            for (const auto & t : layer_tensors_[L]) append_moves_for(t);
        };

        int L = -1;
        auto lit = layer_of_.find(tensor_name);
        if (lit != layer_of_.end()) L = lit->second;

        append_moves_for(tensor_name);
        if (L >= 0 && lookahead_ > 0) {
            ensure_layer(L + lookahead_);
        }

        ++n_plans_;
        return &p;
    }

    const char * name() const override { return "prefetch"; }

private:
    StreamllmRuntime * rt_ = nullptr;
    int lookahead_ = 1;
    std::unordered_map<std::string, int> layer_of_;
    std::unordered_map<std::string, int> P_of_;
    std::vector<std::vector<std::string>> layer_tensors_;
    std::unordered_map<std::string, Plan> plans_;
    size_t n_plans_ = 0;
};


// ---- Budgeted -------------------------------------------------------
//
// Hybrid streaming scheduler. Sits between eager (A) and pure
// prefetch (B): a **prefix** of the most-significant bit planes is
// pinned resident (zero H2D per forward after warmup), and a **tail**
// of the remaining planes is streamed in on each forward and flushed
// via LRU when the next tensor's tail needs room.
//
//   STREAMLLM_VRAM_CAP_MB       hard cap on pool footprint (MB)
//   STREAMLLM_CHUNKS          total planes per weight (default P_max)
//   STREAMLLM_PINNED_CHUNKS        # of prefix planes that stay pinned;
//                               in [0, INIT_BPW]. Default = INIT_BPW
//                               (behaves like eager at init_bpw).
//   STREAMLLM_EVICT_UNPINNED        1 = evict tail chunks immediately
//                               after chunk_matmul (true streaming,
//                               peak VRAM = prefix + 1 tensor's tail).
//                               0 = leave tail resident and rely on
//                               LRU/cap for eviction (cache-y mode).
//                               Default 1 — the "tail flushed right
//                               after use" Scenario-C semantic.
//   STREAMLLM_TPS_FLOOR         tok/s floor; drops one tail plane per
//                               violation (never encroaches on prefix).
//   STREAMLLM_TPS_WINDOW        smoothing window in steps (default 8)
//   STREAMLLM_LOOKAHEAD        # of mul_mats ahead to prefetch tail
//                               chunks for. Default 0 (no lookahead,
//                               serial H2D on the critical path).
//                               Set to 1-4 on async-capable pools to
//                               overlap tail-H2D with compute. Memory
//                               cost: pool must hold (L+1) tails
//                               concurrently instead of 1.
//   STREAMLLM_FRONT_LAYERS      N ≥ 0. Number of leading decoder
//                               layers that get FRONT_{BPW,PREFIX}
//                               instead of the uniform values.
//                               Default 0 (disabled — all layers use
//                               the uniform BPW/PREFIX).
//   STREAMLLM_MID_LAYERS        Size of the middle region, starting
//                               at FRONT_LAYERS. Default 0 (skip —
//                               degenerates to a 2-way split).
//   STREAMLLM_FRONT_CHUNKS         Kernel precision (planes consumed) for
//                               layers [0, FRONT_LAYERS). Default INIT_BPW.
//   STREAMLLM_MID_CHUNKS           Kernel precision for layers
//                               [FRONT_LAYERS, FRONT_LAYERS+MID_LAYERS).
//   STREAMLLM_REAR_CHUNKS          Kernel precision for the remaining
//                               (rear) layers.
//   STREAMLLM_FRONT_PINNED,
//   STREAMLLM_MID_PINNED,
//   STREAMLLM_REAR_PINNED       Prefix planes for each region;
//                               default to the region's BPW (eager).
//
// Iso-budget example (Qwen3-4B, 36 layers, uniform prefix=4 baseline
// pins 144 layer-planes). 3-way split keeping that budget:
//   FRONT_LAYERS=12 FRONT_BPW=FRONT_PREFIX=5
//   MID_LAYERS=12   MID_BPW=MID_PREFIX=4
//                   REAR_BPW=REAR_PREFIX=3       (last 12 layers)
// → 12·5 + 12·4 + 12·3 = 144 pinned. Same memory, same loading,
// different per-layer kernel precision.
//
// Per-tensor plan: {q_bias, plane_0..plane_{init_bpw-1}}. The first
// ``prefix_bpw`` entries plus q_bias are pinned on first touch. The
// tail is unpinned; with EVICT_TAIL=1 it is explicitly evicted in
// ``after_compute`` so the pool slot is freed as soon as the kernel
// finishes reading it.
//
// Lookahead (PREFETCH_L > 0) records the tensor traversal order during
// step 0, then from step 1 onward each plan() call emits tail-H2D
// moves for the next L tensors in the order. These moves run on the
// copy stream while the current tensor's kernel executes on the
// compute stream — that's the overlap the pool's event machinery
// was built for.
//
// Empirical caveat (4B on RTX 4070 Ti Super, decode tg64, prefix=0):
// lookahead = 0 gives 21.2 t/s; lookahead = 1 → 19.4 t/s (slower!).
// Reason: decode is already CPU-dispatch-bound, not copy-stream-bound.
// The copy stream runs at ~1.5 GB/s out of ~18 GB/s ceiling — plenty
// of headroom. Adding more move_chunk calls per plan() adds mutex +
// cudaMemcpyAsync + event-create calls that grow the CPU overhead
// faster than they save copy-stream waits. Prefill regimes and slow
// storage tiers (microSD, SSD) are the intended targets; default
// stays 0.

class BudgetedScheduler : public Scheduler {
public:
    void on_install(StreamllmRuntime & rt,
                    const StreamReader & reader,
                    const std::string & gguf_path) override {
        rt_ = &rt;

        if (const char * s = std::getenv("STREAMLLM_CHUNKS"))
            desired_n_chunks_ = std::max(1, std::atoi(s));
        if (const char * s = std::getenv("STREAMLLM_PINNED_CHUNKS"))
            n_pinned_ = std::max(0, std::atoi(s));
        else
            n_pinned_ = -1;  // defer: will equal desired_n_chunks_ after clamp
        if (const char * s = std::getenv("STREAMLLM_EVICT_UNPINNED"))
            evict_tail_ = std::atoi(s) != 0;
        if (const char * s = std::getenv("STREAMLLM_TPS_FLOOR"))
            tps_floor_ = std::max(0.0, std::atof(s));
        if (const char * s = std::getenv("STREAMLLM_TPS_WINDOW"))
            tps_window_ = std::max(1, std::atoi(s));
        if (const char * s = std::getenv("STREAMLLM_LOOKAHEAD"))
            lookahead_ = std::max(0, std::atoi(s));
        if (const char * s = std::getenv("STREAMLLM_FRONT_LAYERS"))
            front_layers_ = std::max(0, std::atoi(s));
        if (const char * s = std::getenv("STREAMLLM_MID_LAYERS"))
            mid_layers_   = std::max(0, std::atoi(s));
        if (const char * s = std::getenv("STREAMLLM_FRONT_CHUNKS"))
            front_bpw_    = std::max(1, std::atoi(s));
        if (const char * s = std::getenv("STREAMLLM_MID_CHUNKS"))
            mid_bpw_      = std::max(1, std::atoi(s));
        if (const char * s = std::getenv("STREAMLLM_REAR_CHUNKS"))
            rear_bpw_     = std::max(1, std::atoi(s));
        if (const char * s = std::getenv("STREAMLLM_FRONT_PINNED"))
            front_prefix_ = std::max(0, std::atoi(s));
        if (const char * s = std::getenv("STREAMLLM_MID_PINNED"))
            mid_prefix_   = std::max(0, std::atoi(s));
        if (const char * s = std::getenv("STREAMLLM_REAR_PINNED"))
            rear_prefix_  = std::max(0, std::atoi(s));

        int max_P = 0;
        for (const auto & name : reader.managed_tensor_names()) {
            UpstreamLayoutHost host = read_one(reader, gguf_path, name);
            int P = host.n_chunks;
            P_of_.emplace(name, P);
            if (P > max_P) max_P = P;
            rt.register_layout(name, std::move(host), UpstreamLayoutDevice{});
            plans_.emplace(name, Plan{});
            layer_of_.emplace(name, parse_layer(name));
        }
        if (desired_n_chunks_ > max_P) desired_n_chunks_ = max_P;
        max_P_ = max_P;
        if (n_pinned_ < 0) n_pinned_ = desired_n_chunks_;  // default: all pinned
        n_pinned_ = std::min(n_pinned_, desired_n_chunks_);

        // Region defaults: uniform (= desired_n_chunks_ / n_pinned_) if
        // not explicitly set. Prefix defaults to the region's bpw
        // (eager within each region).
        if (front_bpw_    < 0) front_bpw_    = desired_n_chunks_;
        if (mid_bpw_      < 0) mid_bpw_      = desired_n_chunks_;
        if (rear_bpw_     < 0) rear_bpw_     = desired_n_chunks_;
        if (front_prefix_ < 0) front_prefix_ = front_bpw_;
        if (mid_prefix_   < 0) mid_prefix_   = mid_bpw_;
        if (rear_prefix_  < 0) rear_prefix_  = rear_bpw_;
        // Clamp each to sane ranges.
        front_bpw_    = std::min(std::max(front_bpw_, 1), max_P_);
        mid_bpw_      = std::min(std::max(mid_bpw_,   1), max_P_);
        rear_bpw_     = std::min(std::max(rear_bpw_,  1), max_P_);
        front_prefix_ = std::min(std::max(front_prefix_, 0), front_bpw_);
        mid_prefix_   = std::min(std::max(mid_prefix_,   0), mid_bpw_);
        rear_prefix_  = std::min(std::max(rear_prefix_,  0), rear_bpw_);

        std::fprintf(stderr,
            "streamllm-scheduler[budgeted]: %zu tensors, init_bpw=%d/%d "
            "prefix_bpw=%d tail_bpw=%d evict_tail=%d vram_cap=%.1f MB "
            "tps_floor=%.1f window=%d lookahead=%d "
            "front=[%d layers, bpw=%d pfx=%d] "
            "mid=[%d layers, bpw=%d pfx=%d] "
            "rear=[bpw=%d pfx=%d]\n",
            plans_.size(), desired_n_chunks_, max_P_,
            n_pinned_, desired_n_chunks_ - n_pinned_, evict_tail_ ? 1 : 0,
            (double)rt.pool().capacity_bytes() / 1024.0 / 1024.0,
            tps_floor_, tps_window_, lookahead_,
            front_layers_, front_bpw_, front_prefix_,
            mid_layers_,   mid_bpw_,   mid_prefix_,
            rear_bpw_,     rear_prefix_);
    }

    // Pick (bpw, prefix) for a tensor based on its layer index.
    // Returns the uniform values when no region split is active.
    // Regions:
    //   [0, front_layers_)                             → front_*
    //   [front_layers_, front_layers_ + mid_layers_)   → mid_*
    //   [front_layers_ + mid_layers_, ...)             → rear_*
    struct Slice { int bpw; int prefix; };
    Slice slice_for_(const std::string & tensor_name) const {
        if (front_layers_ <= 0 && mid_layers_ <= 0)
            return {desired_n_chunks_, n_pinned_};
        auto it = layer_of_.find(tensor_name);
        const int L = (it != layer_of_.end()) ? it->second : -1;
        if (L < 0)                            return {desired_n_chunks_, n_pinned_};
        if (L < front_layers_)                return {front_bpw_, front_prefix_};
        if (L < front_layers_ + mid_layers_)  return {mid_bpw_,   mid_prefix_};
        return {rear_bpw_, rear_prefix_};
    }

    const Plan * plan(const std::string & tensor_name,
                      StreamHandle /*compute_stream*/) override {
        // Adaptive precision: once a "step" has been observed (all 196
        // tensors seen once), we check whether the last step exceeded
        // the deadline and drop a plane if so.
        maybe_tick_step(tensor_name);

        auto it = plans_.find(tensor_name);
        if (it == plans_.end()) return nullptr;
        Plan & p = it->second;
        p.chunks.clear();
        p.moves.clear();

        int P_here   = P_of_[tensor_name];
        Slice sl     = slice_for_(tensor_name);
        int bpw      = std::min(sl.bpw,    P_here);
        int prefix   = std::min(sl.prefix, bpw);

        auto maybe_move_and_pin = [&](int cid) {
            if (!rt_->pool().is_resident(tensor_name, cid)) {
                p.moves.push_back(
                    {tensor_name, cid, Tier::RAM, Tier::VRAM});
                // The hook will call move_chunk; we need the chunk
                // pinned after upload. Pin here — pin() on a non-
                // resident chunk just records the intent; the pool
                // checks the flag when considering LRU eviction, so
                // this is safe to call before the H2D completes.
                rt_->pool().pin(tensor_name, cid);
            }
        };
        auto maybe_move_unpinned = [&](int cid) {
            if (!rt_->pool().is_resident(tensor_name, cid)) {
                p.moves.push_back(
                    {tensor_name, cid, Tier::RAM, Tier::VRAM});
            }
        };

        // q_bias + prefix planes are pinned on first touch so they
        // stay resident forever (zero H2D cost after warmup).
        p.chunks.push_back(kCidQBias);
        maybe_move_and_pin(kCidQBias);
        for (int pi = 0; pi < prefix; ++pi) {
            p.chunks.push_back(cid_chunk(pi));
            maybe_move_and_pin(cid_chunk(pi));
        }
        // Tail planes are unpinned: they're re-uploaded as needed and
        // the pool's LRU evicts the oldest tail chunks to make room
        // for incoming ones. Under a tight cap this is the "fetch +
        // flush per forward" behaviour B also exhibits.
        for (int pi = prefix; pi < bpw; ++pi) {
            p.chunks.push_back(cid_chunk(pi));
            maybe_move_unpinned(cid_chunk(pi));
        }

        // Record tensor traversal order during the warm-up step, so
        // subsequent steps know which mul_mats are coming next.
        if (step_idx_ == 0 &&
            step_pos_.find(tensor_name) == step_pos_.end()) {
            step_pos_[tensor_name] = (int)step_order_.size();
            step_order_.push_back(tensor_name);
        }

        // Lookahead prefetch: after the warm-up step, emit tail-H2D
        // moves for the next L tensors in the recorded order. These
        // ride the copy stream alongside the current tensor's own
        // tail move (if any), waiting for the previous compute event
        // — so they run in parallel with this tensor's chunk_matmul
        // rather than serialising behind it.
        if (step_idx_ >= 1 && lookahead_ > 0 && !step_order_.empty()) {
            auto cur_it = step_pos_.find(tensor_name);
            if (cur_it != step_pos_.end()) {
                const int pos = cur_it->second;
                const int n   = (int)step_order_.size();
                for (int k = 1; k <= lookahead_; ++k) {
                    const std::string & future = step_order_[(pos + k) % n];
                    auto pit = P_of_.find(future);
                    if (pit == P_of_.end()) continue;
                    const int fP      = pit->second;
                    const Slice fsl   = slice_for_(future);
                    const int fbpw    = std::min(fsl.bpw,    fP);
                    const int fprefix = std::min(fsl.prefix, fbpw);
                    // Only tail planes — prefix is already pinned
                    // after step 0 and so always resident.
                    for (int pi = fprefix; pi < fbpw; ++pi) {
                        const int cid = cid_chunk(pi);
                        if (!rt_->pool().is_resident(future, cid)) {
                            p.moves.push_back(
                                {future, cid, Tier::RAM, Tier::VRAM});
                        }
                    }
                }
            }
        }

        return &p;
    }

    void after_compute(const std::string & tensor_name,
                       StreamHandle /*compute_stream*/) override {
        if (!evict_tail_) return;
        int P_here  = P_of_[tensor_name];
        Slice sl    = slice_for_(tensor_name);
        int bpw     = std::min(sl.bpw,    P_here);
        int prefix  = std::min(sl.prefix, bpw);
        // Evict only the tail planes — prefix stays pinned, q_bias stays.
        for (int pi = prefix; pi < bpw; ++pi) {
            rt_->pool().evict(tensor_name, cid_chunk(pi));
        }
    }

    const char * name() const override { return "budgeted"; }

private:
    void maybe_tick_step(const std::string & tensor_name) {
        // Treat the first managed tensor we see repeatedly as the
        // step boundary. ``first_seen_`` is the wid of that tensor.
        if (first_seen_.empty()) {
            first_seen_ = tensor_name;
            last_step_start_ = std::chrono::steady_clock::now();
            return;
        }
        if (tensor_name != first_seen_) return;

        auto now = std::chrono::steady_clock::now();
        double dt = std::chrono::duration<double>(
            now - last_step_start_).count();
        last_step_start_ = now;
        ++step_idx_;
        if (step_idx_ <= 1) return;  // first step is warmup

        // Rolling window of step durations.
        step_times_.push_back(dt);
        if ((int)step_times_.size() > tps_window_) step_times_.erase(step_times_.begin());

        // After the first few forwards, the Budgeted scheduler has
        // settled: VRAM cap engaged, planes cycled in/out. Now apply
        // the tps-floor feedback.
        if (tps_floor_ <= 0.0) return;
        if ((int)step_times_.size() < tps_window_) return;

        double avg = 0.0;
        for (double t : step_times_) avg += t;
        avg /= step_times_.size();
        double observed_tps = 1.0 / std::max(avg, 1e-9);
        // Don't let the tps feedback drop below the prefix — the
        // prefix is a quality floor the user explicitly picked.
        const int min_bpw = std::max(1, n_pinned_);
        if (observed_tps < tps_floor_ && desired_n_chunks_ > min_bpw) {
            --desired_n_chunks_;
            std::fprintf(stderr,
                "streamllm-scheduler[budgeted]: observed %.1f tok/s < "
                "floor %.1f → bpw %d → %d (prefix floor=%d)\n",
                observed_tps, tps_floor_, desired_n_chunks_ + 1,
                desired_n_chunks_, n_pinned_);
            step_times_.clear();  // reset window after adjustment
        } else if (observed_tps > tps_floor_ * 1.5 &&
                   desired_n_chunks_ < max_P_) {
            ++desired_n_chunks_;
            std::fprintf(stderr,
                "streamllm-scheduler[budgeted]: observed %.1f tok/s >> "
                "floor %.1f → bpw %d → %d\n",
                observed_tps, tps_floor_, desired_n_chunks_ - 1, desired_n_chunks_);
            step_times_.clear();
        }
    }

    StreamllmRuntime * rt_ = nullptr;
    int    desired_n_chunks_ = 8;
    int    n_pinned_  = -1;   // set in on_install; < 0 = not yet finalized
    int    max_P_       = 8;
    bool   evict_tail_  = true;  // Scenario-C default: flush after use
    double tps_floor_   = 0.0;
    int    tps_window_  = 8;
    int    lookahead_   = 0;     // STREAMLLM_LOOKAHEAD — 0 disables

    // Per-layer precision heterogeneity, 3-way split. See slice_for_
    // for boundary semantics. Uniform mode = all three *_layers_ == 0.
    int    front_layers_ = 0;
    int    mid_layers_   = 0;
    int    front_bpw_    = -1;
    int    front_prefix_ = -1;
    int    mid_bpw_      = -1;
    int    mid_prefix_   = -1;
    int    rear_bpw_     = -1;
    int    rear_prefix_  = -1;
    std::unordered_map<std::string, int> layer_of_;

    std::unordered_map<std::string, int>  P_of_;
    std::unordered_map<std::string, Plan> plans_;

    std::string first_seen_;
    std::chrono::steady_clock::time_point last_step_start_;
    std::vector<double> step_times_;
    int step_idx_ = 0;
    // Recorded tensor traversal order from step 0, for lookahead.
    std::vector<std::string>             step_order_;
    std::unordered_map<std::string, int> step_pos_;
};


} // anonymous


std::unique_ptr<Scheduler> make_scheduler(const char * which) {
    if (which == nullptr || which[0] == '\0' ||
        std::strcmp(which, "eager") == 0) {
        return std::make_unique<EagerScheduler>();
    }
    if (std::strcmp(which, "lazy") == 0) {
        return std::make_unique<LazyScheduler>();
    }
    if (std::strcmp(which, "prefetch") == 0 ||
        std::strcmp(which, "layer_prefetch") == 0) {
        return std::make_unique<LayerPrefetchScheduler>();
    }
    if (std::strcmp(which, "budgeted") == 0) {
        return std::make_unique<BudgetedScheduler>();
    }
    std::fprintf(stderr,
        "streamllm-scheduler: unknown name %s; defaulting to eager\n",
        which);
    return std::make_unique<EagerScheduler>();
}

} // namespace streamllm_ext
