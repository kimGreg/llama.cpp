// streamllm-ext — runtime-hook implementation.

#include "runtime_hook.h"
#include "runtime.h"
#include "stream_reader.h"
#include "naver_gemv.h"
#include "cast_f16.h"
#include "dequant_planes.h"
#include "batched_gemm.h"

#include <ggml.h>
#include <gguf.h>
#include <ggml-cuda.h>

#include <cuda_fp16.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace streamllm_ext {

namespace {

// Global runtime owned by the hook. Model load constructs it, model
// free clears it. Protected by a mutex so pathological reload paths
// don't race — in normal usage there's exactly one active model at a
// time so the mutex is uncontended.
std::mutex g_runtime_mu;
std::unique_ptr<StreamllmRuntime> g_runtime;

// Per-stream scratch for:
//   x_f16 / y_f16        — single-token F32↔F16 cast bridge
//                          (decode path, one vector at a time)
//   xb_f16 / yb_f16      — batched F32↔F16 cast bridge for prefill
//                          (whole [K × N] / [M × N] blocks at once)
//   w_f16                — dequantised weight buffer, [M × K] fp16,
//                          reused per mul_mat call on this stream
//
// All sized once at install time to the largest (K, M, N_batch) across
// managed tensors. N_batch is set from STREAMLLM_BATCH_N_MAX (default
// 2048 — enough for llama.cpp's PPL ctx or -b batch size).
struct StreamScratch {
    void *  x_f16  = nullptr;
    void *  y_f16  = nullptr;
    void *  xb_f16 = nullptr;
    void *  yb_f16 = nullptr;
    void *  w_f16  = nullptr;
};
std::mutex g_scratch_mu;
std::unordered_map<cudaStream_t, StreamScratch> g_scratch;
size_t g_scratch_x_bytes   = 0;
size_t g_scratch_y_bytes   = 0;
size_t g_scratch_xb_bytes  = 0;
size_t g_scratch_yb_bytes  = 0;
size_t g_scratch_w_bytes   = 0;
int    g_batch_n_max       = 0;

bool size_scratch_for(const StreamReader & r) {
    size_t max_k = 0;
    size_t max_m = 0;
    size_t max_mk = 0;
    for (const auto & name : r.managed_tensor_names()) {
        const auto * L = r.layout(name);
        if (L == nullptr) continue;
        size_t K  = (size_t)L->padded_m;
        size_t M  = (size_t)L->shape[0];
        max_k  = std::max(max_k,  K);
        max_m  = std::max(max_m,  M);
        max_mk = std::max(max_mk, M * K);
    }
    g_batch_n_max = 2048;
    if (const char * s = getenv("STREAMLLM_BATCH_N_MAX")) {
        int v = std::atoi(s); if (v > 0) g_batch_n_max = v;
    }
    g_scratch_x_bytes  = max_k * sizeof(uint16_t);
    g_scratch_y_bytes  = max_m * sizeof(uint16_t);
    g_scratch_xb_bytes = max_k * (size_t)g_batch_n_max * sizeof(uint16_t);
    g_scratch_yb_bytes = max_m * (size_t)g_batch_n_max * sizeof(uint16_t);
    g_scratch_w_bytes  = max_mk * sizeof(uint16_t);
    return g_scratch_x_bytes > 0 && g_scratch_y_bytes > 0 &&
           g_scratch_w_bytes > 0;
}

StreamScratch * scratch_for_stream(cudaStream_t stream) {
    std::lock_guard<std::mutex> lk(g_scratch_mu);
    auto it = g_scratch.find(stream);
    if (it != g_scratch.end()) return &it->second;
    StreamScratch s{};
    auto fail = [&]() {
        if (s.x_f16)  cudaFree(s.x_f16);
        if (s.y_f16)  cudaFree(s.y_f16);
        if (s.xb_f16) cudaFree(s.xb_f16);
        if (s.yb_f16) cudaFree(s.yb_f16);
        if (s.w_f16)  cudaFree(s.w_f16);
        return nullptr;
    };
    if (cudaMalloc(&s.x_f16,  g_scratch_x_bytes)  != cudaSuccess) return fail();
    if (cudaMalloc(&s.y_f16,  g_scratch_y_bytes)  != cudaSuccess) return fail();
    if (cudaMalloc(&s.xb_f16, g_scratch_xb_bytes) != cudaSuccess) return fail();
    if (cudaMalloc(&s.yb_f16, g_scratch_yb_bytes) != cudaSuccess) return fail();
    if (cudaMalloc(&s.w_f16,  g_scratch_w_bytes)  != cudaSuccess) return fail();
    auto ins = g_scratch.emplace(stream, s);
    return &ins.first->second;
}

void free_scratch() {
    std::lock_guard<std::mutex> lk(g_scratch_mu);
    for (auto & [s, sc] : g_scratch) {
        if (sc.x_f16)  cudaFree(sc.x_f16);
        if (sc.y_f16)  cudaFree(sc.y_f16);
        if (sc.xb_f16) cudaFree(sc.xb_f16);
        if (sc.yb_f16) cudaFree(sc.yb_f16);
        if (sc.w_f16)  cudaFree(sc.w_f16);
    }
    g_scratch.clear();
    g_scratch_x_bytes = g_scratch_y_bytes = 0;
    g_scratch_xb_bytes = g_scratch_yb_bytes = g_scratch_w_bytes = 0;
}

// Conservative pool sizing used by install_for_gguf. Mirrors
// test_runtime_full.cpp::estimate_pool_bytes — summed over every
// managed tensor with a 2× margin for allocator fragmentation.
size_t estimate_pool_bytes(const StreamReader & r) {
    size_t total = 0;
    for (const auto & name : r.managed_tensor_names()) {
        const auto * L = r.layout(name);
        if (L == nullptr) continue;
        int32_t M  = (int32_t)L->shape[0];
        int32_t K  = L->padded_m;
        int32_t P  = (int32_t)L->chunk_bytes.size();
        int32_t Kg = (int32_t)L->n_groups_per_row;
        total += (size_t)(K / 32) * P * M * 4;
        total += (size_t)Kg * P * M * 2;
        total += (size_t)Kg * M * 2;
    }
    return total * 2;
}

} // anonymous namespace


bool install_for_gguf(const char * gguf_path) {
    if (gguf_path == nullptr || gguf_path[0] == '\0') return false;

    gguf_init_params p{ /*no_alloc=*/true, /*ctx=*/nullptr };
    gguf_context * ctx = gguf_init_from_file(gguf_path, p);
    if (ctx == nullptr) return false;

    auto reader_opt = StreamReader::from_gguf(ctx, gguf_path);
    gguf_free(ctx);
    if (!reader_opt) {
        // No streamllm.* block — stock GGUF, leave default dispatch alone.
        return false;
    }

    const auto & reader = *reader_opt;

    std::lock_guard<std::mutex> lk(g_runtime_mu);

    if (g_runtime) {
        // Previous model wasn't cleared. Drop it before rebuilding so
        // pinned VRAM isn't leaked across reloads.
        std::fprintf(stderr,
            "streamllm-ext: replacing previously-installed runtime "
            "(prior model not cleared explicitly)\n");
        g_runtime.reset();
    }

    size_t cap = estimate_pool_bytes(reader);
    if (const char * s = getenv("STREAMLLM_VRAM_CAP_MB")) {
        // Explicit cap overrides the estimate. Budgeted scheduler
        // needs this to exercise eviction; other schedulers still
        // benefit when the user wants a tighter ceiling.
        long mb = std::atol(s);
        if (mb > 0) cap = (size_t)mb * 1024UL * 1024UL;
    }
    const char * sched_name = getenv("STREAMLLM_SCHEDULER");
    // Async copy stream: M7 onward overlaps H2D with compute. The pool
    // chains H2D after the latest compute event and records a ready
    // event per load so chunk_matmul can wait before launching.
    g_runtime = std::make_unique<StreamllmRuntime>(
        cap, /*device=*/0, /*copy_stream=*/true, sched_name);

    g_runtime->install(reader, std::string(gguf_path));
    cudaDeviceSynchronize();

    if (!size_scratch_for(reader)) {
        g_runtime.reset();
        throw std::runtime_error(
            "streamllm-ext: failed to size cast scratch");
    }

    std::fprintf(stderr,
        "streamllm-ext: runtime ready "
        "(scheduler=%s, pool %.1f / %.1f MB used)\n",
        g_runtime->scheduler().name(),
        (double)g_runtime->pool().used_bytes()     / 1024.0 / 1024.0,
        (double)g_runtime->pool().capacity_bytes() / 1024.0 / 1024.0);

    // If the user wants streaming stats isolated from install warmup,
    // reset the counters here so ``STREAMLLM_STATS=1`` at teardown
    // reports post-install cumulative traffic only.
    if (getenv("STREAMLLM_STATS_RESET_AFTER_INSTALL")) {
        g_runtime->pool().reset_stats();
    }

    ggml_cuda_set_mul_mat_hook((void *) &streamllm_try_cuda_mul_mat);
    ggml_cuda_set_fusion_skip_hook((void *) &streamllm_claims_tensor);
    return true;
}

// Lightweight name lookup — called from ggml-cuda during fusion
// candidate checks. Must be cheap since it runs per mul_mat per
// graph compile. The lookup walks the runtime's upstream-layout map
// which is keyed by name.
extern "C" bool streamllm_claims_tensor(const struct ggml_tensor * w) {
    if (w == nullptr || w->name[0] == '\0') return false;
    std::lock_guard<std::mutex> lk(g_runtime_mu);
    if (!g_runtime) return false;
    return g_runtime->layout(w->name) != nullptr;
}


void clear() {
    std::lock_guard<std::mutex> lk(g_runtime_mu);
    if (g_runtime && getenv("STREAMLLM_STATS")) {
        const auto & p = g_runtime->pool();
        std::fprintf(stderr,
            "streamllm-ext stats: scheduler=%s peak_pool=%.1f MB "
            "h2d=%.1f MB in %zu moves (avg %.1f KB/move)\n",
            g_runtime->scheduler().name(),
            (double)p.peak_used_bytes() / 1024.0 / 1024.0,
            (double)p.total_h2d_bytes() / 1024.0 / 1024.0,
            p.total_h2d_calls(),
            p.total_h2d_calls()
                ? (double)p.total_h2d_bytes() / 1024.0 / p.total_h2d_calls()
                : 0.0);
    }
    ggml_cuda_set_mul_mat_hook(nullptr);
    ggml_cuda_set_fusion_skip_hook(nullptr);
    g_runtime.reset();
    free_scratch();
    batched_gemm_shutdown();
}


// ---- the hook ----------------------------------------------------------

extern "C" bool streamllm_try_cuda_mul_mat(
    cudaStream_t stream,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    struct ggml_tensor * dst)
{
    static std::atomic<int> n_called{0};
    static std::atomic<int> n_hit{0};
    static std::atomic<int> n_miss_shape{0};
    static std::atomic<int> n_miss_dtype{0};
    static std::atomic<int> n_miss_stride{0};
    static std::atomic<int> n_miss_runtime{0};
    static std::atomic<int> n_miss_name{0};

    auto diag = [&](const char * reason, const char * extra = nullptr) {
        if (getenv("STREAMLLM_TRACE")) {
            std::fprintf(stderr,
                "streamllm-hook: %s | name=%s src0.type=%d src1.type=%d "
                "dst.type=%d ne=[%lld,%lld,%lld,%lld]x[%lld,%lld,%lld,%lld] "
                "extra=%s\n",
                reason,
                src0 && src0->name[0] ? src0->name : "?",
                src0 ? (int)src0->type : -1,
                src1 ? (int)src1->type : -1,
                dst  ? (int)dst->type  : -1,
                src0?(long long)src0->ne[0]:-1, src0?(long long)src0->ne[1]:-1,
                src0?(long long)src0->ne[2]:-1, src0?(long long)src0->ne[3]:-1,
                src1?(long long)src1->ne[0]:-1, src1?(long long)src1->ne[1]:-1,
                src1?(long long)src1->ne[2]:-1, src1?(long long)src1->ne[3]:-1,
                extra ? extra : "");
        }
    };

    if (src0 == nullptr || dst == nullptr || src1 == nullptr) return false;
    n_called.fetch_add(1);

    // Diagnostic: bypass the hook entirely (fall through to stock
    // ggml-cuda mul_mat). Useful for A/B-testing whether our kernel
    // path is producing the expected output when the canonical F16
    // placeholder also holds real weights.
    if (getenv("STREAMLLM_BYPASS")) return false;

    StreamllmRuntime * rt = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_runtime_mu);
        rt = g_runtime.get();
    }
    if (rt == nullptr) { n_miss_runtime.fetch_add(1); return false; }

    const char * name = src0->name;
    if (name == nullptr || name[0] == '\0') {
        n_miss_name.fetch_add(1); diag("miss-noname"); return false;
    }

    // Scheduler consult: ``plan.chunks`` is what chunk_matmul should
    // consume, ``plan.moves`` are the moves we fire before compute.
    const std::string wid(name);
    const Plan * plan = rt->scheduler().plan(wid, stream);
    if (plan == nullptr) {
        n_miss_name.fetch_add(1); diag("miss-name"); return false;
    }
    // Under CUDA graph capture, prefetch moves that target tensors other
    // than the current `wid` would leave copy_stream with H2Ds whose
    // downstream wait (on a future mul_mat) happens in a later forward
    // pass — the captured graph would have "unjoined work" on copy_stream
    // and cudaStreamEndCapture would abort. Each captured forward pass
    // must be self-contained, so we skip cross-forward-boundary
    // prefetches while a capture is active.
    cudaStreamCaptureStatus cap_status = cudaStreamCaptureStatusNone;
    const bool in_capture =
        cudaStreamIsCapturing(stream, &cap_status) == cudaSuccess &&
        cap_status == cudaStreamCaptureStatusActive;
    for (const auto & mv : plan->moves) {
        if (in_capture && mv.wid != wid) continue;
        rt->move_chunk(mv.wid, mv.cid, mv.src, mv.dst, (StreamHandle)stream);
    }
    const UpstreamLayoutDevice * L = rt->layout(wid);
    if (L == nullptr || L->q_bias_fp16 == nullptr ||
        L->chunk_ptrs[0] == nullptr) {
        n_miss_name.fetch_add(1); diag("miss-layout"); return false;
    }

    // NAVER's LUT-GEMV is strictly GEMV: one X vector at a time. For
    // prefill (ne[1] > 1) loop over tokens. This is O(n_tokens × M×K/32)
    // kernel launches — correct but not optimal; a batched kernel is a
    // later optimisation.
    //
    // Dtype guards: X and Y must be fp16 to match the kernel.
    // NAVER's kernel wants fp16 X in, fp16 Y out. llama.cpp graphs
    // typically plumb F32 activations, so we accept {F16, F32} on
    // both sides and bridge via cast kernels. Mixed F16/F32 (e.g. F32
    // in, F16 out) is unusual but permitted.
    const bool src1_f16 = (src1->type == GGML_TYPE_F16);
    const bool src1_f32 = (src1->type == GGML_TYPE_F32);
    const bool dst_f16  = (dst->type  == GGML_TYPE_F16);
    const bool dst_f32  = (dst->type  == GGML_TYPE_F32);
    if (!(src1_f16 || src1_f32) || !(dst_f16 || dst_f32)) {
        n_miss_dtype.fetch_add(1); diag("miss-dtype");
        return false;
    }

    // Shape sanity:
    //   src0 (W):  ne = {K, M}  in ggml convention (K=in, M=out)
    //   src1 (X):  ne = {K, n_tokens, ...}
    //   dst  (Y):  ne = {M, n_tokens, ...}
    if ((int)src0->ne[0] != L->K || (int)src0->ne[1] != L->M) {
        n_miss_shape.fetch_add(1); diag("miss-shape-src0"); return false;
    }
    if ((int)src1->ne[0] != L->K) {
        n_miss_shape.fetch_add(1); diag("miss-shape-src1"); return false;
    }
    if ((int)dst->ne[0]  != L->M) {
        n_miss_shape.fetch_add(1); diag("miss-shape-dst"); return false;
    }

    const int64_t n_tokens = src1->ne[1];
    if (n_tokens != dst->ne[1]) { n_miss_shape.fetch_add(1); return false; }
    if (src1->ne[2] > 1 || src1->ne[3] > 1) {
        n_miss_shape.fetch_add(1); diag("miss-shape-batch"); return false;
    }

    const size_t x_elt = src1_f16 ? sizeof(uint16_t) : sizeof(float);
    const size_t y_elt = dst_f16  ? sizeof(uint16_t) : sizeof(float);
    if (src1->nb[1] != (size_t)src1->ne[0] * x_elt) {
        n_miss_stride.fetch_add(1); diag("miss-stride-src1"); return false;
    }
    if (dst->nb[1]  != (size_t)dst->ne[0]  * y_elt) {
        n_miss_stride.fetch_add(1); diag("miss-stride-dst"); return false;
    }

    // Per-stream scratch for the F32↔F16 cast bridge. Concurrent mul_mats
    // on different streams no longer serialize on a global mutex.
    const size_t need_x = (size_t)L->K * sizeof(uint16_t);
    const size_t need_y = (size_t)L->M * sizeof(uint16_t);
    if (need_x > g_scratch_x_bytes || need_y > g_scratch_y_bytes) {
        diag("miss-scratch-too-small"); return false;
    }
    StreamScratch * sc = scratch_for_stream(stream);
    if (sc == nullptr) {
        diag("miss-scratch-alloc"); return false;
    }
    void * scratch_x_f16 = sc->x_f16;
    void * scratch_y_f16 = sc->y_f16;

    const uint8_t * x_base = (const uint8_t *) src1->data;
    uint8_t * y_base       = (uint8_t *)       dst->data;
    const size_t x_stride  = src1->nb[1];
    const size_t y_stride  = dst->nb[1];

    const bool stamp_constant = getenv("STREAMLLM_STAMP_DST") != nullptr;

    // Batched prefill fast path: n_tokens > 1. Dequant planes → W_f16[M,K]
    // once, then cuBLAS F16 GEMM over the whole batch. ~100× faster than
    // the per-token GEMV loop below for prefill batches. Decode
    // (n_tokens = 1) keeps the existing GEMV path (that's what NAVER's
    // kernel was designed for and where it's competitive).
    //
    // Env kill-switch: STREAMLLM_NO_BATCHED=1 forces the per-token path
    // even at n_tokens > 1 — useful for A/B sanity checks between the
    // dequant+cublas path and the original per-token kernel.
    const bool use_batched = (n_tokens > 1) && !stamp_constant &&
                             (getenv("STREAMLLM_NO_BATCHED") == nullptr) &&
                             ((size_t)L->M * (size_t)L->K * sizeof(uint16_t)
                              <= g_scratch_w_bytes) &&
                             ((size_t)L->K * (size_t)n_tokens * sizeof(uint16_t)
                              <= g_scratch_xb_bytes) &&
                             ((size_t)L->M * (size_t)n_tokens * sizeof(uint16_t)
                              <= g_scratch_yb_bytes);

    bool did_batched = false;
    if (use_batched) {
        // Cast the whole X block to F16 (or point at src1 if already F16).
        // src1 is contiguous col-major [K × n_tokens] (stride K·elt per
        // column — we checked nb[1] == ne[0] * elt above).
        const void * x_block_f16 = src1->data;
        if (src1_f32) {
            launch_f32_to_f16(src1->data, sc->xb_f16,
                              (int)((size_t)L->K * n_tokens), stream);
            x_block_f16 = sc->xb_f16;
        }
        void * y_block_f16 = dst_f16 ? dst->data : sc->yb_f16;

        bool ok = rt->chunk_matmul_batched(
            wid, plan->chunks,
            x_block_f16, y_block_f16, (int)n_tokens,
            sc->w_f16, stream);
        if (ok) {
            if (dst_f32) {
                launch_f16_to_f32(y_block_f16, dst->data,
                                  (int)((size_t)L->M * n_tokens), stream);
            }
            did_batched = true;
        } else {
            diag("batched-fallback");
            // Fall through to the per-token loop so the forward still
            // completes correctly.
        }
    }

    if (!did_batched) {
        for (int64_t t = 0; t < n_tokens; ++t) {
            const void * x_t = x_base + (size_t)t * x_stride;
            void       * y_t = y_base + (size_t)t * y_stride;

            if (stamp_constant) {
                cudaMemsetAsync(y_t, 0x3c, (size_t)L->M * y_elt, stream);
                continue;
            }

            const void * x_f16 = x_t;
            if (src1_f32) {
                launch_f32_to_f16(x_t, scratch_x_f16, L->K, stream);
                x_f16 = scratch_x_f16;
            }

            void * y_f16 = dst_f16 ? y_t : scratch_y_f16;

            rt->chunk_matmul(wid, plan->chunks,
                             x_f16, y_f16,
                             /*n_tokens=*/1,
                             /*x_stride=*/0, /*y_stride=*/0,
                             stream);

            if (dst_f32) {
                launch_f16_to_f32(y_f16, y_t, L->M, stream);
            }
        }
    }
    // Record a compute-complete event on ``stream`` so the copy stream
    // serializes future H2Ds *after* the kernel reads from its scratch.
    // No stream-sync needed — scratch is per-stream and chunk_matmul
    // already waited on in-flight ready events via the pool.
    rt->pool().record_compute_event(stream);
    // Give the scheduler a chance to flush tail chunks (Budgeted
    // streaming mode). Safe here: copy-stream H2Ds from this point on
    // wait on the event just recorded, so evicting a slot whose
    // bytes a still-running kernel is reading can't cause an
    // overwrite — the next load() into that slot blocks until the
    // compute event fires.
    rt->scheduler().after_compute(wid, stream);
    if (getenv("STREAMLLM_DEVICE_SYNC")) {
        cudaDeviceSynchronize();
    }
    // Optional XY dump: capture src1 (X) and dst (Y) to disk for one
    // specific tensor on its FIRST hit, so a Python reference can
    // verify decompressed_W @ X ≈ Y and isolate any discrepancy
    // between the in-situ hook path and the streamllm-runtime-test
    // reference path.
    //
    //   STREAMLLM_DUMP_XY=<tensor_name>     (e.g. blk.0.attn_q.weight)
    //   STREAMLLM_DUMP_DIR=<path>           (defaults to /tmp/sllm_hook)
    //
    // Files written:
    //   <dir>/<name>.x.bin    — src1 contents, fp32 or fp16
    //   <dir>/<name>.y.bin    — dst contents, fp32 or fp16
    //   <dir>/<name>.meta.txt — M, K, P, group_size, n_tokens, dtypes
    if (const char * wanted = getenv("STREAMLLM_DUMP_XY")) {
        if (std::strcmp(wanted, name) == 0) {
            static std::atomic<int> xy_dumped{0};
            // Optional: capture a specific hit index instead of the
            // first. STREAMLLM_DUMP_HIT=K dumps the Kth call (0-based).
            int target_hit = 0;
            if (const char * hs = getenv("STREAMLLM_DUMP_HIT")) {
                target_hit = std::atoi(hs);
            }
            int this_hit = xy_dumped.fetch_add(1);
            if (this_hit == target_hit) {
                cudaStreamSynchronize(stream);
                const char * dir = getenv("STREAMLLM_DUMP_DIR");
                if (dir == nullptr) dir = "/tmp/sllm_hook";
                // Best-effort mkdir (system-independent enough for PoC).
                std::string mk = std::string("mkdir -p ") + dir;
                (void) system(mk.c_str());

                const size_t x_total = (size_t) L->K * n_tokens * x_elt;
                const size_t y_total = (size_t) L->M * n_tokens * y_elt;
                std::vector<uint8_t> x_buf(x_total), y_buf(y_total);
                cudaMemcpy(x_buf.data(), src1->data, x_total, cudaMemcpyDeviceToHost);
                cudaMemcpy(y_buf.data(), dst->data,  y_total, cudaMemcpyDeviceToHost);

                auto write_file = [&](const std::string & p, const void * data, size_t n) {
                    FILE * f = fopen(p.c_str(), "wb");
                    if (!f) return;
                    fwrite(data, 1, n, f);
                    fclose(f);
                };
                std::string x_path = std::string(dir) + "/" + name + ".x.bin";
                std::string y_path = std::string(dir) + "/" + name + ".y.bin";
                std::string m_path = std::string(dir) + "/" + name + ".meta.txt";
                write_file(x_path, x_buf.data(), x_total);
                write_file(y_path, y_buf.data(), y_total);

                FILE * mf = fopen(m_path.c_str(), "w");
                if (mf) {
                    std::fprintf(mf,
                        "name=%s\nM=%d\nK=%d\nP=%d\ngroup_size=%d\n"
                        "n_tokens=%lld\nsrc1_type=%s\ndst_type=%s\n",
                        name, L->M, L->K, L->n_chunks, L->group_size,
                        (long long)n_tokens,
                        src1_f16 ? "fp16" : "fp32",
                        dst_f16  ? "fp16" : "fp32");
                    fclose(mf);
                }
                std::fprintf(stderr,
                    "streamllm-dump-xy: wrote %s (X %zuB, Y %zuB, nt=%lld)\n",
                    name, x_total, y_total, (long long)n_tokens);
            }
        }
    }

    n_hit.fetch_add(1);
    diag("hit");
    return true;
}

// Debug helper — dumps counters to stderr. Called from the model-free
// path so users get a one-line summary of how well the hook was engaged.
void log_hook_stats() {
    // Unused for now; counters are private atomics above. A future
    // refactor should expose them; this stub keeps the symbol alive.
}

} // namespace streamllm_ext
