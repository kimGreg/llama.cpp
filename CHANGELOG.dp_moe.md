# DP_MoE fork — upstream-file changelog

Chronological list of edits this fork makes to files that are part of
upstream llama.cpp / ggml. Everything under `dp-moe-ext/` is fork-
only and not listed here.

The target is to keep this list short: any entry here is code we will
have to re-apply during an upstream rebase. Before adding a new line,
ask whether the same goal can be achieved in `dp-moe-ext/`.

## 2025 — initial hook wiring

`dp-moe-ext: mul_mat hook infrastructure + llama.cpp glue` (`e84856a81`)

- `ggml/include/ggml-cuda.h`: add `ggml_cuda_set_mul_mat_hook(void *)`.
- `ggml/src/ggml-cuda/ggml-cuda.cu`: call the registered hook at the
  top of `ggml_cuda_mul_mat`; fall through on `false`.
- `src/llama.cpp`: after a successful `llama_model_load`, call
  `dp_moe_ext::install_for_gguf(path_model)` guarded by
  `#if defined(DP_MOE_EXT_ENABLED)`. Balanced by
  `dp_moe_ext::clear()` in `llama_model::~model` (see
  `src/llama-model.cpp`).

Rationale: model-level hook is the only way to intercept the per-
mul_mat dispatch without taking over graph execution. Touching
ggml-cuda is kept to the minimum (one function pointer +
one early-return call).

## 2025 — CUDA-graph capture guard + loader byte-blob skip

`dp-moe-ext: hook debug infra + graph-capture guard` (`f14a2d322`)

- `ggml/src/ggml-cuda/ggml-cuda.cu`: when the hook is installed,
  disable CUDA graph capture for graphs that contain at least one
  managed mul_mat (our hook uses shared host-side scratch — not
  capture-safe). **Updated 2026-04-24**: after the GEMV scratch
  refactor the hot-path hook is now graph-safe. The disable is still
  default-on but lifted by `DP_MOE_ENABLE_CUDA_GRAPHS=1`, which
  gains +8% decode on eager. **Updated again 2026-04-24**: now
  works with swap1 too — see "multi-stream capture fix" below.
- `ggml/src/ggml-cuda/ggml-cuda.cu`: added a direct `fprintf(stderr)`
  duplicate of `GGML_LOG_ERROR` inside `ggml_cuda_error()` so CUDA
  error strings survive through to process abort — the log callback
  can be filtered / buffered, losing the error message before abort.

## 2026-04-24 — multi-stream CUDA graph capture (swap1)

`DP_MOE_ENABLE_CUDA_GRAPHS=1` initially only worked for eager. In
swap1 (H2D during decode on the pool's copy stream), capture died
with `"capturing stream has unjoined work"` at `cudaStreamEndCapture`
because (a) copy_stream wasn't part of the compute-stream capture
session, and (b) the lookahead prefetch (`DP_MOE_PREFETCH_L`)
issued H2Ds for next-forward tensors with no corresponding wait in
this capture window.

Fix:
- `dp-moe-ext/vram_pool.{h,cu}`: `load()` and `launch_copy_()`
  take an optional `compute_stream`. When that stream is in active
  capture, copy_stream forks into the same capture via a persistent
  `capture_fork_event_` (event-based fork pattern). Under capture,
  event destroys are deferred (`pending_event_destroys_`) and drained
  after the capture window exits — destroying a captured event
  mid-capture leaves the graph with dangling references.
  `record_compute_event()` returns early under capture so the shared
  `latest_compute_event_` handle doesn't carry capture-session
  artifacts across windows.
- `dp-moe-ext/runtime.{h,cpp}`: `move_chunk()` grows an optional
  `compute_stream` and plumbs it into `pool.load()`.
- `dp-moe-ext/runtime_hook.cpp`: passes `stream` through to
  `move_chunk`. When compute_stream is capturing, it filters
  `plan.moves` down to only the current tensor's moves — the
  lookahead prefetches for other tensors would create
  copy_stream work whose downstream wait lives in the next forward
  pass (outside the current capture). Each captured graph must be
  self-contained.

Measured: swap1 no longer crashes with graphs on; tg64 matches
graphs-off (within measurement noise) because swap1 is
PCIe-bandwidth bound, not CPU-launch-bound. Eager unchanged at
+19.7% vs baseline. Safe to default `DP_MOE_ENABLE_CUDA_GRAPHS=1`
across all scheduler modes; still env-var-gated for the safety
margin.

`dp-moe-ext: split-layout GGUF (F16 placeholder + byte blob)` (`86b0d1c8b`)

- `src/llama-model-loader.cpp`: skip tensors whose name starts with
  `dp_moe.bytes.` in both GGUF-context readers. These are raw
  byte blobs consumed out-of-band by `DPMoERuntime`; they have
  no arch-level `create_tensor()` call and would otherwise unbalance
  `n_tensors`.

## 2026-04-24 — loader skip for managed weight tensors

Memory savings: ~7.6 GB of redundant CUDA0 backend buffer dropped on
Qwen3-4B (the managed tensors no longer get a regular fp16 mirror).

- `src/llama-model-loader.h`: add `std::unordered_set<std::string>
  dp_moe_managed` (names listed in the GGUF's
  `dp_moe.managed_tensors` key) + `<unordered_set>` include.
- `src/llama-model-loader.cpp`:
  - Populate `dp_moe_managed` in the loader constructor right
    after metadata is loaded. No-op on stock GGUFs.
  - In `load_all_data`, skip the disk copy for managed tensors —
    their bytes are consumed out-of-band by `DPMoERuntime`.
- `src/llama-model.cpp`:
  - Before `ggml_backend_alloc_ctx_tensors_from_buft`, point each
    managed tensor's `t->data` at a shared 4 KB CUDA "seed"
    allocation. `ggml_backend_alloc_ctx_tensors_from_buft_impl`
    treats tensors with non-null `data` as already-placed and
    excludes their bytes from the buffer (reducing the CUDA0 model
    buffer from ~7.6 GB → ~0.7 GB on Qwen3-4B).
  - After alloc, restore `t->buffer = buf` so `ggml_backend_sched`
    routes the mul_mat to CUDA via `src0->buffer->buft`.
  - Add `dp_moe_seed_ptrs` to `llama_model::impl` and free the
    seeds in the destructor.
  - Include `<cuda_runtime.h>` inside the `DP_MOE_EXT_ENABLED`
    guard.

## 2026-04-24 — framework / algorithm vocabulary split

Code-wide rename to keep the framework surface algorithm-neutral.
"Plane" / "bpw" / "precision" are AnyBCQ concepts that had leaked
into the framework interface; the framework only sees *chunks*.

Env vars:
- `DP_MOE_INIT_BPW`         → `DP_MOE_CHUNKS`
- `DP_MOE_PREFIX_BPW`       → `DP_MOE_PINNED_CHUNKS`
- `DP_MOE_EVICT_TAIL`       → `DP_MOE_EVICT_UNPINNED`
- `DP_MOE_PREFETCH_L`       → `DP_MOE_LOOKAHEAD`
- `DP_MOE_{FRONT,MID,REAR}_BPW`    → `..._CHUNKS`
- `DP_MOE_{FRONT,MID,REAR}_PREFIX` → `..._PINNED`

C++ symbols (framework-facing):
- `kMaxPlanesRt`, `kMaxPlanes` → `kMaxChunksPerTensor` (runtime.h)
                                  + `kNaverMaxPrecision` (naver_gemv.h),
                                  tied via `static_assert` in runtime.h.
- `cid_plane(i)`, `kCidPlaneBase`, `cid_is_plane`, `cid_plane_index`
  → `cid_chunk(i)`, `kCidChunkBase`, `cid_is_chunk`, `cid_chunk_index`.
- `UpstreamLayoutHost::{P, plane_chunks, chunk_bytes_per_plane,
   alpha_bytes_per_plane}` → `{n_chunks, chunks, bytes_per_chunk,
   alpha_bytes_per_chunk}`.
- `UpstreamLayoutDevice::{P, plane_chunk_ptrs}` → `{n_chunks,
   chunk_ptrs}`.
- `Entry::{d_plane_qw_ptrs, d_plane_alpha_ptrs}` → `{d_chunk_qw_ptrs,
   d_chunk_alpha_ptrs}`.
- `BudgetedScheduler::{desired_bpw_, prefix_bpw_}` → `{desired_n_chunks_,
   n_pinned_}`.

Names left algorithm-internal on purpose: `naver_gemv`, `naver_gemm`,
`nqmv_bias_planes`, `upstream_layout`, `dequant_planes`, `q_bias`,
`qw`, `alpha`. These are NAVER/AnyBCQ terms and don't need to be
generic — swapping to a different algorithm replaces them entirely.

No behaviour change. Eager tg64 at 6 chunks = 97.98 t/s matches the
prior 99.35 within measurement noise.

## 2026-04-24 — MoE readiness audit (no code changes)

Qwen3-30B-A3B is the nominal scale target but no encoded AnyBCQ GGUF
exists, and encoding requires >30 GB VRAM (impossible on the
4070 Ti SUPER). Rather than ship dead-code stubs, the state of MoE
support is filed here:

- **scheduler's `parse_layer()` (scheduler.cpp:101-121)**: already
  handles MoE names via its scan-digits-after-"blk." implementation.
  Returns the outer block index, ignoring the inner expert index X
  in `blk.<N>.ffn_*_exps.<X>.weight`. No change needed. Comment in
  the function body locks in the contract.
- **Encoder (`dp_moe/scripts/encode_qwen3.py:57-74`)**: currently
  excludes `.mlp.experts.*` tensors. Needed: (a) `--include-moe`
  CLI flag, (b) switch to `TensorNameMap(MODEL_ARCH.QWEN3MOE)` for
  `blk.<N>.ffn_{up,gate,down}_exps.<X>.weight` GGUF name mapping,
  (c) emit the expert tensors into `dp_moe.managed_tensors`.
- **CUDA hook (`ggml-cuda.cu:2532+`)**: the dp_moe hook only fires
  for `GGML_OP_MUL_MAT`. MoE uses `GGML_OP_MUL_MAT_ID` (separate
  dispatch at `ggml_cuda_mul_mat_id`, never consults the hook). A
  parallel hook `g_cuda_mul_mat_id_hook` with signature
  `bool(*)(cudaStream_t, const ggml_tensor *src0_exps,
           const ggml_tensor *src1, const ggml_tensor *ids,
           ggml_tensor *dst)` would be needed, consulted at entry of
  `ggml_cuda_mul_mat_id`.

All three TODOs will land together when a Qwen3-30B-A3B AnyBCQ GGUF
becomes available for end-to-end validation.

## 2026-04-24 — fusion-skip hook

- `ggml/include/ggml-cuda.h`: add
  `ggml_cuda_set_fusion_skip_hook(void *)`.
- `ggml/src/ggml-cuda/ggml-cuda.cu`:
  - Store the hook pointer at file scope.
  - In `ggml_cuda_should_fuse_mul_mat` (ffn_up + ffn_gate + glu),
    `ggml_cuda_should_fuse_mul_mat_vec_f`, and `..._vec_q`: bail out
    early if the hook claims the candidate src0 tensor. Without this,
    those fused kernels read src0 directly, bypassing the dp_moe
    mul_mat hook and dereferencing the 4 KB seed pointer (illegal-
    memory-access / misaligned-address crash on the last-layer FFN).

Rationale: the fusion paths are compiled-in optimisations that
predate the hook protocol. A one-line call per predicate is enough
to opt managed weights out of fusion; the unfused mul_mat then hits
the existing hook dispatch.

## 2026-04-25 — mul_mat_id hook (MoE Phase B)

- `ggml/include/ggml-cuda.h`: add
  `ggml_cuda_set_mul_mat_id_hook(void *)`. Mirrors the existing
  dense `ggml_cuda_set_mul_mat_hook` but for `GGML_OP_MUL_MAT_ID`
  (the MoE expert dispatch).
- `ggml/src/ggml-cuda/ggml-cuda.cu`:
  - Add `ggml_cuda_mul_mat_id_hook_t` typedef + static
    `g_cuda_mul_mat_id_hook` ptr + setter (mirrors the dense
    pattern at lines 2440-2446).
  - At the top of `ggml_cuda_mul_mat_id` (line 2541), call the
    registered hook before any of the dispatch branches; return
    early if the hook returns true.

Total: ~13 lines of upstream-touch (1 in the header, 12 in the
.cu). Rebase cost matches the dense hook entry from 2025 plus the
fusion-skip entry from 2026-04-24.

The dp-moe-ext side currently lands as a Phase B *skeleton*:
the hook registers correctly, fires on managed expert-stack
tensors, and logs a one-time trace per tensor name; it returns
false (falls through) until Phase C's per-(token, expert)
`chunk_matmul` loop lands. The `dp_moe-moe-hook-test` binary
exercises the protocol — setter accepts fn + nullptr, idempotent.
