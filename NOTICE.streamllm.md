# StreamLLM fork of llama.cpp

This branch (`streamllm-ext`) adds the StreamLLM chunked-weight
extension on top of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
at pinned upstream commit `cd03ec7642f192c13689c7f20096dcca38dc5e33`
(2026-04-21, "llama-ext : fix exports", recorded in
`UPSTREAM_COMMIT.streamllm`).

Upstream is MIT licensed (see `LICENSE`). Our additions are
Apache-2.0 — compatible under MIT's permissive terms.

## What this branch adds

- `streamllm-ext/` — a new subdirectory containing our extension
  sources (stream reader, VRAM pool, upstream-layout builder, custom
  ggml op, scheduler). Build is opt-in via `-DSTREAMLLM_EXT=ON`.
- A small set of opt-in patches to upstream files (`src/llama*.cpp`,
  `src/llama*.h`, `ggml/include/ggml-cuda.h`, `ggml/src/ggml-cuda/ggml-cuda.cu`,
  top-level `CMakeLists.txt`) that expose the hook points our extension
  binds to. Every patch is documented in `CHANGELOG.streamllm.md` and
  gated by `#if defined(STREAMLLM_EXT_ENABLED)` or by a null
  function-pointer check, so `-DSTREAMLLM_EXT=OFF` (default) produces
  a build bit-identical to upstream at the pinned commit.
- `NOTICE.streamllm.md` (this file).
- `CHANGELOG.streamllm.md` — per-file log of upstream edits.
- `UPSTREAM_COMMIT.streamllm` — pinned hash + date + message.

Every other upstream file is unchanged.

## What StreamLLM is

A runtime for LLM inference that streams quantized weight chunks in
and out of VRAM on demand, with a programmable scheduler controlling
what's loaded, what's evicted, and which chunk subsets each weight
uses per forward. The kernel is NAVER's LUT-GEMV (from
[naver-aics/anybcq](https://github.com/naver-aics/anybcq), vendored
by the parent StreamLLM repo). Our extension grafts that kernel +
the scheduler + a chunked VRAM pool onto llama.cpp's C++ runtime.

The **encoder** (offline, Python) produces `.stream` files that this
branch's runtime reads. Encoder lives in the parent `StreamLLM` repo:
https://github.com/kimGreg/StreamLLM.

## Planned file layout (milestones M2–M5)

- `streamllm-ext/stream_reader.{h,cpp}` — reads `.bin` + `.json`
  artifacts produced by `streamllm.scripts.encode_qwen3`.
- `streamllm-ext/vram_pool.{h,cpp}` — chunked VRAM pool (cudaMalloc
  arena + first-fit + LRU + pin + optional copy stream).
- `streamllm-ext/upstream_layout.{h,cpp}` — builds NAVER's kernel-
  ready `q_weight[K/32][P][M]` + fp16 `alpha[K_groups][P][M]` + fp16
  `q_bias[K_groups][M]` tensors from on-disk sign bytes.
- `streamllm-ext/scheduler.{h,cpp}` — Lazy / LayerPrefetch / Budgeted.
  Decides chunks per weight + what to prefetch; runs between
  llama.cpp's per-token iterations.
- `streamllm-ext/custom_op.{h,cpp}` — registers a custom ggml op that
  replaces `ggml_mul_mat` for managed linear layers. Forward function
  dispatches to NAVER's `nqmv_bias` kernel for GEMV or to a fused
  fallback otherwise.

## Upstream sync

```bash
git remote add upstream https://github.com/ggml-org/llama.cpp.git
git fetch upstream
git rebase upstream/master  # onto a newer pinned commit
# resolve any conflicts in CMakeLists.txt (the anchor text may have moved)
# update UPSTREAM_COMMIT.streamllm with the new hash/date/message
# re-run streamllm-ext integration tests
```

Don't cherry-pick individual upstream patches — always rebase onto a
pinned upstream commit so the provenance stays clean.

## How the parent repo consumes this

`StreamLLM/third-party/llama.cpp/` is a git submodule pointing at this
fork's `streamllm-ext` branch. The parent repo pins a specific SHA so
clones are reproducible. See the parent repo's README.
