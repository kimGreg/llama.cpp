# Roofline analysis — StreamLLM on RTX 4070 Ti SUPER + NVMe

Hardware bandwidths and FLOPS measured on this box, applied against
the streamllm runtime's measured per-token byte traffic + cache hit
rates + per-call CPU overhead, to derive theoretical TPS bounds for
Qwen3-30B-A3B and explain the gap to measured.

The model splits per-token wall time into a **bandwidth lower bound**
and an **additive per-chunk software-overhead** term. The bandwidth-
only model in earlier drafts conflated the two; the software term is
~33% of move_chunk wall at config C and is the next thing to attack.

## 1. Hardware ceilings

| tier | bandwidth | how |
|---|---|---|
| **NVMe**, sequential 1 GB read, `O_DIRECT` | **2.3 GB/s** | `dd if=$MODEL of=/dev/null bs=1M count=1024 iflag=direct` |
| **NVMe**, random 100 KB at QD≈8 (cold cache) | ~2 GB/s | spec / nsys traces |
| **DRAM** (DDR5) | ~30 GB/s | spec |
| **PCIe 4.0 x16 H2D, pinned** | **26.7 GB/s** | torch `empty(1 GB).copy_(pinned)` |
| **VRAM** (GDDR6X) | **288 GB/s** one-way | torch DtoD, 2 GB |
| **GPU FP16 GEMM** | **78 TFLOPS** | torch 8192³ matmul, ≈87% of 90 TFLOPS spec |

DRAM never binds in this analysis; numpy memcpy's 5 GB/s reading was
python interpreter overhead.

## 2. Per-token byte streams (refined)

The streaming pipeline carries **three distinct byte streams** whose
sizes differ by ~25–30 %. Earlier drafts used one shared `bytes(X)`
for all tiers; this hides the format transformation done at
`plane_disk_to_kernel`.

| stream | what | typical for Qwen3 chunk |
|---|---|---|
| `B_ssd_disk` | bytes pread'd from GGUF — signs + α-as-fp16 (v2 default) | `qw_bytes_per_chunk + K_groups·n·2` |
| `B_h2d_kernel` | bytes uploaded H2D — signs + α-as-fp16 (kernel layout) | `host.bytes_per_chunk = qw_bytes_per_chunk + K_groups·n·2` |
| `B_vram_kernel` | bytes the kernel reads from VRAM during chunk_matmul | `B_h2d_kernel` |

**Updated 2026-05-07** — byte-stream v2 narrows α (and β) to fp16 on
disk. `B_ssd_disk == B_h2d_kernel` now, and `plane_disk_to_kernel`'s α
conversion is a transpose-only memcpy of half-words instead of an RNE
narrow. The disk-format chunk is ~18 % smaller than v1 fp32-α; the
roofline still wants **`B_ssd_disk` against `BW_SSD`** and
**`B_h2d_kernel` against `BW_PCIe`** as separate budgets, but the two
sizes are now equal in v2 streams. Legacy v1 streams (fp32 α/β) are
no longer accepted — re-encode artifacts to v2.

Streamllm streams **only the FFN routed-expert weights**. Attention,
dense layers, embeddings, router gates stay fp16-resident in VRAM
(≈2.3 GB at start).

## 3. The roofline equation

```
T_token  ≥  T_bw_lower_bound  =  max(
              B_ssd_disk    × p_SSD(X)  / BW_SSD,
              B_h2d_kernel  × p_PCIe(X) / BW_PCIe,
              B_vram_kernel             / BW_VRAM,
              FLOPs                     / FLOPS )

T_token_observed  =  T_bw_lower_bound
                  +  N_ssd_chunks    × t_xform
                  +  N_uploads       × t_pool_load_enq
                  +  N_cache_inserts × t_memcpy
                  +  N_allocs        × t_alloc
                  +  T_scheduler_lock
                  +  T_sync_wait               ← futex traffic
                  +  T_dup_load_waste

p_SSD  (X) = 1 − hit_VRAM(X) − hit_DRAM(X)
p_PCIe (X) = 1 − hit_VRAM(X)

bandwidth_efficiency      = T_bw_lower_bound / T_token_observed
software_overhead_total   = T_token_observed − T_bw_lower_bound
```

For our workload only the first two bandwidth terms ever bind. The
**`T_sync_wait` term** — workers blocked on the prefetch queue's
mutex/condvars — turned out to be the largest software-overhead
summand, only visible via `strace`/`perf`, not chrono instrumentation.

## 4. Measured strategies — `score` policy catalog

Active policy is **`STREAMLLM_MOE_POLICY=score`**: per-(t, u) gate
score `g` is looked up in a descending threshold table; per-expert
**MAX(g) across the minibatch** drives a single per-expert precision
that every (t, u) routed to that expert shares. This makes batched
prefill cheap — chunks loaded for the highest-scoring token in a
batch are reused free for the lower-scoring ones.

All runs below: 32-token decoded run on prompt `"The capital of
France is"`, `STREAMLLM_VRAM_CAP_MB=8192`, no host cap, 8 prefetch
workers, DIAG=ON, score policy, RTX 4070 Ti SUPER + NVMe.

| name | `SCORE_THRESHOLDS` | `SCORE_CHUNKS` | prefill TPS | decode TPS | peak pool | h2d total | SSD chunks | mean_des r0 / r7 |
|---|---|---|---:|---:|---:|---:|---:|---|
| **always-8** | `0.0` | `8` | **1.24** | **4.15** | 8192 MB | 27.9 GB | 58 752 | 8.00 / 8.00 |
| **coarse** | `0.4,0.3,0.1,0.05` | `8,6,4,2` | **3.61** | **8.88** | 5722 MB | 5.7 GB | 20 726 | 2.96 / 2.07 |
| **fine** | `0.3,0.25,0.2,0.15,0.1,0.05` | `8,7,6,4,3,1` | **5.16** | **9.87** | 4384 MB | 4.4 GB | 15 019 | 2.63 / 1.09 |

(Historical reference: the prior `rank_cum=[2,1,1,1]` policy delivered
prefill 5.24 / decode 11.18 at this same cap — fastest, but at the
floor of the precision dial. The score policy with comparable chunk
budget would match it; the table above intentionally spans the dial
from full-precision down.)

Compute-planes histogram per strategy, % of dispatches at each plane
count, shows what fraction of (t, u) pairs actually land at each
precision tier:

```
                  1     2     3     4     6     7     8
always-8                                                100.0
coarse                  80.2        18.7        0.5    0.6
fine            79.7          12.3   4.6  1.7   0.7    1.0
```

Reading: the **fine** ladder pushes 79.7 % of (t, u) to the 1-chunk
floor while still letting top-rank tokens reach 8 planes when their
score crosses 0.3. The **coarse** ladder has no 1-chunk floor (its
fallback is 2), so its compute work is roughly uniformly higher.
**always-8** is the static P=8 baseline.

Per-expert MAX aggregation visible at rank 7: under **coarse**, mean_
desired at rank 7 is 2.07 (not exactly 2) because ~7 % of the
lowest-rank tokens are bumped to the 4-tier when their expert was
also picked at higher rank in the same minibatch.

## 5. SSD-bandwidth roof per strategy

```
SSD-bound TPS = (BW_SSD × parallelism) / B_ssd_disk_per_tok
```

`BW_SSD` here is the **effective** bandwidth observed at the prefetch
worker pool (not raw NVMe spec). Across all runs, per-chunk pread
averages 700–770 µs at 8 workers ⇒ ≈220 MB/s effective at
~150 KB/chunk, with ~3.3× effective parallelism inside `wait_barrier`.

| name | SSD bytes/tok | bw-roof (parallel) | observed | bw-efficiency |
|---|---:|---:|---:|---:|
| always-8 | ≈ 870 MB (huge churn at cap) | ~2.4 TPS | 4.15 TPS | 1.7× ¹ |
| coarse | ≈ 100 MB | ~7.3 TPS | 8.88 TPS | 1.2× |
| fine | ≈ 73 MB | ~9.9 TPS | 9.87 TPS | 1.0× |

¹ always-8 exceeds the seq-bw roof because at this cap the pool
churns: 17 % of attempts hit DRAM (chunks just-evicted that come
back), softening the effective `B_ssd`.

The "bw-efficiency > 1" entries reflect that hooks issue ~3 SSD
chunks in parallel and wait sees max() not sum() — for batched
prefill this stretches further. The takeaway: **fine** sits on its
parallel-SSD roof, **coarse** has ~17 % headroom, **always-8** is
fundamentally cap-pressure-bound.

## 6. Where decode time goes (perf + strace + per-call diag)

Three diagnostic methods, all on config C, each gives a different
view that together explain the gap.

### (a) `perf record -F 99 -g`: top user-space symbols by CPU time

```
29.50%  streamllm_ext::plane_disk_to_kernel
        |-22.25% from MoEScheduler::on_install (one-time install cost)
        \- 7.23% from prefetch_worker_loop_      (decode hot path)
12.31%  __strcmp_evex  (gguf_find_key — install only)
```

**During decode** plane_disk_to_kernel takes ~7 % of CPU samples, not
the 13 % the chrono diag suggested — because workers spend a lot of
their wall time blocked, not running CPU.

### (b) `strace -fc`: syscall counts (config C)

```
% time     seconds  usecs/call     calls    syscall
 64.27   57.166119          74    770824    futex      ← inter-thread sync
 28.56   25.402452         372     68116    pread64
  7.09    6.305289       15530       406    mmap
```

**futex aggregate = 57 s** dominates pread (25 s). With 8 workers
plus the demand-path hook contending on `prefetch_mu_` + the
condition variables, the synchronization cost is **2.3 × the SSD
read cost**. This is invisible to chrono-based instrumentation
because workers wait on futex (not CPU-busy) — the chrono span
inside the worker stops at the dequeue but the futex burns wall
time on the locked thread.

### (c) `perf stat`: cache + scheduler effects

```
   552 G cycles · 1.51 T instructions · IPC 2.73   (good)
    L1-dcache miss rate: 18.2 %    (chunk-streaming evicts L1)
   cache-references miss rate: 4.79 %   (LLC pressure)
       1 027 812 context-switches in 116 s = 8 900/s   (worker thrashing)
       3 142 952 page-faults
```

8.9 K context-switches/sec means the kernel is preempting workers
~once every 113 µs on average. Combined with the futex traffic, this
is consistent with workers all racing for the same prefetch_mu_,
acquiring it for ~50 µs of work, getting preempted, releasing.

## 6.5 Ablation at full workload (458 prompt + 128 decode)

A short prompt (~5 tokens) doesn't saturate the 8 GB VRAM cap, so
the eviction-pressure regime is invisible. Run a 458-token wiki
prefill + 128-token decode at the same cap to expose it. All three
score strategies hit `peak_pool=8192 MB` (cap-saturated).

Decode TPS, four states × three strategies:

| state                       | always-8 | coarse | fine  |
|---|---:|---:|---:|
| **baseline (final, both reverted)** | **10.28** | **41.86** | **35.87** |
| + Phase 2 only              | 10.43    | 36.40  | 31.06 |
| + Phase 3 only              |  3.17    | 25.02  | 27.73 |
| + Phase 2 + Phase 3         |  3.67    | 25.85  | 29.59 |

Tier-hit rates in baseline:

| strategy | VRAM hit | DRAM hit | SSD miss |
|---|---:|---:|---:|
| always-8 | 16.8 % | **5.2 %** | 78.0 % |
| coarse   | 10.9 % | 0.3 %     | 88.8 % |
| fine     |  7.1 % | 0.1 %     | 92.8 % |

**Phase 3 is regressive at saturated cap.** The earlier "DRAM hit
0.0 %" measurement was at small-prompt runs that don't trigger
eviction. Under cap pressure, LRU-evicted chunks bounce back as
the prefill walks every layer's experts, and the host cache
absorbs those re-asks at ~36 µs/insert vs ~700 µs/re-pread. The
5.2 % DRAM hit rate at always-8 saves ~280 K SSD preads ≈ 200 s
of aggregate worker pread time.

**Phase 2 is also a net loss at this regime, despite cleaner per-
call diagnostics.** The xform target moved from a pageable scratch
to a pinned kernel-ring slot, eliminating the implicit CUDA
staging copy on `cudaMemcpyAsync`. But the host-cache populate
now reads from pinned memory into a pageable host vector — a
slower memcpy than pageable→pageable (≈ 2-3 GB/s vs ≈ 5+ GB/s).
At cap-saturated workloads the populate runs every miss, so the
slower memcpy is on the hot path. The pool_load_enq savings
don't recover the loss because pread dominates total wall.

**Both Phase 2 and Phase 3 reverted (commits eed1d573c and
2ed9571f2 superseded by fb9d5fdb5 + 23a93e83c).** The diagnostic
infrastructure (`DiagSpan`, `MoveSite`/`MoveEvent` counters, the
`should_cache_host` virtual) stays in place for future
experiments — only the regressive code paths are removed.

## 7. Updated reading (post score-policy refactor)

1. **Move_chunk overhead breakdown is invariant across strategies.**
   pread sits at 73–76 % of move_chunk wall regardless of policy or
   chunk count; xform 14–17 %, host_cache_insert 4 %, pool_load_enq
   3–6 %. The strategies move bytes; they don't move where the
   per-chunk software overhead lives.

2. **DRAM cache is dead at moderate cap.** `coarse` and `fine` both
   measured 0.0 % DRAM hit rate. Only `always-8` (cap-saturated)
   sees ~17 % DRAM hit because it churns. For non-saturated runs
   the host-cache populate is 100 % wasted work — Phase 3 lands
   directly on this.

3. **Score policy makes prefill cheap** when the table has a low
   floor. `fine` (floor=1 chunk) reaches **5.16 prefill TPS** at
   the 8 GB cap because most rank-7 tokens drop to the 1-chunk
   tier and per-expert MAX still lifts the few that share an
   expert with a high-score token. `always-8` is 1.24 prefill TPS
   for comparison — a 4× prefill gap purely from precision choice.

4. **Per-expert MAX is the right batching primitive.** The 7 %
   bump-up at rank 7 under `coarse` (mean_desired 2.07 vs floor 2)
   is exactly the savings: those tokens get high-precision compute
   without paying for an additional chunk load.

5. **SSD per-chunk latency is the same 700–770 µs across every
   strategy.** The bottleneck is QD-bound NVMe at this chunk size,
   not per-call CPU. Phase 4 dedup helps only if duplicate
   in-flight loads exist — current data shows `prefetch_skipped` ≈
   78 % already (most attempts find the chunk VRAM-resident), so
   cross-worker duplicate landings are likely small.

## 8. Updated experiment ladder (post-ablation)

The Phase 2 / 3 attempts taught us that small-prompt diagnostics
mislead about eviction-pressure regimes. New priorities, learned
from the wiki-workload ablation:

1. **Right-size the host LRU.** The DRAM cache is providing 5 %
   of attempts at always-8 (≈ 280 K reuses). The current populate
   path grows `host.chunks[p]` unbounded in pageable memory until
   `STREAMLLM_HOST_CAP_MB` triggers an LRU eviction. A sweep over
   `STREAMLLM_HOST_CAP_MB ∈ {8, 16, 32}` GB at the saturated
   workload should establish whether decode TPS lifts further as
   the LRU absorbs more re-asks.
2. **Phase 4 — in-flight (wid, cid) dedup** (still conditional).
   With `prefetch_skipped ≈ 78 %` absorbing duplicate intent at
   the LOAD-walk level, cross-worker duplicates are likely small.
   Add an `n_vram_idempotent_hit` counter first; promote only
   if it exceeds 5 %.
3. **Phase 6 — pre-transformed sidecar.** Eliminates xform's
   ~16 % of move_chunk wall entirely. Big lift (new GGUF format +
   encoder change) but the only remaining attack on the per-chunk
   software-overhead column. Worth doing only if the host-LRU
   sweep doesn't move TPS further.
4. **Phase 5 — thread-local scratch reuse on the cache-hit
   path.** Tiny effect; skip unless touching that code anyway.

The previously-listed Phase 2 (pinned kernel ring) and Phase 3
(skip host-cache populate) stay implemented in the diagnostic
sense (their counters remain) but their code paths are reverted
because both regressed at the saturated-cap workload.

## 9. Single-line takeaway

```
T_token = max( B_ssd × p_SSD / BW_SSD , … )
        + N_chunks × ( per_chunk_xform + per_chunk_sync_wait + … )
```

The score policy controls `B_ssd × p_SSD` (= bytes/tok × miss rate
× per-chunk size). It does NOT touch the additive per-chunk
overhead — Phases 2/3 do. To break beyond ~10 TPS decode at this
NVMe + 8 GB cap, both knobs must move.
