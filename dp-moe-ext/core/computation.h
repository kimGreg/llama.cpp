// DPMoE / core — ChunkedComputation abstract base class.
//
// One of the framework's two encoder/architecture-blind ABCs (the
// other is ChunkedTensor in chunked_tensor.h). A ChunkedComputation
// represents one kernel-shaped unit of work that consumes one or
// more ChunkedTensors and writes a result. Concrete subclasses live
// in ``decoder/`` (kernel impls) or ``model/`` (architecture-fused
// kernels).
//
// Lifecycle (driven by Runtime::run):
//
//   1. ``state_key``  — stable identity used by the runtime to cache
//                       per-comp state (pinned buffers, ready_event,
//                       cached ComputationInput pointer).
//
//   2. ``pre_inputs`` — captured D2H copies that must complete before
//                       ``plan`` can run. Each MemcpySpec describes
//                       one device→pinned-host transfer; supports 1D
//                       and 2D-pitched copies. Returns empty for
//                       computations that need no device-side input
//                       on host.
//
//   3. ``plan``       — pure host function. Reads its pinned input
//                       buffers; fills its pinned output buffers;
//                       returns ChunkPlan { evict_set, load_set }
//                       declaring chunk movements. NO CUDA API CALLS
//                       allowed in this body — the runtime invokes it
//                       from inside a cudaLaunchHostFunc node, which
//                       forbids CUDA calls per the CUDA Programming
//                       Guide. Default returns empty plan.
//
//   4. ``execute``    — launch the kernel(s). Captured. By entry:
//                       pre_inputs landed, plan() ran, all chunks in
//                       load_set are resident with pointers updated,
//                       replay reservations include them. Free to
//                       issue captured H2D for any control data.
//
// The framework knows nothing about what a kernel does. New kernel
// kinds (quantized matmul, fused MoE, fused attention, KV cache
// attention, etc.) plug in as new ChunkedComputation subclasses.

#pragma once

#include "vram_pool.h"   // StreamHandle

#include <string>
#include <vector>

namespace dp_moe_ext {

class ChunkedTensor;

// Heterogeneous input/output payload. Concrete computations cast
// these to their own struct types via static_cast or define a
// concrete subclass to carry their inputs. The base class is empty
// so every kernel kind defines its own contract.
//
// Convention: keep these small (just pointers + scalar params); no
// vtable on the payloads themselves beyond the virtual dtor.
struct ComputationInput  { virtual ~ComputationInput()  = default; };
struct ComputationOutput { virtual ~ComputationOutput() = default; };

// ``ChunkKey`` is defined in vram_pool.h (already used by the pool's
// residency map). ChunkPlan reuses that type so loader-side callers
// see the same identifier shape.

// One captured D2H or H2D copy. host_dst points into pinned host
// memory owned by the calling ChunkedComputation (typically allocated
// at construction with cudaMallocHost). The runtime emits one
// cudaMemcpyAsync (or cudaMemcpy2DAsync if is_2d) per spec on the
// stream passed to Runtime::run.
struct MemcpySpec {
    const void * device_src = nullptr;
    void *       host_dst   = nullptr;
    size_t       bytes      = 0;     // 1D byte count, OR width_bytes for 2D
    bool         is_2d      = false;
    size_t       src_pitch  = 0;     // 2D only
    size_t       dst_pitch  = 0;     // 2D only
    size_t       height     = 0;     // 2D only
};

// Result of plan(): the chunks the loader thread should evict /
// load before the captured kernel runs.  Either set may be empty.
//
// ``required_set`` (Milestone 1 Step 6) is the *superset* the kernel
// will read on this dispatch — load_set ∪ (chunks already resident
// and reused on this layer). The executor's pre-launch validation
// walks this set asserting every (wid, cid) is at
// ``ChunkState::POINTER_TABLE_READY`` before the fused MoE kernel
// fires. Empty for hooks that don't need the check.
struct ChunkPlan {
    std::vector<ChunkKey> evict_set;
    std::vector<ChunkKey> load_set;
    std::vector<ChunkKey> required_set;
};

// Kind enum lets the dispatch path short-circuit the canonical
// lifecycle when a subclass doesn't need every step. Today's MoE
// path is LoadCompute (the default). PipelinedMoEMatMulComp would be
// Custom — it owns its own orchestration via ``run_custom`` instead
// of the three simple-lifecycle virtuals.
enum class HookKind {
    Load,         // copy_stream work only; no compute. Used by
                  // speculative-prefetch hooks.
    Compute,      // compute_stream work only; no chunk movement.
                  // Used by hand-fused subgraph replacements.
    LoadCompute,  // simple lifecycle (pre_inputs + plan + execute).
                  // The canonical case; today's MoEMatMulComp.
    Sync,         // event record / wait, no kernels.
    Observe,      // host-side state update, no GPU work.
    Custom,       // fully-custom callback via ``run_custom``.
};

struct SchedulerContext {
    // Reserved for the framework — pinned-buffer accessors, pool
    // references, residency-tracker handles, etc.  Filled in as
    // P6/P7 subclasses (PipelinedMoEMatMulComp, KvAttnComp) define
    // what they actually need.  Today's MoEMatMulComp doesn't use
    // run_custom so this stays empty.
};

class ChunkedComputation {
public:
    virtual ~ChunkedComputation() = default;

    // Stable identity used by the runtime to cache per-comp state.
    // Typical implementation: return the canonical tensor name (e.g.
    // ``blk.0.ffn_gate_exps.weight``).
    virtual std::string state_key() const = 0;

    // Hook flavour the dispatch path uses to route this comp.
    // Default is LoadCompute — pre_inputs / plan / execute lifecycle.
    // Subclasses that own their orchestration (pipelined MoE, KV
    // attention) override to Custom and implement ``run_custom``.
    virtual HookKind kind() const { return HookKind::LoadCompute; }

    // Captured D2H copies that must complete BEFORE plan() can run.
    // Default: no inputs needed.
    virtual std::vector<MemcpySpec> pre_inputs(
        const ComputationInput & /*in*/) {
        return {};
    }

    // Pure host function. NO CUDA API CALLS. Subclasses fill their
    // own pinned output buffers as a side effect (e.g., per-(t,u)
    // precision array the kernel will read). Default: empty plan.
    virtual ChunkPlan plan(const ComputationInput & /*in*/) {
        return {};
    }

    // Run the kernel.  Caller (the dispatch shim) has waited on each
    // loaded chunk's ready event on ``stream`` after the planner
    // completed, so all chunks in plan().load_set are resident with
    // their pointer-table entries updated. The subclass casts
    // ``ComputationInput`` and ``ComputationOutput`` to whatever
    // concrete payload it expects and launches the kernel(s).
    //
    // Default = required for LoadCompute subclasses.  Subclasses
    // overriding ``run_custom`` can leave this as a no-op
    // (the dispatch path won't call execute on Custom comps).
    virtual void execute(const ComputationInput & /*inputs*/,
                          ComputationOutput &      /*out*/,
                          StreamHandle             /*stream*/) {}

    // Custom callback form. Used by subclasses whose dispatch doesn't
    // fit the canonical lifecycle — pipelining, iteration-level
    // scheduling, anything that wants to interleave loads and
    // launches its own way.  Default falls through to the simple
    // lifecycle (pre_inputs + plan + execute orchestrated by the
    // caller) so a subclass can opt in by overriding only the
    // virtuals it needs.
    virtual void run_custom(const ComputationInput & /*in*/,
                             ComputationOutput &      /*out*/,
                             StreamHandle             /*stream*/,
                             SchedulerContext &       /*ctx*/) {}
};

}  // namespace dp_moe_ext
