// streamllm-ext / core — ChunkedComputation abstract base class.
//
// One of the framework's two encoder/architecture-blind ABCs (the
// other is ChunkedTensor in chunked_tensor.h). A ChunkedComputation
// represents one kernel-shaped unit of work that consumes one or
// more ChunkedTensors and writes a result. Concrete subclasses live
// in ``decoder/`` (kernel impls) or ``model/`` (architecture-fused
// kernels).
//
// Two responsibilities:
//
//   1. ``demand``  — at plan-time, declare which chunks (from which
//                    ChunkedTensors) need to be resident before the
//                    kernel runs. The scheduler reads this to issue
//                    async loads; the runtime reads it to decide
//                    which residency events to wait on. Multiple
//                    ChunkedTensors can be declared, which is how
//                    fused kernels declare their multi-tensor
//                    demand.
//
//   2. ``execute`` — run the kernel. By the time this is called the
//                    runtime has waited on each declared chunk's
//                    ready_event on ``stream``. The subclass casts
//                    ``ComputationInput`` and ``ComputationOutput``
//                    to whatever concrete payload it expects and
//                    launches the kernel.
//
// The framework knows nothing about what a kernel does. New kernel
// kinds (quantized matmul, fused attention, KV cache attention,
// etc.) plug in as new ChunkedComputation subclasses.

#pragma once

#include "vram_pool.h"   // StreamHandle

#include <vector>

namespace streamllm_ext {

class ChunkedTensor;

// Heterogeneous input/output payload. Concrete computations cast
// these to their own struct types via static_cast or define a
// concrete subclass to carry their inputs. The base class is empty
// so every kernel kind defines its own contract.
//
// Convention: keep these small (just pointers + scalar params); no
// vtable on the payloads themselves.
struct ComputationInput  { virtual ~ComputationInput()  = default; };
struct ComputationOutput { virtual ~ComputationOutput() = default; };

class ChunkedComputation {
public:
    virtual ~ChunkedComputation() = default;

    // What chunks does this computation need, from which tensors?
    // Returned at plan time.  The scheduler may issue loads ahead
    // of time based on graph lookahead; the runtime ensures
    // residency at execute time as a safety net.
    struct ChunkDemand {
        ChunkedTensor *  tensor;
        std::vector<int> cids;   // chunk indices into ``tensor``
    };
    virtual std::vector<ChunkDemand>
    demand(const ComputationInput & inputs) const = 0;

    // Run the kernel.  Caller (Runtime::run) has waited on every
    // declared chunk's ready_event on ``stream``.  The subclass
    // launches its kernel(s) on ``stream``; everything else is
    // private detail.
    virtual void execute(const ComputationInput & inputs,
                          ComputationOutput &      out,
                          StreamHandle             stream) = 0;
};

}  // namespace streamllm_ext
