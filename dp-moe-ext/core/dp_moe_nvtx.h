// DPMoE — NVTX range helpers for Nsight Systems profiling.
//
// Compile-time toggled by ``-DDP_MOE_NVTX``: with NVTX off the
// macros are no-ops; with NVTX on they push/pop named ranges on the
// host thread, which Nsight Systems renders as labelled bars in the
// CPU/GPU timeline (compute_stream and copy_stream activity already
// gets traced automatically by ``nsys profile --trace=cuda``).

#pragma once

#if defined(DP_MOE_NVTX)
#  include <nvtx3/nvToolsExt.h>

namespace dp_moe_ext {

struct NvtxRange {
    NvtxRange(const char * name) { nvtxRangePushA(name); }
    ~NvtxRange()                  { nvtxRangePop(); }
    NvtxRange(const NvtxRange &)            = delete;
    NvtxRange & operator=(const NvtxRange &) = delete;
};

inline void nvtx_mark(const char * name) { nvtxMarkA(name); }

}  // namespace dp_moe_ext

#  define DP_MOE_NVTX_CONCAT_(a, b) a##b
#  define DP_MOE_NVTX_CONCAT(a, b)  DP_MOE_NVTX_CONCAT_(a, b)
#  define DP_MOE_NVTX_RANGE(name)   ::dp_moe_ext::NvtxRange  \
        DP_MOE_NVTX_CONCAT(_nvtx_range_, __LINE__){name}
#  define DP_MOE_NVTX_MARK(name)    ::dp_moe_ext::nvtx_mark(name)

#else  // !DP_MOE_NVTX

#  define DP_MOE_NVTX_RANGE(name) ((void)0)
#  define DP_MOE_NVTX_MARK(name)  ((void)0)

#endif
