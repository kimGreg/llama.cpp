// streamllm-ext — NVTX range helpers for Nsight Systems profiling.
//
// Compile-time toggled by ``-DSTREAMLLM_NVTX``: with NVTX off the
// macros are no-ops; with NVTX on they push/pop named ranges on the
// host thread, which Nsight Systems renders as labelled bars in the
// CPU/GPU timeline (compute_stream and copy_stream activity already
// gets traced automatically by ``nsys profile --trace=cuda``).

#pragma once

#if defined(STREAMLLM_NVTX)
#  include <nvtx3/nvToolsExt.h>

namespace streamllm_ext {

struct NvtxRange {
    NvtxRange(const char * name) { nvtxRangePushA(name); }
    ~NvtxRange()                  { nvtxRangePop(); }
    NvtxRange(const NvtxRange &)            = delete;
    NvtxRange & operator=(const NvtxRange &) = delete;
};

inline void nvtx_mark(const char * name) { nvtxMarkA(name); }

}  // namespace streamllm_ext

#  define STLM_NVTX_CONCAT_(a, b) a##b
#  define STLM_NVTX_CONCAT(a, b)  STLM_NVTX_CONCAT_(a, b)
#  define STLM_NVTX_RANGE(name)   ::streamllm_ext::NvtxRange  \
        STLM_NVTX_CONCAT(_nvtx_range_, __LINE__){name}
#  define STLM_NVTX_MARK(name)    ::streamllm_ext::nvtx_mark(name)

#else  // !STREAMLLM_NVTX

#  define STLM_NVTX_RANGE(name) ((void)0)
#  define STLM_NVTX_MARK(name)  ((void)0)

#endif
