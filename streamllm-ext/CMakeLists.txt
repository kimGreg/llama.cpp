# streamllm-ext — StreamLLM runtime grafted onto llama.cpp.
#
# Enable with:
#   cmake -B build -DGGML_CUDA=ON -DSTREAMLLM_EXT=ON ...
#
# Layout:
#   core/            framework (vram_pool, runtime, ggml-cuda hook, scheduler trait)
#   decoder/anybcq/  AnyBCQ kernels and chunk format
#   qwen3/           Qwen3-MoE scheduler and cache policy
cmake_minimum_required(VERSION 3.18)

message(STATUS "streamllm-ext: enabled")

if (NOT GGML_CUDA)
    message(WARNING
        "streamllm-ext: -DGGML_CUDA=OFF — extension will build but the "
        "CUDA compute path will be unavailable.")
endif()

set(STREAMLLM_EXT_SOURCES
    core/stream_reader.cpp
    core/runtime.cpp
    qwen3/qwen3_moe_scheduler.cpp
    decoder/anybcq/upstream_layout.cpp
)

# ggml-cuda hook glue. Requires GGML_CUDA=ON (the hook signature takes
# a cudaStream_t); CPU-only builds skip it and the loader stub never fires.
# qwen3_moe_dispatch.cpp holds the actual hook bodies (called by the
# scheduler) and likewise needs the CUDA driver headers.
if (GGML_CUDA)
    list(APPEND STREAMLLM_EXT_SOURCES
        core/runtime_hook.cpp
        qwen3/qwen3_moe_dispatch.cpp
    )
endif()

# CUDA sources. naver_kernel_copy.cu is a verbatim copy of NAVER's
# nqmv_bias kernel: the upstream anybcq.cu pulls in <torch/extension.h>
# transitively, so copying the kernel avoids a libtorch dependency.
if (GGML_CUDA)
    enable_language(CUDA)
    list(APPEND STREAMLLM_EXT_SOURCES
        core/vram_pool.cu
        decoder/anybcq/naver_gemv.cu
        decoder/anybcq/naver_kernel_copy.cu
        decoder/anybcq/naver_gemm_batched.cu
        decoder/anybcq/cast_f16.cu
        decoder/anybcq/dequant_planes.cu
        decoder/anybcq/batched_gemm.cu
        decoder/anybcq/per_plane_layout.cu
        decoder/anybcq/chunked_matmul_anybcq.cu
        decoder/anybcq/moe_fused.cu
    )
endif()

add_library(streamllm_ext STATIC ${STREAMLLM_EXT_SOURCES})

target_include_directories(streamllm_ext PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${CMAKE_CURRENT_SOURCE_DIR}/decoder/anybcq
    ${CMAKE_CURRENT_SOURCE_DIR}/qwen3
)

# Link against ggml for gguf_* symbols. The full `llama` lib isn't
# needed by M2–M3; M4 pulls it in for model loading + backend scheduling.
target_link_libraries(streamllm_ext PUBLIC ggml)

# M4b-integration: runtime_hook.cpp (plain C++) needs <cuda_runtime.h>
# and links against cudart for cudaDeviceSynchronize; also calls the
# ggml-cuda set-hook symbol, so link ggml-cuda too.
if (GGML_CUDA)
    find_package(CUDAToolkit REQUIRED)
    # cuBLAS is used by the batched prefill fast path (batched_gemm.cu).
    # CUDA::cublas is always present in CUDAToolkit and already loaded
    # by ggml-cuda in-process, so the extra link-time dep is trivial.
    target_link_libraries(streamllm_ext PUBLIC
        CUDA::cudart
        CUDA::cublas
        ggml-cuda)
endif()

set_target_properties(streamllm_ext PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# NVTX annotations for Nsight Systems traces. Off by default (header
# is no-op). Enable via cmake -DSTREAMLLM_NVTX=ON. We just need the
# header — nsys auto-discovers nvtxRangePush/Pop without a libnvtx
# link.
option(STREAMLLM_NVTX "Compile with NVTX range annotations" OFF)
if (STREAMLLM_NVTX)
    target_compile_definitions(streamllm_ext PUBLIC STREAMLLM_NVTX)
    message(STATUS "streamllm-ext: NVTX annotations ON")
endif()

# Cache-stack diagnostic counters: per-rank / per-plane tier-hit
# buckets, install-log tier-hit summary, optional STREAMLLM_TRACE_FILE
# chunk-request recorder for offline Belady simulation. Pure
# measurement infrastructure — gated off by default to keep the hot
# path lean. Enable for tuning rounds via cmake -DSTREAMLLM_DIAG=ON.
option(STREAMLLM_DIAG "Compile with cache-stack diagnostic counters" OFF)
if (STREAMLLM_DIAG)
    target_compile_definitions(streamllm_ext PUBLIC STREAMLLM_DIAG)
    target_sources(streamllm_ext PRIVATE core/runtime_hook_diag.cpp)
    message(STATUS "streamllm-ext: diagnostics ON")
endif()

# Enable F16C intrinsics on x86-64 for hardware-accelerated fp32↔fp16
# (upstream_layout uses _mm_cvtps_ph for bit-exact IEEE 754 RNE matching
# numpy / PyTorch). No-op on ARM — the portable fallback kicks in.
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-mf16c HAS_MF16C)
if (HAS_MF16C)
    target_compile_options(streamllm_ext PRIVATE -mf16c)
endif()

# NAVER's kernel uses native half-operator syntax (a + b for __half).
# ggml's CUDA build leaves these enabled by default; the vendored
# NAVER kernel doesn't compile otherwise.
if (GGML_CUDA)
    target_compile_options(streamllm_ext PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            -U__CUDA_NO_HALF_OPERATORS__
            -U__CUDA_NO_HALF_CONVERSIONS__
            -U__CUDA_NO_HALF2_OPERATORS__
            -U__CUDA_NO_BFLOAT16_CONVERSIONS__>
    )
    # CUDA_SEPARABLE_COMPILATION OFF is intentional: each .cu is
    # self-contained (NAVER kernel body + launcher + pool — no
    # cross-TU device calls), and leaving it ON dragged relocatable
    # device code into libllama.so, which broke linking with
    # ``undefined reference to __cudaRegisterLinkedBinary_*``.
    set_target_properties(streamllm_ext PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
    )
endif()

# streamllm-gguf-dump: reads a GGUF, prints streamllm.* metadata,
# validates byte accounting against the chunked layout.
add_executable(streamllm-gguf-dump streamllm_gguf_dump.cpp)
target_link_libraries(streamllm-gguf-dump PRIVATE streamllm_ext ggml)

# streamllm-upstream-test: byte-for-byte parity check of the host-side
# upstream-layout builder against Python reference dumps.
add_executable(streamllm-upstream-test test_upstream_layout.cpp)
target_link_libraries(streamllm-upstream-test PRIVATE streamllm_ext ggml)

if (GGML_CUDA)
    find_package(CUDAToolkit REQUIRED)
    # streamllm-gpu-test: end-to-end kernel check (GGUF → VramChunkPool
    # → NAVER LUT-GEMV → compare Y with a Python reference).
    add_executable(streamllm-gpu-test test_gpu_kernel.cpp)
    target_link_libraries(streamllm-gpu-test PRIVATE streamllm_ext ggml CUDA::cudart)

    # streamllm-runtime-test: spot-checks kernel output for a handful
    # of named tensors against Python reference dumps.
    add_executable(streamllm-runtime-test test_runtime_full.cpp)
    target_link_libraries(streamllm-runtime-test PRIVATE streamllm_ext ggml CUDA::cudart)

    # streamllm-microbench: μs/call + GB/s for naver_gemv_launch and
    # VramChunkPool::load across representative tensor shapes.
    add_executable(streamllm-microbench test_microbench.cpp)
    target_link_libraries(streamllm-microbench PRIVATE streamllm_ext ggml CUDA::cudart)

    # streamllm-moe-hook-test: confirms ggml_cuda_set_mul_mat_id_hook
    # accepts our function pointer and is null-safe.
    add_executable(streamllm-moe-hook-test test_moe_hook.cpp)
    target_link_libraries(streamllm-moe-hook-test PRIVATE streamllm_ext ggml ggml-cuda CUDA::cudart)
endif()

