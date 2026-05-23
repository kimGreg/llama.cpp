# streamllm-ext — StreamLLM runtime grafted onto llama.cpp.
#
# Enable with:
#   cmake -B build -DGGML_CUDA=ON -DSTREAMLLM_EXT=ON ...
#
# Layout:
#   core/             framework (vram_pool, runtime, ggml-cuda hook, scheduler trait)
#   decoder/anybcq/   AnyBCQ kernels and chunk format
#   moe/common/       shared MoE runtime utilities (scheduler, matmul_comp, fused
#                     kernels, dispatch, residency, runtime_glue) — architecture-neutral
#   moe/qwen3/        Qwen3-MoE executor
#   moe/deepseek_moe/ DeepSeek-MoE V1 executor (stub today; delegates to common base)
#   moe/gemma_4/      Gemma 4 MoE executor (stub today; delegates to common base)
#   moe/qwen3_6/      placeholder for Qwen3.6-MoE — reuses qwen3_moe at runtime
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
    core/executor.cpp
    decoder/chunked_tensor_factory.cpp
    moe/common/moe_scheduler.cpp
    decoder/anybcq/upstream_layout.cpp
    decoder/anybcq/tensor.cpp
    decoder/direct_matrix/tensor.cpp
    decoder/ss_anybcq/upstream_layout.cpp
)

# ggml-cuda hook glue. Requires GGML_CUDA=ON (the hook signature takes
# a cudaStream_t); CPU-only builds skip it and the loader stub never fires.
# moe/common/dispatch.cpp holds the actual hook bodies (called by the
# scheduler) and likewise needs the CUDA driver headers.  Per-arch
# executor files (moe/{qwen3,deepseek_moe,gemma_4}/executor.cpp) are
# added at the end; the common base class lives in moe/qwen3/executor.cpp
# today and is subclassed by the per-arch files.
if (GGML_CUDA)
    list(APPEND STREAMLLM_EXT_SOURCES
        moe/common/runtime_glue.cpp
        moe/common/dispatch.cpp
        moe/common/matmul_comp.cpp
        moe/common/moe_executor.cpp
        moe/qwen3/qwen3_moe_executor.cpp
        moe/deepseek_moe/deepseek_moe_executor.cpp
    )
    if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/moe/gemma_4/gemma4_moe_executor.cpp")
        list(APPEND STREAMLLM_EXT_SOURCES
            moe/gemma_4/gemma4_moe_executor.cpp
        )
        add_compile_definitions(STREAMLLM_HAVE_GEMMA4_EXECUTOR=1)
    endif()
endif()

# CUDA sources. naver_kernel_copy.cu is a verbatim copy of NAVER's
# nqmv_bias kernel: the upstream anybcq.cu pulls in <torch/extension.h>
# transitively, so copying the kernel avoids a libtorch dependency.
if (GGML_CUDA)
    enable_language(CUDA)
    list(APPEND STREAMLLM_EXT_SOURCES
        core/vram_pool.cu
        core/pointer_patch.cu
        decoder/anybcq/kernels/chunked_matmul_gemv.cu
        decoder/anybcq/kernels/chunked_matmul_gemm.cu
        decoder/anybcq/chunked_matmul.cu
        decoder/ss_anybcq/chunked_matmul.cu
        moe/common/fused_kernels.cu
    )
endif()

add_library(streamllm_ext STATIC ${STREAMLLM_EXT_SOURCES})

target_include_directories(streamllm_ext PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${CMAKE_CURRENT_SOURCE_DIR}/decoder/anybcq
    ${CMAKE_CURRENT_SOURCE_DIR}/decoder/anybcq/kernels
    ${CMAKE_CURRENT_SOURCE_DIR}/decoder/direct_matrix
    ${CMAKE_CURRENT_SOURCE_DIR}/decoder/ss_anybcq
    ${CMAKE_CURRENT_SOURCE_DIR}/moe/common
    ${CMAKE_CURRENT_SOURCE_DIR}/moe/qwen3
    ${CMAKE_CURRENT_SOURCE_DIR}/moe/deepseek_moe
    ${CMAKE_CURRENT_SOURCE_DIR}/moe/gemma_4
    ${CMAKE_CURRENT_SOURCE_DIR}/moe/qwen3_6
)

# Link against ggml for gguf_* symbols. The full `llama` lib isn't
# needed by M2–M3; M4 pulls it in for model loading + backend scheduling.
target_link_libraries(streamllm_ext PUBLIC ggml)

# M4b-integration: qwen3_runtime_glue.cpp (plain C++) needs <cuda_runtime.h>
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
    target_sources(streamllm_ext PRIVATE core/runtime_diag.cpp)
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

# streamllm-loader-gate-test: synthetic-failure test for Milestone 1
# Step 1's required_runtime + executor gate. Builds three minimal
# in-memory GGUFs (pre-gate / required-ok / required-bad), drives
# them through StreamReader::from_gguf + the executor registry
# lookup, and asserts the gate semantics. No CUDA needed.
add_executable(streamllm-loader-gate-test test_loader_gate.cpp)
target_link_libraries(streamllm-loader-gate-test PRIVATE streamllm_ext ggml)

# streamllm-sentinel-compute-poc: reference POC kept under
# tests/reference/ for re-validating sentinel mechanics against
# future ggml-cuda hook-surface changes. Builds a synthetic
# compute-fed sentinel cgraph (W*x → aux ops → sentinel →
# consumer) and verifies the seven properties documented in
# docs/MODE_A_MILESTONE1.md §Q3. Not part of the production
# runtime; no production path links against it.
if (GGML_CUDA)
    add_executable(streamllm-sentinel-compute-poc
        tests/reference/test_sentinel_compute_poc.cpp)
    target_link_libraries(streamllm-sentinel-compute-poc
        PRIVATE streamllm_ext ggml ggml-cuda CUDA::cudart)
endif()

# streamllm-weighted-reduce-test: correctness test for M1 cutover S4's
# launch_weighted_reduce_slots kernel. CPU reference + GPU kernel +
# max_abs_diff < 1e-5 across multiple shape cases (Qwen3-30B-A3B
# decode shapes, M-not-multiple-of-tile boundary, n_used=1 edge,
# n_used=16). Pure CUDA test; no model load.
if (GGML_CUDA)
    add_executable(streamllm-weighted-reduce-test test_weighted_reduce.cpp)
    target_link_libraries(streamllm-weighted-reduce-test
        PRIVATE streamllm_ext CUDA::cudart)
endif()

# streamllm-swiglu-mul-test: correctness test for M1 cutover S6's
# launch_swiglu_mul kernel. CPU reference + GPU kernel + max_abs_diff
# tolerance check across Qwen3-30B-A3B decode shapes plus a small
# known-answer block (silu(0)=0, silu(±10)≈{0,10}, silu(1)≈0.731...).
# Pure CUDA test; no model load.
if (GGML_CUDA)
    add_executable(streamllm-swiglu-mul-test test_swiglu_mul.cpp)
    target_link_libraries(streamllm-swiglu-mul-test
        PRIVATE streamllm_ext CUDA::cudart)
endif()

# streamllm-router-equiv-test: equivalence test for M1 cutover S3's
# llm_build_moe_routing_softmax_topk helper. Builds the router
# subgraph two ways (helper + hand-rolled reference emitting the
# same raw ggml ops in the same order) into the same ggml_context,
# computes on CUDA backend, and asserts ids byte-for-byte and
# probs/weights bit-exact identical. Catches any drift between
# the helper and its documented op sequence. Needs the llama-graph.h
# declaration of the helper — include path widened for this target.
if (GGML_CUDA)
    add_executable(streamllm-router-equiv-test test_router_equiv.cpp)
    target_include_directories(streamllm-router-equiv-test
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
    target_link_libraries(streamllm-router-equiv-test
        PRIVATE streamllm_ext ggml ggml-cuda llama CUDA::cudart)
endif()

if (GGML_CUDA)
    find_package(CUDAToolkit REQUIRED)
    # streamllm-gpu-test: end-to-end kernel check (GGUF → VramChunkPool
    # → NAVER LUT-GEMV → compare Y with a Python reference).
    add_executable(streamllm-gpu-test test_gpu_kernel.cpp)
    target_link_libraries(streamllm-gpu-test PRIVATE streamllm_ext ggml CUDA::cudart)

    # streamllm-microbench: μs/call + GB/s for naver_gemv_launch and
    # VramChunkPool::load across representative tensor shapes.
    add_executable(streamllm-microbench test_microbench.cpp)
    target_link_libraries(streamllm-microbench PRIVATE streamllm_ext ggml CUDA::cudart)
endif()
