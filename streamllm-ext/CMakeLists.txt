# streamllm-ext — the runtime half of StreamLLM grafted onto llama.cpp.
#
# Enable with:
#   cmake -B build -DGGML_CUDA=ON -DSTREAMLLM_EXT=ON ...
#
# Milestone layout (filled in progressively):
#
#   M2: stream_reader.{h,cpp}      — GGUF metadata + byte layout parser
#                                    streamllm_gguf_dump (CLI)
#   M3: vram_pool.{h,cpp},
#       upstream_layout.{h,cpp}    — chunked VRAM pool + NAVER format builder
#   M4: custom_op.{h,cpp}          — GGML_OP_STREAMLLM_MATMUL registration,
#                                    links in CUDA kernel sources
#   M5: scheduler.{h,cpp}          — Lazy / LayerPrefetch / Budgeted
#
# CUDA kernel sources come from the parent repo:
#   ../../../streamllm/runtime/native/csrc/kernels.cu
#   ../../../streamllm/runtime/native/csrc/anybcq_upstream_launch.cu
# and from the AnyBCQ vendor:
#   ../../anybcq/anybcq/inference/custom_kernel/anybcq.cu

cmake_minimum_required(VERSION 3.18)

message(STATUS "streamllm-ext: enabled")

# Resolve absolute paths to the parent repo (two levels up from this fork).
get_filename_component(_STREAMLLM_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
set(STREAMLLM_KERNEL_DIR "${_STREAMLLM_REPO_ROOT}/streamllm/runtime/native/csrc")
set(ANYBCQ_KERNEL_DIR    "${_STREAMLLM_REPO_ROOT}/third-party/anybcq/anybcq/inference/custom_kernel")

if (NOT EXISTS "${STREAMLLM_KERNEL_DIR}/kernels.cu")
    message(FATAL_ERROR
        "streamllm-ext: cannot find kernels.cu at ${STREAMLLM_KERNEL_DIR}\n"
        "Expected parent repo layout: <parent>/streamllm/runtime/native/csrc/. "
        "This fork must be vendored under <parent>/third-party/llama.cpp/.")
endif()
if (NOT EXISTS "${ANYBCQ_KERNEL_DIR}/anybcq.cu")
    message(FATAL_ERROR
        "streamllm-ext: cannot find NAVER kernel at ${ANYBCQ_KERNEL_DIR}. "
        "Make sure third-party/anybcq/ is vendored alongside this fork.")
endif()

# CUDA is required for the real compute path (M4 onward). M2–M3 don't
# need it yet — only M4's custom op pulls in the .cu sources. Warn if
# off, don't block.
if (NOT GGML_CUDA)
    message(WARNING
        "streamllm-ext: -DGGML_CUDA=OFF — extension will build but the "
        "CUDA compute path (M4+) will be unavailable.")
endif()

# --- Core library: M2 + M3 host-side + M4a/b runtime sources -------------
set(STREAMLLM_EXT_SOURCES
    stream_reader.cpp
    upstream_layout.cpp
    runtime.cpp
    scheduler.cpp
)

# M4b-integration: glue between llama.cpp's model load + ggml-cuda's
# mul_mat dispatch. Requires GGML_CUDA=ON (the hook signature takes a
# cudaStream_t); on CPU-only builds this file is skipped and the model
# loader's stub simply never fires.
if (GGML_CUDA)
    list(APPEND STREAMLLM_EXT_SOURCES runtime_hook.cpp)
endif()

# M4a: CUDA sources — VRAM pool + NAVER kernel launcher. Requires
# -DGGML_CUDA=ON (enables CMake's CUDA language). Without CUDA, these
# are omitted and any code trying to instantiate VramChunkPool gets a
# link error (documented in vram_pool.h).
#
# naver_kernel_copy.cu is a verbatim copy of NAVER's nqmv_bias kernel
# with attribution preserved. We can't compile the upstream
# third-party/anybcq/.../anybcq.cu directly because it transitively
# includes <torch/extension.h> and <ATen/ATen.h> via its datatype.h /
# typetraits.h helpers (for ATen-typed utilities the kernel body
# doesn't use). Copying the kernel avoids dragging libtorch into a
# llama.cpp fork that otherwise has no torch dependency.
if (GGML_CUDA)
    enable_language(CUDA)
    list(APPEND STREAMLLM_EXT_SOURCES
        vram_pool.cu
        naver_gemv.cu
        naver_kernel_copy.cu
        naver_gemm_batched.cu
        cast_f16.cu
        dequant_planes.cu
        batched_gemm.cu
    )
endif()

add_library(streamllm_ext STATIC ${STREAMLLM_EXT_SOURCES})

target_include_directories(streamllm_ext PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
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

# --- CLI: streamllm-gguf-dump --------------------------------------------
# Reads a GGUF, prints streamllm.* metadata, validates byte accounting.
# Useful for verifying the Python encoder's artifacts during M2–M4
# bring-up without needing a full model-load pipeline.
add_executable(streamllm-gguf-dump streamllm_gguf_dump.cpp)
target_link_libraries(streamllm-gguf-dump PRIVATE streamllm_ext ggml)

# --- Test: streamllm-upstream-test (M3 parity check) ---------------------
# Runs the host-side upstream-layout builder on a GGUF and compares the
# output byte-for-byte against a Python reference dump produced by
# streamllm/scripts/dump_upstream_layout.py.
add_executable(streamllm-upstream-test test_upstream_layout.cpp)
target_link_libraries(streamllm-upstream-test PRIVATE streamllm_ext ggml)

# --- Test: streamllm-gpu-test (M4a end-to-end kernel check) --------------
# Reads a GGUF, builds the upstream layout host-side, uploads into a
# VramChunkPool, runs NAVER's LUT-GEMV kernel, compares Y with a
# Python reference produced by streamllm/scripts/dump_gpu_reference.py.
if (GGML_CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_executable(streamllm-gpu-test test_gpu_kernel.cpp)
    target_link_libraries(streamllm-gpu-test PRIVATE streamllm_ext ggml CUDA::cudart)

    # --- Test: streamllm-runtime-test (M4b data-plane check) -------------
    # Loads every managed tensor of a GGUF into a StreamllmRuntime, then
    # spot-checks kernel output for a handful of named tensors against
    # Python reference dumps.
    add_executable(streamllm-runtime-test test_runtime_full.cpp)
    target_link_libraries(streamllm-runtime-test PRIVATE streamllm_ext ggml CUDA::cudart)

    # --- Microbench: per-primitive timing --------------------------------
    # Times naver_gemv_launch (kernel) and VramChunkPool::load (move_chunk
    # RAM→VRAM) across a few representative tensor shapes. Reports
    # μs/call and effective GB/s — the two numbers that set the floor
    # on every scheduler policy's theoretical throughput.
    add_executable(streamllm-microbench test_microbench.cpp)
    target_link_libraries(streamllm-microbench PRIVATE streamllm_ext ggml CUDA::cudart)
endif()

message(STATUS "streamllm-ext: KERNEL_DIR=${STREAMLLM_KERNEL_DIR}")
message(STATUS "streamllm-ext: ANYBCQ_KERNEL_DIR=${ANYBCQ_KERNEL_DIR}")
