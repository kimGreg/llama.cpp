# streamllm-gate-trace — capture per-(layer, token) MoE gate-score
# trace from a reference (typically BF16) model via cb_eval.

set(TARGET streamllm-gate-trace)
add_executable(${TARGET} streamllm-gate-trace.cpp)
target_link_libraries(${TARGET} PRIVATE
    llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
