#  Transformer Inference Engine -- Stage 0 build system.
#
#  Builds every microbenchmark and every unit test, and records the exact
#  compiler and flag set into the build output and into build_info.h, so every
#  results file carries the toolchain it was produced with.
#
#  THE FLAG SET BELOW IS FROZEN FOR THE PROJECT. Later stages compare their
#  measurements against Stage 0, and that comparison is void if the flags
#  change. The same flags are recorded in HARDWARE.md section 5.5.
#
#  TOOLSET PINNING: CUDA 13.1 accepts Visual Studio 2019 through 2022 only.
#  Visual Studio 2026 (MSVC 19.50+) is also installed on the measurement
#  machine and nvcc rejects it in host_config.h. The check below fails the
#  configure step rather than letting a build silently pick up the wrong
#  toolset, and -allow-unsupported-compiler is deliberately NOT used: it
#  permits incorrect runtime execution and this project is gated on numerical
#  correctness against a reference.
#
#  Configure with scripts/build.ps1, which enters the VS 2022 environment and
#  passes the 2022 cl.exe explicitly as CMAKE_C/CXX/CUDA_HOST_COMPILER.

cmake_minimum_required(VERSION 3.24)

project(transformer_inference_engine LANGUAGES C CXX CUDA)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ---------------------------------------------------------------- toolset ---
if(MSVC)
    if(MSVC_VERSION GREATER_EQUAL 1950)
        message(FATAL_ERROR
            "MSVC ${MSVC_VERSION} (cl ${CMAKE_CXX_COMPILER_VERSION}) is the Visual "
            "Studio 2026 toolset. CUDA ${CMAKE_CUDA_COMPILER_VERSION} rejects it. "
            "Build from the VS 2022 environment -- run scripts/build.ps1, which "
            "pins the 2022 cl.exe explicitly. Do not pass "
            "-allow-unsupported-compiler.")
    endif()
    if(MSVC_VERSION LESS 1920)
        message(FATAL_ERROR "MSVC ${MSVC_VERSION} is older than Visual Studio 2019, "
                            "which CUDA ${CMAKE_CUDA_COMPILER_VERSION} requires as a minimum.")
    endif()
endif()

set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING "sm_75. No sm_80+ features anywhere in this project.")
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)

# ------------------------------------------------------------ frozen flags ---
if(MSVC)
    set(TIE_HOST_FLAGS "/W3 /arch:AVX2 /fp:precise")
    set(CMAKE_C_FLAGS_RELEASE   "/MD /O2 /Ob2 /DNDEBUG" CACHE STRING "" FORCE)
    set(CMAKE_CXX_FLAGS_RELEASE "/MD /O2 /Ob2 /DNDEBUG" CACHE STRING "" FORCE)
    set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} ${TIE_HOST_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TIE_HOST_FLAGS}")
    set(TIE_CUDA_FLAGS "-O3 --generate-line-info -Xptxas=-O3 -Xptxas=-v -Xcompiler=/arch:AVX2 -Xcompiler=/fp:precise")
else()
    set(TIE_HOST_FLAGS "-Wall -Wextra -O3 -mavx2 -mfma")
    set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} ${TIE_HOST_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TIE_HOST_FLAGS}")
    set(TIE_CUDA_FLAGS "-O3 --generate-line-info -Xptxas=-O3 -Xptxas=-v")
endif()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${TIE_CUDA_FLAGS}")

find_package(CUDAToolkit REQUIRED)

# The exact strings recorded into every results file.
set(TIE_CXX_COMPILER_STR  "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} (${CMAKE_CXX_COMPILER})")
set(TIE_CUDA_COMPILER_STR "nvcc ${CMAKE_CUDA_COMPILER_VERSION} (${CMAKE_CUDA_COMPILER})")
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(TIE_CXX_FLAGS_STR "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
else()
    set(TIE_CXX_FLAGS_STR "${CMAKE_CXX_FLAGS}")
endif()
set(TIE_CUDA_FLAGS_STR "${CMAKE_CUDA_FLAGS} -arch=sm_75")
string(TIMESTAMP TIE_BUILD_TIMESTAMP "%Y-%m-%dT%H:%M:%SZ" UTC)

message(STATUS "=== Stage 0 frozen toolchain =========================================")
message(STATUS "  host compiler : ${TIE_CXX_COMPILER_STR}")
message(STATUS "  host flags    : ${TIE_CXX_FLAGS_STR}")
message(STATUS "  cuda compiler : ${TIE_CUDA_COMPILER_STR}")
message(STATUS "  cuda flags    : ${TIE_CUDA_FLAGS_STR}")
message(STATUS "  cuda arch     : sm_${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "======================================================================")

# ------------------------------------------------------------ build_info.h ---
set(TIE_GENERATED_DIR "${CMAKE_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${TIE_GENERATED_DIR}")
set(TIE_RESULTS_DIR "${CMAKE_SOURCE_DIR}/bench/results")

# The flag strings contain quotes and, on Windows, backslashes. Escape both
# before they are pasted into C string literals in build_info.h.
foreach(_v TIE_CXX_COMPILER_STR TIE_CUDA_COMPILER_STR
           TIE_CXX_FLAGS_STR TIE_CUDA_FLAGS_STR TIE_RESULTS_DIR)
    string(REPLACE "\\" "\\\\" ${_v} "${${_v}}")
    string(REPLACE "\"" "\\\"" ${_v} "${${_v}}")
endforeach()

configure_file("${CMAKE_SOURCE_DIR}/scripts/build_info.h.in"
               "${TIE_GENERATED_DIR}/build_info.h" @ONLY)

# git_info.h is regenerated on every build so the commit hash a results file
# carries is the commit the binary was actually built from.
add_custom_target(tie_git_info ALL
    COMMAND ${CMAKE_COMMAND}
            -DSRC_DIR=${CMAKE_SOURCE_DIR}
            -DOUT_DIR=${TIE_GENERATED_DIR}
            -P ${CMAKE_SOURCE_DIR}/scripts/git_info.cmake
    BYPRODUCTS "${TIE_GENERATED_DIR}/git_info.h"
    COMMENT "Recording git commit into git_info.h")

# ----------------------------------------------------------------- targets ---
set(TIE_MB_DIR "${CMAKE_SOURCE_DIR}/bench/microbench")

add_library(bench_core STATIC
    "${TIE_MB_DIR}/bench_common.cu"
    "${TIE_MB_DIR}/device_info.cu")
add_dependencies(bench_core tie_git_info)
target_include_directories(bench_core PUBLIC "${TIE_MB_DIR}" "${TIE_GENERATED_DIR}")
target_link_libraries(bench_core PUBLIC CUDA::cudart)
if(TARGET CUDA::nvml)
    target_link_libraries(bench_core PUBLIC CUDA::nvml)
else()
    target_link_libraries(bench_core PUBLIC "${CUDAToolkit_LIBRARY_DIR}/nvml.lib")
endif()
set_target_properties(bench_core PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")

# tie_microbench(<name> <source> [extra libs...])
#   Produces the runnable benchmark <name> and an object library <name>_obj
#   compiled with BENCH_NO_MAIN that the unit test links against. One source of
#   truth per benchmark: the tested code and the run code are the same code.
function(tie_microbench name source)
    add_executable(${name} "${TIE_MB_DIR}/${source}")
    target_link_libraries(${name} PRIVATE bench_core ${ARGN})

    add_library(${name}_obj OBJECT "${TIE_MB_DIR}/${source}")
    target_compile_definitions(${name}_obj PRIVATE BENCH_NO_MAIN)
    target_include_directories(${name}_obj PUBLIC "${TIE_MB_DIR}" "${TIE_GENERATED_DIR}")
    add_dependencies(${name}_obj tie_git_info)
    set_target_properties(${name} ${name}_obj PROPERTIES
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
endfunction()

tie_microbench(gpu_bandwidth           gpu_bandwidth.cu)
tie_microbench(gpu_fp32_peak           gpu_fp32_peak.cu)
tie_microbench(cpu_cache_ladder        cpu_cache_ladder.c)
tie_microbench(cpu_simd_peak           cpu_simd_peak.c)
tie_microbench(host_device_transfer    host_device_transfer.cu)
tie_microbench(cublas_sgemm_ref        cublas_sgemm_ref.cu CUDA::cublas)
tie_microbench(kernel_launch_overhead  kernel_launch_overhead.cu)
tie_microbench(shared_mem_bandwidth    shared_mem_bandwidth.cu)
tie_microbench(occupancy_sweep         occupancy_sweep.cu)
target_link_libraries(cublas_sgemm_ref_obj PUBLIC CUDA::cublas)

# ---------------------------------------------------------------------- PTX ---
# Two tests assert structural properties of generated device code (the FP32
# peak loop touches no global memory; the launch-overhead kernel is genuinely
# empty). They read the PTX rather than trusting a comment.
set(TIE_PTX_DIR "${CMAKE_BINARY_DIR}/ptx")
file(MAKE_DIRECTORY "${TIE_PTX_DIR}")
function(tie_ptx name source)
    add_custom_command(
        OUTPUT "${TIE_PTX_DIR}/${name}.ptx"
        COMMAND ${CMAKE_CUDA_COMPILER} -ptx -arch=sm_75
                -I "${TIE_MB_DIR}" -I "${TIE_GENERATED_DIR}"
                "${TIE_MB_DIR}/${source}" -o "${TIE_PTX_DIR}/${name}.ptx"
        DEPENDS "${TIE_MB_DIR}/${source}" tie_git_info
        COMMENT "Emitting PTX for ${name}")
    add_custom_target(${name}_ptx DEPENDS "${TIE_PTX_DIR}/${name}.ptx")
endfunction()
tie_ptx(gpu_fp32_peak          gpu_fp32_peak.cu)
tie_ptx(kernel_launch_overhead kernel_launch_overhead.cu)

# --------------------------------------------------------------------- tests ---
enable_testing()

# tie_test(<test name> <source> <benchmark object library> [extra libs...])
function(tie_test name source objlib)
    if(objlib STREQUAL "NONE")
        add_executable(${name} "${CMAKE_SOURCE_DIR}/tests/${source}")
    else()
        add_executable(${name} "${CMAKE_SOURCE_DIR}/tests/${source}"
                       $<TARGET_OBJECTS:${objlib}>)
    endif()
    target_link_libraries(${name} PRIVATE bench_core ${ARGN})
    target_include_directories(${name} PRIVATE "${TIE_MB_DIR}" "${TIE_GENERATED_DIR}")
    target_compile_definitions(${name} PRIVATE TIE_PTX_DIR="${TIE_PTX_DIR}")
    set_target_properties(${name} PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    add_test(NAME ${name} COMMAND ${name})
    set_tests_properties(${name} PROPERTIES TIMEOUT 300)   # 5-minute per-test cap
endfunction()

tie_test(test_bench_common           test_bench_common.cu           NONE)
tie_test(test_gpu_bandwidth          test_gpu_bandwidth.cu          gpu_bandwidth_obj)
tie_test(test_gpu_fp32_peak          test_gpu_fp32_peak.cu          gpu_fp32_peak_obj)
tie_test(test_cpu_cache_ladder       test_cpu_cache_ladder.cu       cpu_cache_ladder_obj)
tie_test(test_cpu_simd_peak          test_cpu_simd_peak.cu          cpu_simd_peak_obj)
tie_test(test_host_device_transfer   test_host_device_transfer.cu   host_device_transfer_obj)
tie_test(test_cublas_sgemm_ref       test_cublas_sgemm_ref.cu       cublas_sgemm_ref_obj CUDA::cublas)
tie_test(test_kernel_launch_overhead test_kernel_launch_overhead.cu kernel_launch_overhead_obj)
tie_test(test_shared_mem_bandwidth   test_shared_mem_bandwidth.cu   shared_mem_bandwidth_obj)
tie_test(test_occupancy_sweep        test_occupancy_sweep.cu        occupancy_sweep_obj)

add_dependencies(test_gpu_fp32_peak          gpu_fp32_peak_ptx)
add_dependencies(test_kernel_launch_overhead kernel_launch_overhead_ptx)

# Python-side tests. They are registered with the same runner so the whole
# suite runs from one command: ctest --test-dir build.
# Prefer the project's own virtual environment: it is the one carrying the
# PyTorch correctness oracle that bench/machine_state.py uses.
if(WIN32 AND EXISTS "${CMAKE_SOURCE_DIR}/.venv/Scripts/python.exe")
    set(Python3_EXECUTABLE "${CMAKE_SOURCE_DIR}/.venv/Scripts/python.exe")
    set(Python3_Interpreter_FOUND TRUE)
elseif(EXISTS "${CMAKE_SOURCE_DIR}/.venv/bin/python")
    set(Python3_EXECUTABLE "${CMAKE_SOURCE_DIR}/.venv/bin/python")
    set(Python3_Interpreter_FOUND TRUE)
else()
    find_package(Python3 COMPONENTS Interpreter)
endif()
message(STATUS "  python        : ${Python3_EXECUTABLE}")
if(Python3_Interpreter_FOUND)
    foreach(pytest test_run_all test_machine_state test_profile test_analyze_variance)
        add_test(NAME ${pytest}
                 COMMAND ${Python3_EXECUTABLE} "${CMAKE_SOURCE_DIR}/tests/${pytest}.py")
        set_tests_properties(${pytest} PROPERTIES TIMEOUT 300)
    endforeach()
else()
    message(WARNING "Python 3 interpreter not found; the three Python unit tests "
                    "are NOT registered and the suite is incomplete.")
endif()
