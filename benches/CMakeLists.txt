cmake_minimum_required(VERSION 3.28)

# Benchmarks require CUDA
if(NOT CMAKE_CUDA_COMPILER)
    message(STATUS "Skipping benchmarks (CUDA not available)")
    return()
endif()

message(STATUS "Configuring GPU HDF5 benchmarks...")

# Fetch Google Benchmark
include(FetchContent)
FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        v1.9.1
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googlebenchmark)
message(STATUS "Google Benchmark fetched successfully")

# Find system HDF5 library for CPU baseline benchmarks
# This is ONLY used in benchmarks, not in the main library
find_package(HDF5 COMPONENTS C)
if(NOT HDF5_FOUND)
    message(WARNING "HDF5 not found - CPU baseline benchmarks will be disabled")
    set(ENABLE_CPU_BASELINE OFF)
else()
    message(STATUS "HDF5 ${HDF5_VERSION} found for CPU baseline benchmarks")
    set(ENABLE_CPU_BASELINE ON)
endif()

# Collect benchmark sources
file(GLOB_RECURSE BENCH_SOURCES
    "*.cu"
)

# Collect HDF5 library sources (need to compile with benchmark)
file(GLOB_RECURSE HDF5_SOURCES
    ${CMAKE_SOURCE_DIR}/src/hdf5/*.cpp
)

# Create benchmark executable
add_executable(hdf5_benchmarks
    ${BENCH_SOURCES}
    ${HDF5_SOURCES}
    ${CMAKE_SOURCE_DIR}/src/iowarp/gpu_posix.cu
    ${CMAKE_SOURCE_DIR}/src/iowarp/cpu_polling.cpp
)

# Set HDF5 cpp files to compile as CUDA
set_source_files_properties(${HDF5_SOURCES} PROPERTIES LANGUAGE CUDA)
set_source_files_properties(${CMAKE_SOURCE_DIR}/src/iowarp/cpu_polling.cpp PROPERTIES LANGUAGE CUDA)

target_include_directories(hdf5_benchmarks PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${cccl_SOURCE_DIR}/libcudacxx/include
    ${cccl_SOURCE_DIR}/thrust
    ${cccl_SOURCE_DIR}/cub
)

target_link_libraries(hdf5_benchmarks PRIVATE
    benchmark::benchmark
    CUDA::cudart
    CCCL::CCCL
    hshm::cuda_cxx
    hshm::serialize
    hshm::lightbeam
)

# Link HDF5 only if found (for CPU baseline)
if(ENABLE_CPU_BASELINE)
    find_package(CURL REQUIRED)
    target_link_libraries(hdf5_benchmarks PRIVATE ${HDF5_C_LIBRARIES} CURL::libcurl)
    target_include_directories(hdf5_benchmarks PRIVATE ${HDF5_C_INCLUDE_DIRS})
    target_compile_definitions(hdf5_benchmarks PRIVATE HDF5_CPU_BASELINE_ENABLED=1)
endif()

target_compile_definitions(hdf5_benchmarks PRIVATE
    CUDA_ENABLED=1
    LIBCUDACXX_AVAILABLE=1
    _CCCL_HAS_SORTING_ALGORITHMS=1
)

target_compile_options(hdf5_benchmarks PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=177,20014,20011,20012>
)

# Force CUDA architecture to sm_89 (Ada Lovelace) for scoped atomics support
# Single architecture for faster builds
set(BENCH_CUDA_ARCHITECTURES "89")

set_target_properties(hdf5_benchmarks PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "${BENCH_CUDA_ARCHITECTURES}"
    CXX_STANDARD 20
    CUDA_STANDARD 20
)

message(STATUS "GPU HDF5 benchmarks configured")
