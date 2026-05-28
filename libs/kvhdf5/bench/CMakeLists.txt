include(FetchContent)

FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.1
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(benchmark)

# Set CUDA architecture to match available GPU (RTX 4090 = compute 8.9)
set(CMAKE_CUDA_ARCHITECTURES "89" CACHE STRING "CUDA architectures" FORCE)

find_package(CUDAToolkit REQUIRED)

add_executable(kvhdf5_benchmarks
    cte_raw_bench.cpp
    container_bench.cpp
    group_bench.cpp
    attribute_bench.cpp
    dataset_bench.cpp
    lifecycle_bench.cpp
    vector_bench.cpp
    gray_scott_bench.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../examples/gray_scott_traditional/gs_kernel.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../examples/gray_scott_traditional/ram_backend.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../examples/gray_scott_traditional/disk_backend.cc
    iowarp_full_bench.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../examples/gray_scott_iowarp/iowarp_step.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../examples/gray_scott_iowarp/gray_scott_host.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../examples/gray_scott_iowarp/gray_scott_init.cc
)

# Same -gline-tables-only workaround the iowarp example uses (see
# examples/gray_scott_iowarp/CMakeLists.txt for the reason).
set_source_files_properties(
    iowarp_full_bench.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../examples/gray_scott_iowarp/iowarp_step.cu
    PROPERTIES COMPILE_FLAGS "-gline-tables-only"
)

target_include_directories(kvhdf5_benchmarks PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/../test/common
    ${CMAKE_CURRENT_SOURCE_DIR}/../test/integration/iowarp/gpu
    ${CMAKE_CURRENT_SOURCE_DIR}/../examples/gray_scott_traditional
    ${CMAKE_CURRENT_SOURCE_DIR}/../examples/gray_scott_iowarp
    ${CMAKE_CURRENT_SOURCE_DIR}/../examples/common
)

target_link_libraries(kvhdf5_benchmarks PRIVATE
    kvhdf5
    benchmark::benchmark_main
    CUDA::cudart
)

set_target_properties(kvhdf5_benchmarks PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_definitions(kvhdf5_benchmarks PRIVATE
    KVHDF5_CHIMAERA_CONF="${KVHDF5_CHIMAERA_CONF_PATH}"
    HSHM_ENABLE_CUDA=1
)
