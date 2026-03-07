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

add_executable(kvhdf5_benchmarks
    cte_raw_bench.cpp
    container_bench.cpp
    group_bench.cpp
    attribute_bench.cpp
    dataset_bench.cpp
    lifecycle_bench.cpp
    vector_bench.cpp
)

target_include_directories(kvhdf5_benchmarks PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/../test/common
)

target_link_libraries(kvhdf5_benchmarks PRIVATE
    kvhdf5
    benchmark::benchmark_main
)

set_target_properties(kvhdf5_benchmarks PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
)

target_compile_definitions(kvhdf5_benchmarks PRIVATE
    KVHDF5_CHIMAERA_CONF="${KVHDF5_CHIMAERA_CONF_PATH}"
)
