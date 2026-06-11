include(FetchContent)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.8.0
)
FetchContent_MakeAvailable(Catch2)

list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)
include(Catch)

# Set CUDA architecture to match available GPU (RTX 4090 = compute 8.9)
set(CMAKE_CUDA_ARCHITECTURES "89" CACHE STRING "CUDA architectures" FORCE)

add_executable(kvhdf5_tests
    unit/gpu_string_test.cpp
    unit/ref_test.cpp
    unit/dataspace_test.cpp
    unit/hdf5_datatype_test.cpp
    unit/serde_test.cpp
    unit/types_test.cpp
    unit/datatype_test.cpp
    unit/context_test.cpp
    unit/metadata_test.cpp
    unit/blob_store_test.cpp
    unit/container_test.cpp
    unit/container_complex_test.cpp
    unit/file_test.cpp
    unit/hdf5_group_test.cpp
    unit/hdf5_dataset_test.cpp
    unit/hdf5_attribute_test.cpp
    unit/dataset_single_chunk_test.cpp
    unit/dataset_multi_chunk_test.cpp
    unit/dataset_hyperslab_test.cpp
    unit/dataset_edge_test.cpp
    unit/dataset_stress_test.cpp
    unit/dataset_extent_test.cpp
    unit/dataset_types_test.cpp
    unit/dataset_chunk_iter_test.cpp
    unit/cte_blob_store_test.cpp
    integration/iowarp/vector_test.cpp
    integration/iowarp/cte_test.cpp
    integration/iowarp/buddy_allocator_test.cpp
)

# These test files include iowarp headers with __device__/__host__ keywords
# guarded by HSHM_ENABLE_CUDA / __CUDACC__. When compiled as plain CXX (not
# CUDA), the macros expand to nothing and everything works.
# GPU-specific tests use the separate kvhdf5_gpu_cte_tests target instead.

target_include_directories(kvhdf5_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/common
)

target_link_libraries(kvhdf5_tests PRIVATE
    kvhdf5
    Catch2::Catch2WithMain
)

target_compile_definitions(kvhdf5_tests PRIVATE
    KVHDF5_CHIMAERA_CONF="${KVHDF5_CHIMAERA_CONF_PATH}"
    $<$<BOOL:${WRP_CORE_ENABLE_CUDA}>:HSHM_ENABLE_CUDA=1>
)

# Set CUDA architecture for the test executable
set_target_properties(kvhdf5_tests PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
)

catch_discover_tests(kvhdf5_tests)

add_subdirectory(integration/iowarp/gpu)
