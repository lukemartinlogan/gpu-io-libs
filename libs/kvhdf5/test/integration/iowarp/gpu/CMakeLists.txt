if(NOT KVHDF5_HAS_CUDA)
    message(STATUS "kvhdf5_gpu_cte_tests: skipped (no CUDA compiler)")
    return()
endif()

add_executable(kvhdf5_gpu_cte_tests
    test_cpu_client.cc
    test_gpu_kernel.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../test/unit/gpu_cte_blob_store_test.cu
)


set_target_properties(kvhdf5_gpu_cte_tests PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION ON
)

# gpu_cte_blob_store_test.cu instantiates cuda::std::span in device code, which
# causes clang-18 to emit __func__ DWARF entries whose symbol names contain dots.
# ptxas (CUDA 12.x) cannot parse dot-in-symbol-name in DWARF variable entries,
# so we use -gline-tables-only (line info only, no variable DWARF) for this file.
set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../test/unit/gpu_cte_blob_store_test.cu
    PROPERTIES COMPILE_FLAGS "-gline-tables-only"
)

target_include_directories(kvhdf5_gpu_cte_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../common
)

target_link_libraries(kvhdf5_gpu_cte_tests PRIVATE
    kvhdf5
    Catch2::Catch2WithMain
)

target_compile_definitions(kvhdf5_gpu_cte_tests PRIVATE
    KVHDF5_CHIMAERA_CONF="${KVHDF5_CHIMAERA_CONF_PATH}"
    HSHM_ENABLE_CUDA=1
)


catch_discover_tests(kvhdf5_gpu_cte_tests)
