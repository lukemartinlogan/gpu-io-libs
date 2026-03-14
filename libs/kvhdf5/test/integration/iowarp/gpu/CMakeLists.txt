if(NOT KVHDF5_HAS_CUDA)
    message(STATUS "kvhdf5_gpu_cte_tests: skipped (no CUDA compiler)")
    return()
endif()

add_executable(kvhdf5_gpu_cte_tests
    test_cpu_client.cc
    test_gpu_kernel.cu
)

set_source_files_properties(test_cpu_client.cc PROPERTIES LANGUAGE CUDA)

set_target_properties(kvhdf5_gpu_cte_tests PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
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

target_compile_options(kvhdf5_gpu_cte_tests PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

catch_discover_tests(kvhdf5_gpu_cte_tests)
