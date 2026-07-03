# GPU integration tests against iowarp's CTE producer-only contract.
#
# These link iowarp directly (NOT the kvhdf5 library) so they keep proving the
# raw iowarp assumptions kvhdf5 depends on — if one breaks, the failure points
# at iowarp, not at our abstractions. Catch2 is fetched once by the parent
# test/CMakeLists.txt; this file only consumes the targets.
#
# Build recipe (link libs + include dirs) mirrors iowarp-core's reference test
#   context-transfer-engine/test/unit/CMakeLists.txt :: test_cte_devmem_putget

if(NOT KVHDF5_HAS_CUDA)
    message(STATUS "iowarp_cte_integration_tests: skipped (no CUDA compiler)")
    return()
endif()

# add_cuda_executable comes from iowarp's ClioCoreCommon.cmake, pulled in when
# the iowarp subdirectory was added by the top-level kvhdf5 CMakeLists.
# DO_COPY=FALSE: compile the .cu in place. Sets CUDA_SEPARABLE_COMPILATION + PIC.
add_cuda_executable(iowarp_cte_integration_tests FALSE
    cte_putget_test.cu
    dataset_gpu_putget_test.cu
    gray_scott_gpu_test.cu
    multichunk_gpu_putget_test.cu
    async_overlap_test.cu
    dataset_lifecycle_test.cu
    gray_scott_async_e2e_test.cu
    dataset_refire_probe_test.cu
    gray_scott_threeway_bench.cu
    INCLUDE_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../src
        ${iowarp_core_SOURCE_DIR}/context-runtime/include
        ${iowarp_core_SOURCE_DIR}/context-runtime/modules/admin/include
        ${iowarp_core_SOURCE_DIR}/context-transfer-engine/core/include
    LINK_LIBS
        clio_run_cxx
        clio_run_cxx_gpu
        clio_admin_client
        clio_bdev_client
        clio_cte_core_client
        clio_cte_core_runtime
        ctp::cuda_cxx
        Catch2::Catch2WithMain
        ${CMAKE_THREAD_LIBS_INIT}
    DEFS
        CTP_ENABLE_CUDA=1
        HSHM_ENABLE_CUDA=1
)

set_target_properties(iowarp_cte_integration_tests PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# catch_discover_tests registers each TEST_CASE as a ctest test that runs the
# binary bare (no shell, default WORKING_DIRECTORY, empty env). The iowarp
# runtime needs CLIO_BIND_ADDR set and finds its chimods by scanning
# CHI_REPO_PATH / LD_LIBRARY_PATH, so inject those into the registered tests.
# (build/bin is where iowarp drops every .so + chimod.)
catch_discover_tests(iowarp_cte_integration_tests
    PROPERTIES
        ENVIRONMENT "CLIO_BIND_ADDR=127.0.0.1;CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin"
)
