# CLIO dependency-contract canary.
#
# This is the ONE test that links clio directly and asserts the raw clio CTE
# producer-only contract kvhdf5 is built on — it includes NO kvhdf5 headers (the
# src/ include dir is deliberately omitted below, so it cannot). If a future clio
# bump breaks this, the failure points at clio, not at our abstractions. The
# kvhdf5 end-to-end tests that exercise our own producer surface live in
# ../e2e; shared CTE server bring-up lives in ../support/cte_env.h.
#
# Build recipe (link libs + include dirs) mirrors clio-core's reference test
#   context-transfer-engine/test/unit/CMakeLists.txt :: test_cte_devmem_putget

if(NOT KVHDF5_HAS_CUDA)
    message(STATUS "clio_contract_tests: skipped (no CUDA compiler)")
    return()
endif()

# add_cuda_executable comes from clio's ClioCoreCommon.cmake, pulled in when the
# clio subdirectory was added by the top-level kvhdf5 CMakeLists.
# DO_COPY=FALSE: compile the .cu in place. Sets CUDA_SEPARABLE_COMPILATION + PIC.
add_cuda_executable(clio_contract_tests FALSE
    cte_putget_test.cu
    INCLUDE_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/../support
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

set_target_properties(clio_contract_tests PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# catch_discover_tests runs the binary bare (no shell, empty env). The clio
# runtime needs CLIO_BIND_ADDR set and finds its chimods by scanning
# CHI_REPO_PATH / LD_LIBRARY_PATH, so inject those into the registered tests.
# (build/bin is where clio drops every .so + chimod.)
catch_discover_tests(clio_contract_tests
    PROPERTIES
        ENVIRONMENT "CLIO_BIND_ADDR=127.0.0.1;CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin"
)
