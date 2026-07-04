# kvhdf5 GPU end-to-end tests.
#
# Unlike the clio contract canary (../clio), these exercise kvhdf5's OWN
# device-facing producer surface (GpuCteDataset, GpuDatasetHandle, chunking,
# tag_path) top-to-bottom against a live clio CTE runtime. A failure here can be
# kvhdf5 OR clio — the ../clio canary is what disambiguates. Shared CTE server
# bring-up lives in ../support/cte_env.h.

if(NOT KVHDF5_HAS_CUDA)
    message(STATUS "kvhdf5_e2e_tests: skipped (no CUDA compiler)")
    return()
endif()

# add_cuda_executable comes from clio's ClioCoreCommon.cmake, pulled in when the
# clio subdirectory was added by the top-level kvhdf5 CMakeLists.
# DO_COPY=FALSE: compile the .cu in place. Sets CUDA_SEPARABLE_COMPILATION + PIC.
add_cuda_executable(kvhdf5_e2e_tests FALSE
    dataset_gpu_putget_test.cu
    gray_scott_gpu_test.cu
    multichunk_gpu_putget_test.cu
    async_overlap_test.cu
    dataset_lifecycle_test.cu
    gray_scott_async_e2e_test.cu
    dataset_refire_probe_test.cu
    gray_scott_threeway_bench.cu
    INCLUDE_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src
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

set_target_properties(kvhdf5_e2e_tests PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# catch_discover_tests runs the binary bare (no shell, empty env). The clio
# runtime needs CLIO_BIND_ADDR set and finds its chimods by scanning
# CHI_REPO_PATH / LD_LIBRARY_PATH, so inject those into the registered tests.
# (build/bin is where clio drops every .so + chimod.)
catch_discover_tests(kvhdf5_e2e_tests
    PROPERTIES
        ENVIRONMENT "CLIO_BIND_ADDR=127.0.0.1;CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin"
)

# Convenience target for the three-way Gray-Scott I/O benchmark. The gsbench
# arms are hidden ([.]) Catch2 cases inside kvhdf5_e2e_tests; each must run in
# its OWN process (clio's per-process GPU-backend ceiling), so a wrapper script
# orchestrates them. It is NOT wired into ctest on purpose: the script does a
# process-global `pkill` + shm wipe between arms, which would disrupt a normal
# `ctest` run.
#
#   cmake --build <dir> --target threeway_bench
#
# GSBENCH_BUILD_DIR is injected here so the script finds the binary regardless
# of which preset built it (build/ vs build-release/); the script's own default
# assumes build/. All other knobs remain env-overridable (see the script header
# and gray_scott_threeway_bench.cu).
add_custom_target(threeway_bench
    COMMAND ${CMAKE_COMMAND} -E env GSBENCH_BUILD_DIR=${CMAKE_BINARY_DIR}
            bash ${CMAKE_CURRENT_SOURCE_DIR}/run_threeway_bench.sh
    DEPENDS kvhdf5_e2e_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    USES_TERMINAL
    COMMENT "Running three-way Gray-Scott I/O benchmark (raw / sync / async)"
)
