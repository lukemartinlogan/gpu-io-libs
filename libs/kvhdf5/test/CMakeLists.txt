include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.8.0
)
FetchContent_MakeAvailable(Catch2)

list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)
include(Catch)

# CMAKE_CUDA_ARCHITECTURES is set once by the top-level CMakeLists (sm_89 for the
# dev container's RTX 4090); the GPU test targets consume it below.

# Shared recipe for the GPU test binaries: the clio contract canary (test/clio)
# and the kvhdf5 end-to-end tests (test/e2e). Both link the same clio runtime
# libs, compile with the same CUDA defs, and need the same runtime env injected
# into their ctest registrations. They differ in only three ways: target name,
# sources, and whether they see kvhdf5's own headers — the canary must NOT (pass
# WITH_KVHDF5_HEADERS only for e2e), so it physically cannot include kvhdf5 code.
#
#   kvhdf5_add_gpu_test_target(<target> [WITH_KVHDF5_HEADERS] SRCS <files...>)
#
# Called from the test/clio and test/e2e subdirs, so the relative include paths
# (../support, ../../src) resolve against the CALLER's dir — both sit at the same
# depth under test/.
function(kvhdf5_add_gpu_test_target target)
    cmake_parse_arguments(ARG "WITH_KVHDF5_HEADERS" "" "SRCS" ${ARGN})

    set(_include_dirs
        ${CMAKE_CURRENT_SOURCE_DIR}/../support
        ${iowarp_core_SOURCE_DIR}/context-runtime/include
        ${iowarp_core_SOURCE_DIR}/context-runtime/modules/admin/include
        ${iowarp_core_SOURCE_DIR}/context-transfer-engine/core/include
    )
    if(ARG_WITH_KVHDF5_HEADERS)
        list(APPEND _include_dirs ${CMAKE_CURRENT_SOURCE_DIR}/../../src)
    endif()

    # add_cuda_executable comes from clio's ClioCoreCommon.cmake, pulled in when
    # the clio subdirectory was added by the top-level kvhdf5 CMakeLists.
    # DO_COPY=FALSE: compile the .cu in place. Sets CUDA_SEPARABLE_COMPILATION + PIC.
    add_cuda_executable(${target} FALSE
        ${ARG_SRCS}
        INCLUDE_DIRS
            ${_include_dirs}
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

    set_target_properties(${target} PROPERTIES
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )

    # catch_discover_tests runs the binary bare (no shell, empty env). The clio
    # runtime needs CLIO_BIND_ADDR set and finds its chimods by scanning
    # CHI_REPO_PATH / LD_LIBRARY_PATH, so inject those into the registered tests.
    # (build/bin is where clio drops every .so + chimod.)
    catch_discover_tests(${target}
        PROPERTIES
            ENVIRONMENT "CLIO_BIND_ADDR=127.0.0.1;CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin"
    )
endfunction()

add_subdirectory(unit)

# clio dependency-contract canary (raw clio, no kvhdf5 headers)
add_subdirectory(clio)

# kvhdf5 GPU end-to-end tests (our producer surface on the clio runtime)
add_subdirectory(e2e)
