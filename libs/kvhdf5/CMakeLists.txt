cmake_minimum_required(VERSION 3.28)

# Determine available languages from parent scope
if(CMAKE_CUDA_COMPILER)
    project(kvhdf5 LANGUAGES CXX CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    find_package(CUDAToolkit REQUIRED)
    set(KVHDF5_HAS_CUDA ON)
else()
    project(kvhdf5 LANGUAGES CXX)
    set(KVHDF5_HAS_CUDA OFF)
    message(STATUS "kvhdf5: Building CPU-only (tests only)")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# Modules we need
set(WRP_CORE_ENABLE_RUNTIME    ON  CACHE BOOL "" FORCE)
set(WRP_CORE_ENABLE_CTE        ON  CACHE BOOL "" FORCE)
set(WRP_CORE_ENABLE_CAE        OFF CACHE BOOL "" FORCE)
set(WRP_CORE_ENABLE_CEE        OFF CACHE BOOL "" FORCE)
set(WRP_CORE_ENABLE_TESTS      OFF CACHE BOOL "" FORCE)
set(WRP_CORE_ENABLE_BENCHMARKS OFF CACHE BOOL "" FORCE)

# Matches debug preset
set(WRP_CORE_ENABLE_ZMQ        ON  CACHE BOOL "" FORCE)  # Required for Chimaera networking
set(WRP_CORE_ENABLE_CEREAL     ON  CACHE BOOL "" FORCE)
set(WRP_CORE_ENABLE_IO_URING   ON  CACHE BOOL "" FORCE)
set(WRP_CORE_ENABLE_RPATH      ON  CACHE BOOL "" FORCE)
set(WRP_CORE_ENABLE_ASAN       OFF CACHE BOOL "" FORCE)
set(HSHM_LOG_LEVEL             "3" CACHE STRING "" FORCE)

# Enable CUDA in iowarp-core when CUDA compiler is available
if(KVHDF5_HAS_CUDA)
    set(WRP_CORE_ENABLE_CUDA   ON  CACHE BOOL "" FORCE)
    # Set architecture before iowarp-core configures (RTX 4090 = sm_89)
    set(CMAKE_CUDA_ARCHITECTURES "89" CACHE STRING "CUDA architectures" FORCE)
else()
    set(WRP_CORE_ENABLE_CUDA   OFF CACHE BOOL "" FORCE)
endif()
set(WRP_CORE_ENABLE_ELF        OFF CACHE BOOL "" FORCE)
set(WRP_CORE_ENABLE_LIBFABRIC  OFF CACHE BOOL "" FORCE)
set(WRP_CORE_ENABLE_MPI        OFF CACHE BOOL "" FORCE)
set(WRP_CORE_ENABLE_COMPRESS   OFF CACHE BOOL "" FORCE)
set(WRP_CORE_ENABLE_ENCRYPT    OFF CACHE BOOL "" FORCE)
set(WRP_CORE_ENABLE_PYTHON     OFF CACHE BOOL "" FORCE)
set(WRP_CORE_ENABLE_DOXYGEN    OFF CACHE BOOL "" FORCE)
set(WRP_CORE_ENABLE_HDF5       OFF CACHE BOOL "" FORCE)
set(WRP_CORE_ENABLE_GRAY_SCOTT OFF CACHE BOOL "" FORCE)
set(WRP_CORE_ENABLE_CONDA      OFF CACHE BOOL "" FORCE)
set(WRP_CORE_ENABLE_COVERAGE   OFF CACHE BOOL "" FORCE)
set(WRP_CORE_ENABLE_DOCKER_CI  OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    iowarp_core
    GIT_REPOSITORY https://github.com/abscosmos/iowarp-core.git
    GIT_TAG        add-missing-guard
    GIT_SHALLOW    TRUE
)
# FetchContent_Populate(<name>) is deprecated in CMake 3.30+ (CMP0169).
# We need it here to interject the cmake file copy between fetch and
# add_subdirectory; set OLD to suppress the warning until a cleaner pattern
# is available upstream.
cmake_policy(SET CMP0169 OLD)
FetchContent_GetProperties(iowarp_core)
if(NOT iowarp_core_POPULATED)
    FetchContent_Populate(iowarp_core)
    list(APPEND CMAKE_MODULE_PATH "${iowarp_core_SOURCE_DIR}/cmake")
    file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/cmake")
    file(COPY
        "${iowarp_core_SOURCE_DIR}/cmake/IowarpCoreConfig.cmake.in"
        "${iowarp_core_SOURCE_DIR}/cmake/IowarpCoreCommon.cmake"
        DESTINATION "${CMAKE_SOURCE_DIR}/cmake/"
    )
    add_subdirectory(${iowarp_core_SOURCE_DIR} ${iowarp_core_BINARY_DIR})
endif()
message(STATUS "iowarp-core loaded from FetchContent (abscosmos/iowarp-core @ 856f364e (add-missing-guard))")

FetchContent_Declare(
    cccl
    GIT_REPOSITORY https://github.com/NVIDIA/cccl.git
    GIT_TAG        e5ed71c7dce1130e582db831fffc61f562203884
)
FetchContent_MakeAvailable(cccl)

add_library(kvhdf5 STATIC
    src/kvhdf5/cte_blob_store.cpp
)

target_include_directories(kvhdf5 PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

if(KVHDF5_HAS_CUDA)
    target_compile_options(kvhdf5 PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )
    target_compile_definitions(kvhdf5 PUBLIC HSHM_ENABLE_CUDA=1)
    target_link_libraries(kvhdf5 PUBLIC
        CUDA::cudart
        CCCL::CCCL
        hshm::cxx
        hshm::serialize
        hshm::lightbeam
        wrp_cte_core_runtime
        chimaera_cxx
    )
else()
    target_link_libraries(kvhdf5 PUBLIC
        CCCL::CCCL
        hshm::cxx
        hshm::serialize
        hshm::lightbeam
        wrp_cte_core_runtime
        chimaera_cxx
    )
endif()

set(KVHDF5_CHIMAERA_CONF_PATH "${CMAKE_CURRENT_SOURCE_DIR}/config/chimaera_ram.yaml")

# Tests
add_subdirectory(test)

# Benchmarks
add_subdirectory(bench)
