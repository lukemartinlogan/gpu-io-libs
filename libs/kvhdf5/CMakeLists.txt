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
set(HSHM_LOG_LEVEL             "0" CACHE STRING "" FORCE)

# Off for our use
set(WRP_CORE_ENABLE_CUDA       OFF CACHE BOOL "" FORCE)
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
    GIT_TAG        1664d8dac364ae3da5c5da2a31e7a693c03c2b70  # Add nonatomic fallback (atomic.h fix)
)
# FetchContent_Populate(<name>) is deprecated in CMake 3.30+ (CMP0169).
# We need it here to interject the cmake file copy between fetch and
# add_subdirectory; set OLD to suppress the warning until a cleaner pattern
# is available upstream in clio-core.
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
message(STATUS "clio-core loaded from FetchContent (abscosmos/iowarp-core)")

FetchContent_Declare(
    cccl
    GIT_REPOSITORY https://github.com/NVIDIA/cccl.git
    GIT_TAG e5ed71c7dce1130e582db831fffc61f562203884
)
FetchContent_MakeAvailable(cccl)

# Header-only library for now; will become STATIC when .cpp files are added
add_library(kvhdf5 INTERFACE)

target_include_directories(kvhdf5 INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

if(KVHDF5_HAS_CUDA)
    target_link_libraries(kvhdf5 INTERFACE
        CUDA::cudart
        CCCL::CCCL
        hshm::cxx
        hshm::serialize
        hshm::lightbeam
        wrp_cte_core_runtime
        chimaera_cxx
    )
else()
    # CPU-only mode: still link CCCL for cuda::std (works on CPU)
    target_link_libraries(kvhdf5 INTERFACE
        CCCL::CCCL
        hshm::cxx
        hshm::serialize
        hshm::lightbeam
    )
endif()

# Tests
add_subdirectory(test)
