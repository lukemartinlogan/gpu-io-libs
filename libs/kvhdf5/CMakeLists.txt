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

# Fetch dependencies
include(FetchContent)

# Fetch IOWarp/core (if not already fetched by hdf5-gpu)
FetchContent_GetProperties(iowarp_core)
if(NOT iowarp_core_POPULATED)
    message(STATUS "Fetching IOWarp/core from GitHub...")
    FetchContent_Declare(
        iowarp_core
        GIT_REPOSITORY https://github.com/iowarp/core.git
        GIT_TAG        0852a9e1ac2efff316c46e790ab8cd175dd3d043  # Latest main as of 2026-02-16
    )

    # Disable all components except what we need
    set(WRP_CORE_ENABLE_RUNTIME OFF CACHE BOOL "" FORCE)
    set(WRP_CORE_ENABLE_CTE OFF CACHE BOOL "" FORCE)
    set(WRP_CORE_ENABLE_CAE OFF CACHE BOOL "" FORCE)
    set(WRP_CORE_ENABLE_CEE OFF CACHE BOOL "" FORCE)
    set(WRP_CORE_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
    set(WRP_CORE_ENABLE_BENCHMARKS OFF CACHE BOOL "" FORCE)
    # Enable CUDA and required features
    set(WRP_CORE_ENABLE_CUDA ON CACHE BOOL "" FORCE)
    set(WRP_CORE_ENABLE_CEREAL ON CACHE BOOL "" FORCE)
    set(WRP_CORE_ENABLE_ZMQ OFF CACHE BOOL "" FORCE)
    set(WRP_CORE_ENABLE_LIBFABRIC OFF CACHE BOOL "" FORCE)
    set(WRP_CORE_ENABLE_MPI OFF CACHE BOOL "" FORCE)
    set(WRP_CORE_ENABLE_COMPRESS OFF CACHE BOOL "" FORCE)
    set(WRP_CORE_ENABLE_ENCRYPT OFF CACHE BOOL "" FORCE)
    set(WRP_CORE_ENABLE_ELF OFF CACHE BOOL "" FORCE)
    set(WRP_CORE_ENABLE_PYTHON OFF CACHE BOOL "" FORCE)
    set(WRP_CORE_ENABLE_DOXYGEN OFF CACHE BOOL "" FORCE)
    set(WRP_CORE_ENABLE_HDF5 OFF CACHE BOOL "" FORCE)

    FetchContent_MakeAvailable(iowarp_core)
    message(STATUS "IOWarp/core built successfully")
else()
    message(STATUS "IOWarp/core already populated, reusing existing")
endif()

# Fetch CCCL (deduplicated by FetchContent if hdf5-gpu already declared it)
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
        hshm::cuda_cxx
        hshm::serialize
        hshm::lightbeam
    )
else()
    # CPU-only mode: still link CCCL for cuda::std (works on CPU)
    target_link_libraries(kvhdf5 INTERFACE
        CCCL::CCCL
        hshm::cuda_cxx
        hshm::serialize
        hshm::lightbeam
    )
endif()

# Tests
add_subdirectory(test)
