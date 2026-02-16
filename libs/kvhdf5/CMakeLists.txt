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

# Fetch CCCL (deduplicated by FetchContent if hdf5-gpu already declared it)
include(FetchContent)
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
    )
else()
    # CPU-only mode: still link CCCL for cuda::std (works on CPU)
    target_link_libraries(kvhdf5 INTERFACE
        CCCL::CCCL
    )
endif()

# Tests
add_subdirectory(test)
