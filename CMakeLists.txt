cmake_minimum_required(VERSION 3.28)

# Declare the project (CXX) BEFORE probing for CUDA: check_language() requires an
# active project, otherwise it always reports NOTFOUND (CMake 4.x). Enable CUDA as
# a second language only if a working compiler is found.
project(gpu-io-libs LANGUAGES CXX)

include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    message(STATUS "CUDA compiler found, enabling CUDA support")
else()
    message(STATUS "CUDA compiler not found, building CPU-only version")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options to control which libraries to build
option(BUILD_HDF5_GPU "Build hdf5-gpu library" ON)
option(BUILD_KVHDF5 "Build kvhdf5 library" ON)

if(CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # Add subdirectories based on options
    if(BUILD_HDF5_GPU)
        add_subdirectory(libs/hdf5-gpu)
    endif()

    if(BUILD_KVHDF5)
        add_subdirectory(libs/kvhdf5)
    endif()
else()
    message(WARNING "CUDA not available - GPU libraries require CUDA")
    # kvhdf5 can still build CPU-only tests for development
    if(BUILD_KVHDF5)
        add_subdirectory(libs/kvhdf5)
    endif()
endif()
