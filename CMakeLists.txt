cmake_minimum_required(VERSION 3.28)

include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    project(gpu-io-libs LANGUAGES CXX CUDA)
    message(STATUS "CUDA compiler found, enabling CUDA support")
else()
    project(gpu-io-libs LANGUAGES CXX)
    message(STATUS "CUDA compiler not found, building CPU-only version")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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
