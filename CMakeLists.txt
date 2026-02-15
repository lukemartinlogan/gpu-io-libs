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

if(CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # Add subdirectories
    add_subdirectory(libs/hdf5-gpu)
    # add_subdirectory(libs/kvhdf5)  # Future library
else()
    message(WARNING "CUDA not available - GPU libraries require CUDA")
endif()
