cmake_minimum_required(VERSION 3.28)

include(CheckLanguage)
include(FetchContent)

check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    project(hdf5_implementation LANGUAGES CXX CUDA)
    message(STATUS "CUDA compiler found, enabling CUDA support")
else()
    project(hdf5_implementation LANGUAGES CXX)
    message(STATUS "CUDA compiler not found, building CPU-only version")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89")
    endif()

    find_package(CUDAToolkit REQUIRED)
    message(STATUS "CUDA Toolkit ${CUDAToolkit_VERSION} found")

    message(STATUS "Fetching CCCL from GitHub...")
    FetchContent_Declare(
        cccl
        GIT_REPOSITORY https://github.com/NVIDIA/cccl.git
        GIT_TAG        e5ed71c7dce1130e582db831fffc61f562203884 # 2025-12-13
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(cccl)
    message(STATUS "CCCL fetched successfully")

    message(STATUS "Fetching stdgpu from GitHub...")
    FetchContent_Declare(
        stdgpu
        GIT_REPOSITORY https://github.com/stotko/stdgpu.git
        GIT_TAG        42ea5eacc8b1665023a22632589f86594dd53b8c # 2025-12-13
        GIT_SHALLOW    TRUE
    )
    set(STDGPU_BUILD_SHARED_LIBS OFF CACHE BOOL "Build stdgpu as static library")
    set(STDGPU_BUILD_EXAMPLES OFF CACHE BOOL "Don't build stdgpu examples")
    set(STDGPU_BUILD_TESTS OFF CACHE BOOL "Don't build stdgpu tests")
    set(STDGPU_BUILD_BENCHMARKS OFF CACHE BOOL "Don't build stdgpu benchmarks")
    FetchContent_MakeAvailable(stdgpu)
    message(STATUS "stdgpu fetched successfully")
endif()

file(GLOB_RECURSE SOURCES
    "src/*.cpp"
    "src/*.h"
    "third_party/*.cpp"
    "third_party/*.h"
)

add_executable(hdf5_implementation ${SOURCES})

if(CMAKE_CUDA_COMPILER)
    target_link_libraries(hdf5_implementation PRIVATE
        CUDA::cudart
        CCCL::CCCL
        stdgpu::stdgpu
    )
    target_compile_definitions(hdf5_implementation PRIVATE
        CUDA_ENABLED=1
        LIBCUDACXX_AVAILABLE=1
        _CCCL_HAS_SORTING_ALGORITHMS=1
    )
    target_include_directories(hdf5_implementation PRIVATE
        ${cccl_SOURCE_DIR}/libcudacxx/include
        ${cccl_SOURCE_DIR}/thrust
        ${cccl_SOURCE_DIR}/cub
    )
    message(STATUS "Building with CUDA, libcu++, and stdgpu support")
else()
    message(STATUS "Building CPU-only version")
endif()