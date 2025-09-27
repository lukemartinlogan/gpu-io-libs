cmake_minimum_required(VERSION 3.28)

include(CheckLanguage)
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
endif()

file(GLOB_RECURSE SOURCES
    "src/*.cpp"
    "src/*.h"
    "include/*.cpp"
    "include/*.h"
)

add_executable(hdf5_implementation ${SOURCES})

if(CMAKE_CUDA_COMPILER)
    target_link_libraries(hdf5_implementation PRIVATE CUDA::cudart)
    target_compile_definitions(hdf5_implementation PRIVATE
        CUDA_ENABLED=1
        LIBCUDACXX_AVAILABLE=1
    )
    target_include_directories(hdf5_implementation PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    message(STATUS "Building with CUDA and libcu++ support")
else()
    message(STATUS "Building CPU-only version")
endif()