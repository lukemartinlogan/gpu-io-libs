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

    # This uses system-installed yaml-cpp, cereal, etc.
    message(STATUS "Fetching IOWarp/core from GitHub...")
    FetchContent_Declare(
        iowarp_core
        GIT_REPOSITORY https://github.com/iowarp/core.git
        GIT_TAG        ed836a1  # 74-fix-context-transport-primitives-for-the-gpu with CUDA fixes
    )

    # Disable all components except context-transport-primitives (HermesShm)
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

    FetchContent_Populate(iowarp_core)

    list(APPEND CMAKE_MODULE_PATH "${iowarp_core_SOURCE_DIR}/cmake")

    add_subdirectory(${iowarp_core_SOURCE_DIR} ${iowarp_core_BINARY_DIR})
    message(STATUS "IOWarp/core built successfully")

    set(HSHM_INCLUDE_DIR "${iowarp_core_SOURCE_DIR}/context-transport-primitives/include")

    message(STATUS "Fetching CCCL from GitHub...")
    FetchContent_Declare(
        cccl
        GIT_REPOSITORY https://github.com/NVIDIA/cccl.git
        GIT_TAG        e5ed71c7dce1130e582db831fffc61f562203884 # 2025-12-13
    )
    FetchContent_MakeAvailable(cccl)
    message(STATUS "CCCL fetched successfully")

endif()

file(GLOB_RECURSE SOURCES
    "src/*.cpp"
    "src/*.h"
    "third_party/*.cpp"
    "third_party/*.h"
)

list(REMOVE_ITEM SOURCES "${CMAKE_SOURCE_DIR}/src/main.cpp")

if(CMAKE_CUDA_COMPILER)
    file(GLOB_RECURSE CUDA_SOURCES
        "src/*.cu"
    )
    list(APPEND SOURCES ${CUDA_SOURCES})

    file(GLOB_RECURSE HDF5_CPP_SOURCES "src/hdf5/*.cpp")
    set_source_files_properties(${HDF5_CPP_SOURCES} PROPERTIES LANGUAGE CUDA)

    # cpu_polling.cpp uses hermes_shm atomics which require CUDA compilation
    set_source_files_properties(
        "${CMAKE_SOURCE_DIR}/src/iowarp/cpu_polling.cpp"
        PROPERTIES LANGUAGE CUDA
    )
endif()

add_library(hdf5_lib STATIC ${SOURCES})

if(CMAKE_CUDA_COMPILER)
    set_target_properties(hdf5_lib PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS OFF
    )
    # Suppress NVCC warnings like iowarp does
    target_compile_options(hdf5_lib PUBLIC
        $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=177,20014,20011,20012>
    )
    target_link_libraries(hdf5_lib PUBLIC
        CUDA::cudart
        CCCL::CCCL
        hshm::cuda_cxx
        hshm::serialize
        hshm::lightbeam
    )
    target_compile_definitions(hdf5_lib PUBLIC
        CUDA_ENABLED=1
        LIBCUDACXX_AVAILABLE=1
        _CCCL_HAS_SORTING_ALGORITHMS=1
    )
    # hshm::cudacxx provides HSHM_ENABLE_CUDA and include directories
    target_include_directories(hdf5_lib PUBLIC
        ${CMAKE_SOURCE_DIR}/src
        ${cccl_SOURCE_DIR}/libcudacxx/include
        ${cccl_SOURCE_DIR}/thrust
        ${cccl_SOURCE_DIR}/cub
    )
    message(STATUS "Building with CUDA and libcu++ support")
else()
    target_include_directories(hdf5_lib PUBLIC
        ${CMAKE_SOURCE_DIR}/src
    )
    message(STATUS "Building CPU-only version")
endif()

add_executable(hdf5_implementation src/main.cpp)
target_link_libraries(hdf5_implementation PRIVATE hdf5_lib)

if(CMAKE_CUDA_COMPILER)
    # main.cpp includes hermes_shm headers which use CUDA intrinsics when HSHM_ENABLE_CUDA=1
    set_source_files_properties(src/main.cpp PROPERTIES LANGUAGE CUDA)
endif()

add_subdirectory(test/unit)
add_subdirectory(benches)