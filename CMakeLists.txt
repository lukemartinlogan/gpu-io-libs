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

    # required by IOWarp HermesShm
    find_package(yaml-cpp CONFIG REQUIRED)
    find_package(cereal CONFIG REQUIRED)

    find_package(HermesShmCore CONFIG REQUIRED)
    message(STATUS "IOWarp HermesShm found")

    # patch hshm::priv::vector with CUDA device annotations
    set(HSHM_VECTOR_HEADER "/usr/local/include/hermes_shm/data_structures/priv/vector.h")
    set(HSHM_PATCH_FILE "${CMAKE_SOURCE_DIR}/third_party/iowarp-patch/hshm_vector_cuda.patch")
    set(HSHM_PATCH_MARKER "${CMAKE_BINARY_DIR}/.hshm_vector_patched")

    if(EXISTS "${HSHM_VECTOR_HEADER}" AND EXISTS "${HSHM_PATCH_FILE}" AND NOT EXISTS "${HSHM_PATCH_MARKER}")
        message(STATUS "Patching hshm::priv::vector for CUDA support...")
        execute_process(
            COMMAND sudo patch -N -p0 -i "${HSHM_PATCH_FILE}"
            WORKING_DIRECTORY /
            RESULT_VARIABLE PATCH_RESULT
            OUTPUT_VARIABLE PATCH_OUTPUT
            ERROR_VARIABLE PATCH_ERROR
        )
        if(PATCH_RESULT EQUAL 0)
            message(STATUS "hshm::priv::vector patched successfully")
            file(TOUCH "${HSHM_PATCH_MARKER}")
        else()
            message(WARNING "Failed to patch hshm::priv::vector: ${PATCH_ERROR}")
            message(WARNING "Patch output: ${PATCH_OUTPUT}")
        endif()
    elseif(EXISTS "${HSHM_PATCH_MARKER}")
        message(STATUS "hshm::priv::vector already patched (marker found)")
    endif()

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
        hshm::cudacxx
        hshm::serialize
        hshm::lightbeam
    )
    target_compile_definitions(hdf5_lib PUBLIC
        CUDA_ENABLED=1
        LIBCUDACXX_AVAILABLE=1
        _CCCL_HAS_SORTING_ALGORITHMS=1
        # Required by hermes_shm (missing from installed cmake config)
        HSHM_ENABLE_PTHREADS=1
        HSHM_DEFAULT_THREAD_MODEL=hshm::thread::Pthread
        HSHM_DEFAULT_THREAD_MODEL_GPU=hshm::thread::Cuda
    )
    # Hermes_shm include path MUST come BEFORE CCCL to avoid conflicts
    target_include_directories(hdf5_lib SYSTEM PUBLIC /usr/local/include)
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

add_subdirectory(test/unit)