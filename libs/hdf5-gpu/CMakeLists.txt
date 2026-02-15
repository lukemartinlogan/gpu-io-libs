cmake_minimum_required(VERSION 3.28)

project(hdf5_gpu LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Force architecture 89 for RTX 4090 (required for scoped atomics)
set(CMAKE_CUDA_ARCHITECTURES "89" CACHE STRING "CUDA architectures" FORCE)

find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDA Toolkit ${CUDAToolkit_VERSION} found")

# Fetch dependencies
include(FetchContent)

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

message(STATUS "Fetching CCCL from GitHub...")
FetchContent_Declare(
    cccl
    GIT_REPOSITORY https://github.com/NVIDIA/cccl.git
    GIT_TAG        e5ed71c7dce1130e582db831fffc61f562203884 # 2025-12-13
)
FetchContent_MakeAvailable(cccl)
message(STATUS "CCCL fetched successfully")

# Collect library sources
file(GLOB_RECURSE SOURCES
    "src/*.cpp"
    "src/*.h"
    "third_party/*.cpp"
    "third_party/*.h"
)

list(REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")

file(GLOB_RECURSE CUDA_SOURCES "src/*.cu")
list(APPEND SOURCES ${CUDA_SOURCES})

# Set HDF5 cpp files to compile as CUDA
file(GLOB_RECURSE HDF5_CPP_SOURCES "src/hdf5/*.cpp")
set_source_files_properties(${HDF5_CPP_SOURCES} PROPERTIES LANGUAGE CUDA)

# cpu_polling.cpp uses hermes_shm atomics which require CUDA compilation
set_source_files_properties(
    "${CMAKE_CURRENT_SOURCE_DIR}/src/iowarp/cpu_polling.cpp"
    PROPERTIES LANGUAGE CUDA
)

# Create library
add_library(hdf5_gpu_lib STATIC ${SOURCES})

set_target_properties(hdf5_gpu_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
)

# Suppress NVCC warnings
target_compile_options(hdf5_gpu_lib PUBLIC
    $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=177,20014,20011,20012>
)

target_link_libraries(hdf5_gpu_lib PUBLIC
    CUDA::cudart
    CCCL::CCCL
    hshm::cuda_cxx
    hshm::serialize
    hshm::lightbeam
)

target_compile_definitions(hdf5_gpu_lib PUBLIC
    CUDA_ENABLED=1
    LIBCUDACXX_AVAILABLE=1
    _CCCL_HAS_SORTING_ALGORITHMS=1
)

target_include_directories(hdf5_gpu_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${cccl_SOURCE_DIR}/libcudacxx/include
    ${cccl_SOURCE_DIR}/thrust
    ${cccl_SOURCE_DIR}/cub
)

# Optional: build example executable
add_executable(hdf5_gpu src/main.cpp)
target_link_libraries(hdf5_gpu PRIVATE hdf5_gpu_lib)
set_source_files_properties(src/main.cpp PROPERTIES LANGUAGE CUDA)

message(STATUS "Building with CUDA and libcu++ support")

# Add subdirectories
add_subdirectory(test)
add_subdirectory(benches)
