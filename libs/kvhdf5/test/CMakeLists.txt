include(FetchContent)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.8.0
)
FetchContent_MakeAvailable(Catch2)

list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)
include(Catch)

# Set CUDA architecture to match available GPU (RTX 4090 = compute 8.9)
set(CMAKE_CUDA_ARCHITECTURES "89" CACHE STRING "CUDA architectures" FORCE)

add_executable(kvhdf5_tests
    unit/gpu_string_test.cpp
    unit/serde_test.cpp
    unit/types_test.cpp
    unit/datatype_test.cpp
    unit/context_test.cpp
    unit/metadata_test.cpp
    unit/blob_store_test.cpp
    unit/container_test.cpp
    unit/container_complex_test.cpp
    unit/cte_blob_store_test.cpp
    integration/iowarp/vector_test.cpp
    integration/iowarp/cte_test.cpp
)

# When iowarp is built with CUDA enabled, its headers contain raw __device__/__host__
# keywords that only nvcc understands, so those test files must be compiled as CUDA.
# When iowarp is built without CUDA (WRP_CORE_ENABLE_CUDA=OFF), all those macros
# expand to empty via hermes_shm's macro guards, so plain g++ (CXX) works fine and
# avoids nvcc's device-pass issues (spurious __shared__ / atomic errors).
if(WRP_CORE_ENABLE_CUDA)
    set_source_files_properties(
        unit/context_test.cpp
        unit/metadata_test.cpp
        unit/blob_store_test.cpp
        unit/container_test.cpp
        unit/container_complex_test.cpp
        unit/cte_blob_store_test.cpp
        integration/iowarp/vector_test.cpp
        integration/iowarp/cte_test.cpp
        PROPERTIES LANGUAGE CUDA
    )
endif()

target_include_directories(kvhdf5_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/common
)

target_link_libraries(kvhdf5_tests PRIVATE
    kvhdf5
    Catch2::Catch2WithMain
)

# Set CUDA architecture for the test executable
set_target_properties(kvhdf5_tests PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
)

catch_discover_tests(kvhdf5_tests)
