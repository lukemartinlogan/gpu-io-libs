include(FetchContent)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.8.0
)
FetchContent_MakeAvailable(Catch2)

list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)
include(Catch)

add_executable(kvhdf5_tests
    unit/gpu_string_test.cpp
    unit/serde_test.cpp
    unit/types_test.cpp
    unit/datatype_test.cpp
    unit/context_test.cpp
    unit/metadata_test.cpp
    integration/iowarp/vector_test.cpp
)

# Compile iowarp-related tests as CUDA (they use hermes_shm atomics)
set_source_files_properties(
    unit/context_test.cpp
    unit/metadata_test.cpp
    integration/iowarp/vector_test.cpp
    PROPERTIES LANGUAGE CUDA
)

target_link_libraries(kvhdf5_tests PRIVATE
    kvhdf5
    Catch2::Catch2WithMain
)

catch_discover_tests(kvhdf5_tests)
