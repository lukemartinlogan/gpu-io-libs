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
    gpu_string_test.cpp
    serde_test.cpp
    types_test.cpp
)

target_link_libraries(kvhdf5_tests PRIVATE
    kvhdf5
    Catch2::Catch2WithMain
)

catch_discover_tests(kvhdf5_tests)
