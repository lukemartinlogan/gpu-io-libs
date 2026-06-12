# CPU-only unit tests. No CUDA language, no iowarp runtime, no server process —
# just cuda::std headers (libcu++ via CCCL) + Catch2. These drive the same
# chunk/metadata logic the GPU producer path reuses.
add_executable(kvhdf5_unit_tests
    chunking_test.cpp
    dataset_io_test.cpp
)

set_target_properties(kvhdf5_unit_tests PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

target_include_directories(kvhdf5_unit_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src
)

target_link_libraries(kvhdf5_unit_tests PRIVATE
    Catch2::Catch2WithMain
    CCCL::CCCL
)

catch_discover_tests(kvhdf5_unit_tests)
