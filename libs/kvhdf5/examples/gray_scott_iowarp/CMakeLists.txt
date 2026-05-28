if(NOT KVHDF5_HAS_CUDA)
    message(STATUS "kvhdf5_gray_scott: skipped (no CUDA compiler)")
    return()
endif()

add_executable(kvhdf5_gray_scott
    gray_scott_gpu.cu
    gray_scott_host.cc
    gray_scott_init.cc
)

set_target_properties(kvhdf5_gray_scott PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION ON
)

# Same -gline-tables-only workaround the GPU tests use: clang-18 full DWARF
# trips two ptxas bugs (dot-in-symbol-name from cuda::std::span and typed-
# pointer isspacep.shared from hshm::priv::vector). Line tables only avoids
# both.
set_source_files_properties(gray_scott_gpu.cu
    PROPERTIES COMPILE_FLAGS "-gline-tables-only"
)

target_include_directories(kvhdf5_gray_scott PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../test/integration/iowarp/gpu
    ${CMAKE_CURRENT_SOURCE_DIR}/../../test/common
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
)

target_link_libraries(kvhdf5_gray_scott PRIVATE kvhdf5)

target_compile_definitions(kvhdf5_gray_scott PRIVATE
    KVHDF5_CHIMAERA_CONF="${KVHDF5_CHIMAERA_CONF_PATH}"
    HSHM_ENABLE_CUDA=1
)
