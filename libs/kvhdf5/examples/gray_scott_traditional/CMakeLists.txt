if(NOT KVHDF5_HAS_CUDA)
    message(STATUS "kvhdf5_gray_scott_traditional: skipped (no CUDA compiler)")
    return()
endif()

add_executable(kvhdf5_gray_scott_traditional
    main.cc
    gs_kernel.cu
    ram_backend.cc
    disk_backend.cc
)

set_target_properties(kvhdf5_gray_scott_traditional PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION ON
)

# The non-CUDA translation units still need cudaStream_t / cudaMalloc etc. from
# <cuda_runtime.h>. CMake's CUDA toolkit module exposes the include paths via
# CUDAToolkit::cudart_static (or the CUDA_TOOLKIT_INCLUDE_DIRECTORIES variable
# in older setups). Use CUDA::cudart for both include paths and link.
find_package(CUDAToolkit REQUIRED)

target_link_libraries(kvhdf5_gray_scott_traditional PRIVATE
    CUDA::cudart
)

target_compile_features(kvhdf5_gray_scott_traditional PRIVATE cxx_std_20)
