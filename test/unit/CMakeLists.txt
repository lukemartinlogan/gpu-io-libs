cmake_minimum_required(VERSION 3.28)

# IOWarp GPU Emulation Test
if(CMAKE_CUDA_COMPILER)
    add_executable(iowarp_gpu_emu_test
        iowarp_gpu_emu_test.cu
        ../../src/iowarp/cpu_polling.cpp
        ../../src/iowarp/gpu_posix.cu
    )

    target_include_directories(iowarp_gpu_emu_test PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${cccl_SOURCE_DIR}/libcudacxx/include
        ${cccl_SOURCE_DIR}/thrust
        ${cccl_SOURCE_DIR}/cub
    )

    target_link_libraries(iowarp_gpu_emu_test PRIVATE
        CUDA::cudart
        CCCL::CCCL
        stdgpu::stdgpu
    )

    target_compile_definitions(iowarp_gpu_emu_test PRIVATE
        LIBCUDACXX_AVAILABLE=1
        _CCCL_HAS_SORTING_ALGORITHMS=1
    )

    set_target_properties(iowarp_gpu_emu_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        CXX_STANDARD 20
        CUDA_STANDARD 20
    )

    message(STATUS "IOWarp GPU emulation test configured")

    # GPU HDF5 File Read Test
    add_executable(gpu_file_read_test
        gpu_file_read_test.cu
    )

    target_link_libraries(gpu_file_read_test PRIVATE
        hdf5_lib
    )

    set_target_properties(gpu_file_read_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        CXX_STANDARD 20
        CUDA_STANDARD 20
    )

    message(STATUS "GPU HDF5 file read test configured")

    # Minimal GPU Test
    add_executable(minimal_gpu_test
        minimal_gpu_test.cu
        ../../src/iowarp/cpu_polling.cpp
        ../../src/iowarp/gpu_posix.cu
    )

    target_include_directories(minimal_gpu_test PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${cccl_SOURCE_DIR}/libcudacxx/include
        ${cccl_SOURCE_DIR}/thrust
        ${cccl_SOURCE_DIR}/cub
    )

    target_link_libraries(minimal_gpu_test PRIVATE
        CUDA::cudart
        CCCL::CCCL
        stdgpu::stdgpu
    )

    target_compile_definitions(minimal_gpu_test PRIVATE
        LIBCUDACXX_AVAILABLE=1
        _CCCL_HAS_SORTING_ALGORITHMS=1
    )

    set_target_properties(minimal_gpu_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        CXX_STANDARD 20
        CUDA_STANDARD 20
    )

    message(STATUS "Minimal GPU test configured")

    # Minimal I/O Test
    add_executable(minimal_io_test
        minimal_io_test.cu
        ../../src/iowarp/cpu_polling.cpp
        ../../src/iowarp/gpu_posix.cu
    )

    target_include_directories(minimal_io_test PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${cccl_SOURCE_DIR}/libcudacxx/include
        ${cccl_SOURCE_DIR}/thrust
        ${cccl_SOURCE_DIR}/cub
    )

    target_link_libraries(minimal_io_test PRIVATE
        CUDA::cudart
        CCCL::CCCL
        stdgpu::stdgpu
    )

    target_compile_definitions(minimal_io_test PRIVATE
        LIBCUDACXX_AVAILABLE=1
        _CCCL_HAS_SORTING_ALGORITHMS=1
    )

    set_target_properties(minimal_io_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        CXX_STANDARD 20
        CUDA_STANDARD 20
    )

    message(STATUS "Minimal I/O test configured")

    # Simple File Test
    add_executable(simple_file_test
        simple_file_test.cu
    )

    target_link_libraries(simple_file_test PRIVATE
        hdf5_lib
    )

    set_target_properties(simple_file_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        CXX_STANDARD 20
        CUDA_STANDARD 20
    )

    message(STATUS "Simple file test configured")

    # Debug Kernel Test
    add_executable(debug_kernel_test
        debug_kernel_test.cu
    )

    target_link_libraries(debug_kernel_test PRIVATE
        CUDA::cudart
    )

    set_target_properties(debug_kernel_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        CXX_STANDARD 20
        CUDA_STANDARD 20
    )

    message(STATUS "Debug kernel test configured")

    # HDF5 Link Test
    add_executable(hdf5_link_test
        hdf5_link_test.cu
    )

    target_link_libraries(hdf5_link_test PRIVATE
        hdf5_lib
    )

    set_target_properties(hdf5_link_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        CXX_STANDARD 20
        CUDA_STANDARD 20
    )

    message(STATUS "HDF5 link test configured")

    # Link Only Test
    add_executable(link_only_test
        link_only_test.cu
    )

    target_link_libraries(link_only_test PRIVATE
        hdf5_lib
    )

    set_target_properties(link_only_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        CXX_STANDARD 20
        CUDA_STANDARD 20
    )

    message(STATUS "Link only test configured")
else()
    message(STATUS "Skipping IOWarp GPU emulation test (CUDA not available)")
endif()
