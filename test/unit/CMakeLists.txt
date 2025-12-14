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
else()
    message(STATUS "Skipping IOWarp GPU emulation test (CUDA not available)")
endif()
