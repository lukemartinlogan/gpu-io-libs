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
    # Compile HDF5 sources directly to avoid device linking conflicts
    file(GLOB_RECURSE HDF5_SOURCES
        ${CMAKE_SOURCE_DIR}/src/hdf5/*.cpp
    )

    add_executable(gpu_file_read_test
        gpu_file_read_test.cu
        ${HDF5_SOURCES}
        ${CMAKE_SOURCE_DIR}/src/iowarp/gpu_posix.cu
        ${CMAKE_SOURCE_DIR}/src/iowarp/cpu_polling.cpp
    )

    # Set HDF5 cpp files to compile as CUDA
    set_source_files_properties(${HDF5_SOURCES} PROPERTIES LANGUAGE CUDA)

    target_include_directories(gpu_file_read_test PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${cccl_SOURCE_DIR}/libcudacxx/include
        ${cccl_SOURCE_DIR}/thrust
        ${cccl_SOURCE_DIR}/cub
    )

    target_link_libraries(gpu_file_read_test PRIVATE
        CUDA::cudart
        CCCL::CCCL
        stdgpu::stdgpu
    )

    target_compile_definitions(gpu_file_read_test PRIVATE
        CUDA_ENABLED=1
        LIBCUDACXX_AVAILABLE=1
        _CCCL_HAS_SORTING_ALGORITHMS=1
    )

    set_target_properties(gpu_file_read_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
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
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        CXX_STANDARD 20
        CUDA_STANDARD 20
    )

    message(STATUS "Simple file test configured")

    # HDF5 Link Test
    add_executable(hdf5_link_test
        hdf5_link_test.cu
    )

    target_link_libraries(hdf5_link_test PRIVATE
        hdf5_lib
    )

    set_target_properties(hdf5_link_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
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
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        CXX_STANDARD 20
        CUDA_STANDARD 20
    )

    message(STATUS "Link only test configured")

    # Minimal Include Test
    add_executable(minimal_include_test
        minimal_include_test.cu
    )

    target_include_directories(minimal_include_test PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${cccl_SOURCE_DIR}/libcudacxx/include
        ${cccl_SOURCE_DIR}/thrust
        ${cccl_SOURCE_DIR}/cub
    )

    target_link_libraries(minimal_include_test PRIVATE
        hdf5_lib
    )

    target_compile_definitions(minimal_include_test PRIVATE
        LIBCUDACXX_AVAILABLE=1
        _CCCL_HAS_SORTING_ALGORITHMS=1
    )

    set_target_properties(minimal_include_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        CXX_STANDARD 20
        CUDA_STANDARD 20
    )

    message(STATUS "Minimal include test configured")

    # CSTD Only Test (no hdf5_lib link)
    add_executable(cstd_only_test
        cstd_only_test.cu
    )

    target_include_directories(cstd_only_test PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${cccl_SOURCE_DIR}/libcudacxx/include
        ${cccl_SOURCE_DIR}/thrust
        ${cccl_SOURCE_DIR}/cub
    )

    target_link_libraries(cstd_only_test PRIVATE
        CUDA::cudart
        CCCL::CCCL
    )

    target_compile_definitions(cstd_only_test PRIVATE
        LIBCUDACXX_AVAILABLE=1
        _CCCL_HAS_SORTING_ALGORITHMS=1
    )

    set_target_properties(cstd_only_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        CXX_STANDARD 20
        CUDA_STANDARD 20
    )

    message(STATUS "CSTD only test configured")

    # No Atomic Test (links hdf5_lib but test doesn't use atomic)
    add_executable(no_atomic_test
        no_atomic_test.cu
    )

    target_include_directories(no_atomic_test PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${cccl_SOURCE_DIR}/libcudacxx/include
        ${cccl_SOURCE_DIR}/thrust
        ${cccl_SOURCE_DIR}/cub
    )

    target_link_libraries(no_atomic_test PRIVATE
        hdf5_lib
    )

    target_compile_definitions(no_atomic_test PRIVATE
        LIBCUDACXX_AVAILABLE=1
        _CCCL_HAS_SORTING_ALGORITHMS=1
    )

    set_target_properties(no_atomic_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        CXX_STANDARD 20
        CUDA_STANDARD 20
    )

    message(STATUS "No atomic test configured")

    # HDF5 Minimal Test (header-only, no sources)
    add_executable(hdf5_minimal_test
        hdf5_minimal_test.cu
    )

    target_include_directories(hdf5_minimal_test PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${cccl_SOURCE_DIR}/libcudacxx/include
        ${cccl_SOURCE_DIR}/thrust
        ${cccl_SOURCE_DIR}/cub
    )

    target_link_libraries(hdf5_minimal_test PRIVATE
        CUDA::cudart
        CCCL::CCCL
        stdgpu::stdgpu
    )

    target_compile_definitions(hdf5_minimal_test PRIVATE
        LIBCUDACXX_AVAILABLE=1
        _CCCL_HAS_SORTING_ALGORITHMS=1
    )

    set_target_properties(hdf5_minimal_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        CXX_STANDARD 20
        CUDA_STANDARD 20
    )

    message(STATUS "HDF5 minimal test configured")

    # HDF5 Single Source Test (binary search for hang)
    add_executable(hdf5_single_source_test
        hdf5_single_source_test.cu
        ${CMAKE_SOURCE_DIR}/src/hdf5/datatype.cpp
        ${CMAKE_SOURCE_DIR}/src/hdf5/file.cpp
        ${CMAKE_SOURCE_DIR}/src/hdf5/superblock.cpp
        ${CMAKE_SOURCE_DIR}/src/hdf5/file_link.cpp
        ${CMAKE_SOURCE_DIR}/src/hdf5/group.cpp
        ${CMAKE_SOURCE_DIR}/src/hdf5/symbol_table.cpp
        ${CMAKE_SOURCE_DIR}/src/hdf5/local_heap.cpp
        ${CMAKE_SOURCE_DIR}/src/hdf5/object.cpp
        ${CMAKE_SOURCE_DIR}/src/hdf5/object_header.cpp
        ${CMAKE_SOURCE_DIR}/src/hdf5/tree.cpp
        # Testing: add dataset
        ${CMAKE_SOURCE_DIR}/src/hdf5/dataset.cpp
    )

    set_source_files_properties(
        ${CMAKE_SOURCE_DIR}/src/hdf5/datatype.cpp
        ${CMAKE_SOURCE_DIR}/src/hdf5/file.cpp
        ${CMAKE_SOURCE_DIR}/src/hdf5/superblock.cpp
        ${CMAKE_SOURCE_DIR}/src/hdf5/file_link.cpp
        ${CMAKE_SOURCE_DIR}/src/hdf5/group.cpp
        ${CMAKE_SOURCE_DIR}/src/hdf5/symbol_table.cpp
        ${CMAKE_SOURCE_DIR}/src/hdf5/local_heap.cpp
        ${CMAKE_SOURCE_DIR}/src/hdf5/object.cpp
        ${CMAKE_SOURCE_DIR}/src/hdf5/object_header.cpp
        ${CMAKE_SOURCE_DIR}/src/hdf5/tree.cpp
        ${CMAKE_SOURCE_DIR}/src/hdf5/dataset.cpp
        PROPERTIES LANGUAGE CUDA
    )

    target_include_directories(hdf5_single_source_test PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${cccl_SOURCE_DIR}/libcudacxx/include
        ${cccl_SOURCE_DIR}/thrust
        ${cccl_SOURCE_DIR}/cub
    )

    target_link_libraries(hdf5_single_source_test PRIVATE
        CUDA::cudart
        CCCL::CCCL
        stdgpu::stdgpu
    )

    target_compile_definitions(hdf5_single_source_test PRIVATE
        LIBCUDACXX_AVAILABLE=1
        _CCCL_HAS_SORTING_ALGORITHMS=1
    )

    set_target_properties(hdf5_single_source_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        CXX_STANDARD 20
        CUDA_STANDARD 20
    )

    message(STATUS "HDF5 single source test configured")
else()
    message(STATUS "Skipping IOWarp GPU emulation test (CUDA not available)")
endif()
