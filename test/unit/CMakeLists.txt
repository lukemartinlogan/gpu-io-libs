cmake_minimum_required(VERSION 3.28)

if(CMAKE_CUDA_COMPILER)
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
        hshm::cudacxx
        hshm::serialize
        hshm::lightbeam
    )

    target_compile_definitions(gpu_file_read_test PRIVATE
        CUDA_ENABLED=1
        LIBCUDACXX_AVAILABLE=1
        _CCCL_HAS_SORTING_ALGORITHMS=1
        HSHM_ENABLE_PTHREADS=1
        HSHM_DEFAULT_THREAD_MODEL=hshm::thread::Pthread
        HSHM_DEFAULT_THREAD_MODEL_GPU=hshm::thread::Cuda
    )

    target_compile_options(gpu_file_read_test PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=177,20014,20011,20012>
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
        hshm::cudacxx
        hshm::serialize
        hshm::lightbeam
    )

    target_compile_definitions(minimal_gpu_test PRIVATE
        CUDA_ENABLED=1
        LIBCUDACXX_AVAILABLE=1
        _CCCL_HAS_SORTING_ALGORITHMS=1
        HSHM_ENABLE_PTHREADS=1
        HSHM_DEFAULT_THREAD_MODEL=hshm::thread::Pthread
        HSHM_DEFAULT_THREAD_MODEL_GPU=hshm::thread::Cuda
    )

    target_compile_options(minimal_gpu_test PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=177,20014,20011,20012>
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
        hshm::cudacxx
        hshm::serialize
        hshm::lightbeam
    )

    target_compile_definitions(minimal_io_test PRIVATE
        CUDA_ENABLED=1
        LIBCUDACXX_AVAILABLE=1
        _CCCL_HAS_SORTING_ALGORITHMS=1
        HSHM_ENABLE_PTHREADS=1
        HSHM_DEFAULT_THREAD_MODEL=hshm::thread::Pthread
        HSHM_DEFAULT_THREAD_MODEL_GPU=hshm::thread::Cuda
    )

    target_compile_options(minimal_io_test PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=177,20014,20011,20012>
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
        hshm::cudacxx
        hshm::serialize
        hshm::lightbeam
    )

    target_compile_definitions(hdf5_single_source_test PRIVATE
        LIBCUDACXX_AVAILABLE=1
        _CCCL_HAS_SORTING_ALGORITHMS=1
        HSHM_ENABLE_PTHREADS=1
        HSHM_DEFAULT_THREAD_MODEL=hshm::thread::Pthread
        HSHM_DEFAULT_THREAD_MODEL_GPU=hshm::thread::Cuda
    )

    set_target_properties(hdf5_single_source_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        CXX_STANDARD 20
        CUDA_STANDARD 20
    )

    message(STATUS "HDF5 single source test configured")

    # Minimal ICE Test (for debugging)
    add_executable(minimal_ice_test
        minimal_ice_test.cu
    )

    target_include_directories(minimal_ice_test PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${cccl_SOURCE_DIR}/libcudacxx/include
        ${cccl_SOURCE_DIR}/thrust
        ${cccl_SOURCE_DIR}/cub
    )

    target_link_libraries(minimal_ice_test PRIVATE
        CUDA::cudart
        CCCL::CCCL
        stdgpu::stdgpu
        hshm::cudacxx
        hshm::serialize
        hshm::lightbeam
    )

    target_compile_definitions(minimal_ice_test PRIVATE
        LIBCUDACXX_AVAILABLE=1
        _CCCL_HAS_SORTING_ALGORITHMS=1
        HSHM_ENABLE_PTHREADS=1
        HSHM_DEFAULT_THREAD_MODEL=hshm::thread::Pthread
        HSHM_DEFAULT_THREAD_MODEL_GPU=hshm::thread::Cuda
    )

    set_target_properties(minimal_ice_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        CXX_STANDARD 20
        CUDA_STANDARD 20
    )

    message(STATUS "Minimal ICE test configured")

    # ICE Bisect Test (systematic ICE debugging)
    add_executable(ice_bisect_test
        ice_bisect_test.cu
    )

    target_include_directories(ice_bisect_test PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${cccl_SOURCE_DIR}/libcudacxx/include
        ${cccl_SOURCE_DIR}/thrust
        ${cccl_SOURCE_DIR}/cub
    )

    target_link_libraries(ice_bisect_test PRIVATE
        CUDA::cudart
        CCCL::CCCL
        stdgpu::stdgpu
        hshm::cudacxx
        hshm::serialize
        hshm::lightbeam
    )

    target_compile_definitions(ice_bisect_test PRIVATE
        LIBCUDACXX_AVAILABLE=1
        _CCCL_HAS_SORTING_ALGORITHMS=1
        HSHM_ENABLE_PTHREADS=1
        HSHM_DEFAULT_THREAD_MODEL=hshm::thread::Pthread
        HSHM_DEFAULT_THREAD_MODEL_GPU=hshm::thread::Cuda
    )

    set_target_properties(ice_bisect_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        CXX_STANDARD 20
        CUDA_STANDARD 20
    )

    message(STATUS "ICE bisect test configured")

    # ICE Full Variant Test (test with HeaderMessageVariant)
    add_executable(ice_full_variant_test
        ice_full_variant_test.cu
    )

    target_include_directories(ice_full_variant_test PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${cccl_SOURCE_DIR}/libcudacxx/include
        ${cccl_SOURCE_DIR}/thrust
        ${cccl_SOURCE_DIR}/cub
    )

    target_link_libraries(ice_full_variant_test PRIVATE
        CUDA::cudart
        CCCL::CCCL
        stdgpu::stdgpu
        hshm::cudacxx
        hshm::serialize
        hshm::lightbeam
    )

    target_compile_definitions(ice_full_variant_test PRIVATE
        LIBCUDACXX_AVAILABLE=1
        _CCCL_HAS_SORTING_ALGORITHMS=1
        HSHM_ENABLE_PTHREADS=1
        HSHM_DEFAULT_THREAD_MODEL=hshm::thread::Pthread
        HSHM_DEFAULT_THREAD_MODEL_GPU=hshm::thread::Cuda
    )

    set_target_properties(ice_full_variant_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        CXX_STANDARD 20
        CUDA_STANDARD 20
    )

    message(STATUS "ICE full variant test configured")

    # ICE ObjectHeaderMessage Test
    add_executable(ice_object_header_msg_test
        ice_object_header_msg_test.cu
    )

    target_include_directories(ice_object_header_msg_test PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${cccl_SOURCE_DIR}/libcudacxx/include
        ${cccl_SOURCE_DIR}/thrust
        ${cccl_SOURCE_DIR}/cub
    )

    target_link_libraries(ice_object_header_msg_test PRIVATE
        CUDA::cudart
        CCCL::CCCL
        stdgpu::stdgpu
        hshm::cudacxx
        hshm::serialize
        hshm::lightbeam
    )

    target_compile_definitions(ice_object_header_msg_test PRIVATE
        LIBCUDACXX_AVAILABLE=1
        _CCCL_HAS_SORTING_ALGORITHMS=1
        HSHM_ENABLE_PTHREADS=1
        HSHM_DEFAULT_THREAD_MODEL=hshm::thread::Pthread
        HSHM_DEFAULT_THREAD_MODEL_GPU=hshm::thread::Cuda
    )

    set_target_properties(ice_object_header_msg_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        CXX_STANDARD 20
        CUDA_STANDARD 20
    )

    message(STATUS "ICE object header message test configured")

    # ICE Variant Plus Field Test
    add_executable(ice_variant_plus_field_test
        ice_variant_plus_field_test.cu
    )

    target_include_directories(ice_variant_plus_field_test PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${cccl_SOURCE_DIR}/libcudacxx/include
        ${cccl_SOURCE_DIR}/thrust
        ${cccl_SOURCE_DIR}/cub
    )

    target_link_libraries(ice_variant_plus_field_test PRIVATE
        CUDA::cudart
        CCCL::CCCL
        stdgpu::stdgpu
        hshm::cudacxx
        hshm::serialize
        hshm::lightbeam
    )

    target_compile_definitions(ice_variant_plus_field_test PRIVATE
        LIBCUDACXX_AVAILABLE=1
        _CCCL_HAS_SORTING_ALGORITHMS=1
        HSHM_ENABLE_PTHREADS=1
        HSHM_DEFAULT_THREAD_MODEL=hshm::thread::Pthread
        HSHM_DEFAULT_THREAD_MODEL_GPU=hshm::thread::Cuda
    )

    set_target_properties(ice_variant_plus_field_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        CXX_STANDARD 20
        CUDA_STANDARD 20
    )

    message(STATUS "ICE variant plus field test configured")

    # ICE Small Variant Plus Field Test
    add_executable(ice_small_variant_plus_field_test
        ice_small_variant_plus_field_test.cu
    )

    target_include_directories(ice_small_variant_plus_field_test PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${cccl_SOURCE_DIR}/libcudacxx/include
        ${cccl_SOURCE_DIR}/thrust
        ${cccl_SOURCE_DIR}/cub
    )

    target_link_libraries(ice_small_variant_plus_field_test PRIVATE
        CUDA::cudart
        CCCL::CCCL
        stdgpu::stdgpu
        hshm::cudacxx
        hshm::serialize
        hshm::lightbeam
    )

    target_compile_definitions(ice_small_variant_plus_field_test PRIVATE
        LIBCUDACXX_AVAILABLE=1
        _CCCL_HAS_SORTING_ALGORITHMS=1
        HSHM_ENABLE_PTHREADS=1
        HSHM_DEFAULT_THREAD_MODEL=hshm::thread::Pthread
        HSHM_DEFAULT_THREAD_MODEL_GPU=hshm::thread::Cuda
    )

    set_target_properties(ice_small_variant_plus_field_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        CXX_STANDARD 20
        CUDA_STANDARD 20
    )

    message(STATUS "ICE small variant plus field test configured")

    # ICE Size Threshold Test
    add_executable(ice_size_threshold_test
        ice_size_threshold_test.cu
    )

    target_include_directories(ice_size_threshold_test PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${cccl_SOURCE_DIR}/libcudacxx/include
        ${cccl_SOURCE_DIR}/thrust
        ${cccl_SOURCE_DIR}/cub
    )

    target_link_libraries(ice_size_threshold_test PRIVATE
        CUDA::cudart
        CCCL::CCCL
        stdgpu::stdgpu
        hshm::cudacxx
        hshm::serialize
        hshm::lightbeam
    )

    target_compile_definitions(ice_size_threshold_test PRIVATE
        LIBCUDACXX_AVAILABLE=1
        _CCCL_HAS_SORTING_ALGORITHMS=1
        HSHM_ENABLE_PTHREADS=1
        HSHM_DEFAULT_THREAD_MODEL=hshm::thread::Pthread
        HSHM_DEFAULT_THREAD_MODEL_GPU=hshm::thread::Cuda
    )

    set_target_properties(ice_size_threshold_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        CXX_STANDARD 20
        CUDA_STANDARD 20
    )

    message(STATUS "ICE size threshold test configured")

    # ICE Vector Test
    add_executable(ice_vector_test
        ice_vector_test.cu
    )

    target_include_directories(ice_vector_test PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${cccl_SOURCE_DIR}/libcudacxx/include
        ${cccl_SOURCE_DIR}/thrust
        ${cccl_SOURCE_DIR}/cub
    )

    target_link_libraries(ice_vector_test PRIVATE
        CUDA::cudart
        CCCL::CCCL
        stdgpu::stdgpu
        hshm::cudacxx
        hshm::serialize
        hshm::lightbeam
    )

    target_compile_definitions(ice_vector_test PRIVATE
        LIBCUDACXX_AVAILABLE=1
        _CCCL_HAS_SORTING_ALGORITHMS=1
        HSHM_ENABLE_PTHREADS=1
        HSHM_DEFAULT_THREAD_MODEL=hshm::thread::Pthread
        HSHM_DEFAULT_THREAD_MODEL_GPU=hshm::thread::Cuda
    )

    set_target_properties(ice_vector_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        CXX_STANDARD 20
        CUDA_STANDARD 20
    )

    message(STATUS "ICE vector test configured")

    # ICE Alignment Test (host-only, no device code)
    add_executable(ice_alignment_test
        ice_alignment_test.cu
    )

    target_include_directories(ice_alignment_test PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${cccl_SOURCE_DIR}/libcudacxx/include
        ${cccl_SOURCE_DIR}/thrust
        ${cccl_SOURCE_DIR}/cub
    )

    target_link_libraries(ice_alignment_test PRIVATE
        CUDA::cudart
        CCCL::CCCL
        stdgpu::stdgpu
        hshm::cudacxx
        hshm::serialize
        hshm::lightbeam
    )

    target_compile_definitions(ice_alignment_test PRIVATE
        LIBCUDACXX_AVAILABLE=1
        _CCCL_HAS_SORTING_ALGORITHMS=1
        HSHM_ENABLE_PTHREADS=1
        HSHM_DEFAULT_THREAD_MODEL=hshm::thread::Pthread
        HSHM_DEFAULT_THREAD_MODEL_GPU=hshm::thread::Cuda
    )

    set_target_properties(ice_alignment_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        CXX_STANDARD 20
        CUDA_STANDARD 20
    )

    message(STATUS "ICE alignment test configured")

    # ICE Field Alignment Test
    add_executable(ice_field_alignment_test
        ice_field_alignment_test.cu
    )

    target_include_directories(ice_field_alignment_test PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${cccl_SOURCE_DIR}/libcudacxx/include
        ${cccl_SOURCE_DIR}/thrust
        ${cccl_SOURCE_DIR}/cub
    )

    target_link_libraries(ice_field_alignment_test PRIVATE
        CUDA::cudart
        CCCL::CCCL
        stdgpu::stdgpu
        hshm::cudacxx
        hshm::serialize
        hshm::lightbeam
    )

    target_compile_definitions(ice_field_alignment_test PRIVATE
        LIBCUDACXX_AVAILABLE=1
        _CCCL_HAS_SORTING_ALGORITHMS=1
        HSHM_ENABLE_PTHREADS=1
        HSHM_DEFAULT_THREAD_MODEL=hshm::thread::Pthread
        HSHM_DEFAULT_THREAD_MODEL_GPU=hshm::thread::Cuda
    )

    set_target_properties(ice_field_alignment_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        CXX_STANDARD 20
        CUDA_STANDARD 20
    )

    message(STATUS "ICE field alignment test configured")

    # ICE Padding Test
    add_executable(ice_padding_test
        ice_padding_test.cu
    )

    target_include_directories(ice_padding_test PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${cccl_SOURCE_DIR}/libcudacxx/include
        ${cccl_SOURCE_DIR}/thrust
        ${cccl_SOURCE_DIR}/cub
    )

    target_link_libraries(ice_padding_test PRIVATE
        CUDA::cudart
        CCCL::CCCL
        stdgpu::stdgpu
        hshm::cudacxx
        hshm::serialize
        hshm::lightbeam
    )

    target_compile_definitions(ice_padding_test PRIVATE
        LIBCUDACXX_AVAILABLE=1
        _CCCL_HAS_SORTING_ALGORITHMS=1
        HSHM_ENABLE_PTHREADS=1
        HSHM_DEFAULT_THREAD_MODEL=hshm::thread::Pthread
        HSHM_DEFAULT_THREAD_MODEL_GPU=hshm::thread::Cuda
    )

    set_target_properties(ice_padding_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        CXX_STANDARD 20
        CUDA_STANDARD 20
    )

    message(STATUS "ICE padding test configured")

    # ICE Group Test
    add_executable(ice_group_test
        ice_group_test.cu
    )

    target_include_directories(ice_group_test PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${cccl_SOURCE_DIR}/libcudacxx/include
        ${cccl_SOURCE_DIR}/thrust
        ${cccl_SOURCE_DIR}/cub
    )

    target_link_libraries(ice_group_test PRIVATE
        CUDA::cudart
        CCCL::CCCL
        stdgpu::stdgpu
        hshm::cudacxx
        hshm::serialize
        hshm::lightbeam
    )

    target_compile_definitions(ice_group_test PRIVATE
        LIBCUDACXX_AVAILABLE=1
        _CCCL_HAS_SORTING_ALGORITHMS=1
        HSHM_ENABLE_PTHREADS=1
        HSHM_DEFAULT_THREAD_MODEL=hshm::thread::Pthread
        HSHM_DEFAULT_THREAD_MODEL_GPU=hshm::thread::Cuda
    )

    set_target_properties(ice_group_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        CXX_STANDARD 20
        CUDA_STANDARD 20
    )

    message(STATUS "ICE group test configured")

    # HDF5 Comprehensive Test (full read/write/create test)
    file(GLOB_RECURSE HDF5_TEST_SOURCES
        ${CMAKE_SOURCE_DIR}/src/hdf5/*.cpp
    )

    add_executable(hdf5_comprehensive_test
        hdf5_comprehensive_test.cu
        ${HDF5_TEST_SOURCES}
        ${CMAKE_SOURCE_DIR}/src/iowarp/gpu_posix.cu
        ${CMAKE_SOURCE_DIR}/src/iowarp/cpu_polling.cpp
    )

    set_source_files_properties(${HDF5_TEST_SOURCES} PROPERTIES LANGUAGE CUDA)

    target_include_directories(hdf5_comprehensive_test PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${cccl_SOURCE_DIR}/libcudacxx/include
        ${cccl_SOURCE_DIR}/thrust
        ${cccl_SOURCE_DIR}/cub
    )

    target_link_libraries(hdf5_comprehensive_test PRIVATE
        CUDA::cudart
        CCCL::CCCL
        stdgpu::stdgpu
        hshm::cudacxx
        hshm::serialize
        hshm::lightbeam
    )

    target_compile_definitions(hdf5_comprehensive_test PRIVATE
        CUDA_ENABLED=1
        LIBCUDACXX_AVAILABLE=1
        _CCCL_HAS_SORTING_ALGORITHMS=1
        HSHM_ENABLE_PTHREADS=1
        HSHM_DEFAULT_THREAD_MODEL=hshm::thread::Pthread
        HSHM_DEFAULT_THREAD_MODEL_GPU=hshm::thread::Cuda
    )

    target_compile_options(hdf5_comprehensive_test PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=177,20014,20011,20012>
    )

    set_target_properties(hdf5_comprehensive_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        CXX_STANDARD 20
        CUDA_STANDARD 20
    )

    message(STATUS "HDF5 comprehensive test configured")
else()
    message(STATUS "Skipping IOWarp GPU emulation test (CUDA not available)")
endif()
