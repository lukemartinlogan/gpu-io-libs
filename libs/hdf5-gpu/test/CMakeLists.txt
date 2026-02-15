cmake_minimum_required(VERSION 3.28)

if(CMAKE_CUDA_COMPILER)
    # GPU HDF5 File Read Test
    # Compile HDF5 sources directly to avoid device linking conflicts
    file(GLOB_RECURSE HDF5_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/hdf5/*.cpp
    )

    add_executable(gpu_file_read_test
        gpu_file_read_test.cu
        ${HDF5_SOURCES}
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/iowarp/gpu_posix.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/iowarp/cpu_polling.cpp
    )

    # Set HDF5 cpp files to compile as CUDA
    set_source_files_properties(${HDF5_SOURCES} PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/../src/iowarp/cpu_polling.cpp PROPERTIES LANGUAGE CUDA)

    target_include_directories(gpu_file_read_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../src
        ${cccl_SOURCE_DIR}/libcudacxx/include
        ${cccl_SOURCE_DIR}/thrust
        ${cccl_SOURCE_DIR}/cub
    )

    target_link_libraries(gpu_file_read_test PRIVATE
        CUDA::cudart
        CCCL::CCCL
        hshm::cuda_cxx
        hshm::serialize
        hshm::lightbeam
    )

    target_compile_definitions(gpu_file_read_test PRIVATE
        CUDA_ENABLED=1
        LIBCUDACXX_AVAILABLE=1
        _CCCL_HAS_SORTING_ALGORITHMS=1
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
    set_source_files_properties(${CMAKE_SOURCE_DIR}/src/iowarp/cpu_polling.cpp PROPERTIES LANGUAGE CUDA)

    target_include_directories(hdf5_comprehensive_test PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${cccl_SOURCE_DIR}/libcudacxx/include
        ${cccl_SOURCE_DIR}/thrust
        ${cccl_SOURCE_DIR}/cub
    )

    target_link_libraries(hdf5_comprehensive_test PRIVATE
        CUDA::cudart
        CCCL::CCCL
        hshm::cuda_cxx
        hshm::serialize
        hshm::lightbeam
    )

    target_compile_definitions(hdf5_comprehensive_test PRIVATE
        CUDA_ENABLED=1
        LIBCUDACXX_AVAILABLE=1
        _CCCL_HAS_SORTING_ALGORITHMS=1
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
