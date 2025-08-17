cmake_minimum_required(VERSION 3.28)
project(hdf5_implementation)

set(CMAKE_CXX_STANDARD 20)

#find_package(HDF5 REQUIRED COMPONENTS C)
#find_package(MPI REQUIRED)

add_executable(hdf5_implementation
    include/jenkins/lookup3.cpp
    src/main.cpp
    src/util/lookup3.h
    src/serialization/serialization.h
    src/serialization/filestream.h
    src/serialization/buffer.h
    src/serialization/stdio.h
    src/hdf5/superblock.cpp
    src/hdf5/superblock.h
    src/hdf5/symbol_table.cpp
    src/hdf5/symbol_table.h
    src/hdf5/object_header.cpp
    src/hdf5/object_header.h
    src/hdf5/datatype.cpp
    src/hdf5/datatype.h
    src/hdf5/tree.cpp
    src/hdf5/tree.h
    src/hdf5/types.h
    src/hdf5/local_heap.h
    src/hdf5/local_heap.cpp
    src/hdf5/file.h
    src/hdf5/file.cpp
    src/hdf5/group.h
    src/hdf5/group.cpp
    src/hdf5/dataset.h
    src/hdf5/dataset.cpp
    src/hdf5/object.h
    src/hdf5/object.cpp
    src/hdf5/file_link.h
    src/hdf5/file_link.cpp
)

#target_link_libraries(hdf5_implementation PRIVATE MPI::MPI_CXX)
#target_include_directories(hdf5_implementation PRIVATE ${HDF5_INCLUDE_DIRS})
#target_link_libraries(hdf5_implementation PRIVATE ${HDF5_LIBRARIES})