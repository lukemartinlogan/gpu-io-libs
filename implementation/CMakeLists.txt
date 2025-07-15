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
)

#target_link_libraries(hdf5_implementation PRIVATE MPI::MPI_CXX)
#target_include_directories(hdf5_implementation PRIVATE ${HDF5_INCLUDE_DIRS})
#target_link_libraries(hdf5_implementation PRIVATE ${HDF5_LIBRARIES})