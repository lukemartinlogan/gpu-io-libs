cmake_minimum_required(VERSION 3.28)
project(hdf5_implementation)

set(CMAKE_CXX_STANDARD 20)

#find_package(HDF5 REQUIRED COMPONENTS C)
#find_package(MPI REQUIRED)

file(GLOB_RECURSE SOURCES
    "src/*.cpp"
    "src/*.h"
    "include/*.cpp"
    "include/*.h"
)

#target_link_libraries(hdf5_implementation PRIVATE MPI::MPI_CXX)
#target_include_directories(hdf5_implementation PRIVATE ${HDF5_INCLUDE_DIRS})
#target_link_libraries(hdf5_implementation PRIVATE ${HDF5_LIBRARIES})
add_executable(hdf5_implementation ${SOURCES})