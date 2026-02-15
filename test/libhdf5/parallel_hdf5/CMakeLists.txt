cmake_minimum_required(VERSION 3.28)
project(parallel_hdf5)

set(CMAKE_CXX_STANDARD 23)

find_package(HDF5 REQUIRED COMPONENTS C)
find_package(MPI REQUIRED)

add_executable(parallel_hdf5 src/main.cpp)

target_link_libraries(parallel_hdf5 PRIVATE MPI::MPI_CXX)
target_include_directories(parallel_hdf5 PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(parallel_hdf5 PRIVATE ${HDF5_LIBRARIES})