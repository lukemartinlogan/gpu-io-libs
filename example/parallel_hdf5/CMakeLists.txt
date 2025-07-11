cmake_minimum_required(VERSION 3.28)
project(parallel_hdf5)

set(CMAKE_CXX_STANDARD 23)

find_package(HDF5 REQUIRED COMPONENTS C CXX HL)
find_package(MPI REQUIRED)

target_link_libraries(parallel_hdf5 PRIVATE MPI::MPI_CXX)