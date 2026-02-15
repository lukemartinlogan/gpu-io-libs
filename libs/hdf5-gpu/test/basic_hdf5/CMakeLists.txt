cmake_minimum_required(VERSION 3.28)
project(basic_hdf5)

set(CMAKE_CXX_STANDARD 23)

find_package(HDF5 REQUIRED COMPONENTS C CXX HL)

add_executable(basic_hdf5 src/main.cpp src/sample.h src/hwinfo.h)

target_include_directories(basic_hdf5 PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(basic_hdf5 PRIVATE ${HDF5_LIBRARIES})
