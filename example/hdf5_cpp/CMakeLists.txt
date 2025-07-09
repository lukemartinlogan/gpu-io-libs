cmake_minimum_required(VERSION 3.28)
project(hdf5_test)

set(CMAKE_CXX_STANDARD 23)

find_package(HDF5 REQUIRED COMPONENTS C CXX HL)

add_executable(hdf5_test src/main.cpp
        src/sample.h
        src/hwinfo.h)

target_include_directories(hdf5_test PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(hdf5_test PRIVATE ${HDF5_LIBRARIES})
