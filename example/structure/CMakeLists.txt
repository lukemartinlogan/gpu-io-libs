cmake_minimum_required(VERSION 3.28)
project(hdf5_structure)

set(CMAKE_CXX_STANDARD 23)

#if(EXISTS ${CMAKE_SOURCE_DIR}/.env.cmake)
#    include(.env.cmake)
#endif()

#find_package(HDF5 REQUIRED COMPONENTS C)
#find_package(MPI REQUIRED)

add_executable(hdf5_structure src/main.cpp)

#target_link_libraries(hdf5_structure PRIVATE MPI::MPI_CXX)
#target_include_directories(hdf5_structure PRIVATE ${HDF5_INCLUDE_DIRS})
#target_link_libraries(hdf5_structure PRIVATE ${HDF5_LIBRARIES})