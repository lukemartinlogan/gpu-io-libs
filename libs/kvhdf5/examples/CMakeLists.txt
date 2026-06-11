add_executable(kvhdf5_test_init
    test_init.cc
)

target_include_directories(kvhdf5_test_init PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../test/integration/iowarp/gpu
    ${CMAKE_CURRENT_SOURCE_DIR}/../test/common
)

target_link_libraries(kvhdf5_test_init PRIVATE kvhdf5)

target_compile_definitions(kvhdf5_test_init PRIVATE
    KVHDF5_CHIMAERA_CONF="${KVHDF5_CHIMAERA_CONF_PATH}"
)

add_subdirectory(gray_scott_iowarp)
add_subdirectory(gray_scott_traditional)
