include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.8.0
)
FetchContent_MakeAvailable(Catch2)

list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)
include(Catch)

# Match the GPU available in the dev container (RTX 4090 = compute 8.9).
set(CMAKE_CUDA_ARCHITECTURES "89" CACHE STRING "CUDA architectures" FORCE)

add_subdirectory(integration/iowarp/gpu)
