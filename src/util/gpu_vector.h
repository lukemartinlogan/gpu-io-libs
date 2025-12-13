#pragma once

#include "../hdf5/cstd.h"
#include "../hdf5/error.h"

// Only use stdgpu when compiling with CUDA compiler (nvcc)
// __CUDACC__ is defined by nvcc, not by regular C++ compiler
#if defined(LIBCUDACXX_AVAILABLE) && defined(__CUDACC__)
    #include <stdgpu/vector.cuh>

    namespace hdf5 {
        // Dynamic vector for GPU - use when size is truly unpredictable
        // Note: Requires explicit creation/destruction
        template<typename T>
        using gpu_vector = stdgpu::vector<T>;
    }
#else
    #include <vector>

    namespace hdf5 {
        // Fallback: use std::vector when not compiling with CUDA
        template<typename T>
        using gpu_vector = std::vector<T>;
    }
#endif