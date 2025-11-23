#pragma once

// Central header for all standard library includes with CUDA/std switching
// All other headers should include this instead of duplicating the conditional logic

#include <cstddef>
#include <cstdint>
#include <limits>

#ifdef LIBCUDACXX_AVAILABLE
    #include <cuda/std/optional>
    #include <cuda/std/array>
    #include <cuda/std/variant>
    #include <cuda/std/span>
    #include <cuda/std/utility>
    #include <cuda/std/bitset>
    #include <cuda/std/tuple>
    #include <cuda/std/chrono>
    #include <cuda/std/inplace_vector>
    #include <cuda/std/expected>
    #include <cuda/std/cassert>
    #include <cuda/std/limits>
    #include <cuda/std/cstdint>

    namespace cstd = cuda::std;

    namespace hdf5 {
        constexpr uint32_t MAX_DIMS = 8;

        template<typename T>
        using dim_vector = cstd::inplace_vector<T, MAX_DIMS>;
    }
#else
    #include <optional>
    #include <array>
    #include <variant>
    #include <span>
    #include <utility>
    #include <bitset>
    #include <tuple>
    #include <chrono>
    #include <vector>  // Fallback for inplace_vector
    #include <expected>
    #include <cassert>
    #include <limits>
    #include <cstdint>

    namespace cstd = std;

    namespace hdf5 {
        template<typename T>
        using dim_vector = std::vector<T>;
    }
#endif