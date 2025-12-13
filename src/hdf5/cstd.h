#pragma once

// Central header for all standard library includes with CUDA/std switching
// All other headers should include this instead of duplicating the conditional logic


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
    #include <cuda/std/cstddef>
    #include <cuda/std/type_traits>
    #include <cuda/std/algorithm>
    #include <cuda/std/atomic>

    // Workaround for MSVC + cuda::std::copy compatibility issue
    namespace cuda::std {
        template<typename InputIt, typename OutputIt>
        constexpr OutputIt _copy(InputIt first, InputIt last, OutputIt d_first) {
            while (first != last) {
                *d_first++ = *first++;
            }
            return d_first;
        }
    }

    namespace cstd = cuda::std;

    namespace hdf5 {
        constexpr uint32_t MAX_DIMS = 8;

        template<typename T>
        using dim_vector = cstd::inplace_vector<T, MAX_DIMS>;
    }
#else
    #include <optional>
    #include <array>
    #include <algorithm>
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
    #include <cstddef>
    #include <type_traits>
    #include <atomic>

    // Alias for non-CUDA builds
    namespace std {
        template<typename InputIt, typename OutputIt>
        constexpr OutputIt _copy(InputIt first, InputIt last, OutputIt d_first) {
            return copy(first, last, d_first);
        }
    }

    namespace cstd = std;

    namespace hdf5 {
        template<typename T>
        using dim_vector = std::vector<T>;
    }
#endif