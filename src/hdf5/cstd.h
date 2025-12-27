#pragma once

// Central header for all standard library includes using cuda::std
// This code only runs on GPU, so we always use cuda::std

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
    __device__
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