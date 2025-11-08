#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>

using byte_t = std::byte;

using offset_t = uint64_t;
using len_t = uint64_t;

constexpr offset_t kUndefinedOffset = std::numeric_limits<offset_t>::max();

namespace hdf5 {
    constexpr uint32_t MAX_DIMS = 8;
}

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

    #define ASSERT(cond, msg) assert((cond) && (msg))
    #define UNREACHABLE(msg) assert(false && (msg))

    namespace cstd = cuda::std;

    namespace hdf5 {
        template<typename T>
        using dim_vector = cstd::inplace_vector<T, MAX_DIMS>;
    }
#endif

// Include error types (requires cstd namespace to be defined)
#include "error.h"

#include "gpu_string.h"

namespace hdf5 {
    // Type aliases for string types
    using string_view = gpu_string_view;
    using string = gpu_string<>;  // Default 255 char max
}