#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>

using byte_t = std::byte;

using offset_t = uint64_t;
using len_t = uint64_t;

constexpr offset_t kUndefinedOffset = std::numeric_limits<offset_t>::max();

#ifdef LIBCUDACXX_AVAILABLE
    #include <cuda/std/optional>
    #include <cuda/std/array>
    #include <cuda/std/variant>
    #include <cuda/std/span>
    #include <cuda/std/utility>
    #include <cuda/std/bitset>

    namespace cstd = cuda::std;
#endif