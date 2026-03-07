#pragma once

#include "../defines.h"
#include <cuda/std/span>
#include <cuda/std/cstring>

namespace algorithms {

// Copy between spans using memcpy (safe for type-punning, avoids TBAA issues)
template<typename T>
CROSS_FUN void copy(cstd::span<const T> src, cstd::span<T> dst) {
    // Assert that destination has enough space
    KVHDF5_ASSERT(dst.size() >= src.size(), "Destination span too small for copy");

    cstd::memcpy(dst.data(), src.data(), src.size() * sizeof(T));
}

// Overload for non-const source (delegates to const version)
template<typename T>
CROSS_FUN void copy(cstd::span<T> src, cstd::span<T> dst) {
    copy(cstd::span<const T>(src.data(), src.size()), dst);
}

} // namespace algorithms
