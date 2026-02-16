#pragma once

#include "../defines.h"
#include <cuda/std/span>

namespace algorithms {

// Manual copy implementation (cuda::std::copy has issues in some contexts)
template<typename T>
CROSS_FUN void copy(cstd::span<const T> src, cstd::span<T> dst) {
    // Assert that destination has enough space
    KVHDF5_ASSERT(dst.size() >= src.size(), "Destination span too small for copy");

    for (size_t i = 0; i < src.size(); ++i) {
        dst[i] = src[i];
    }
}

// Overload for non-const source (delegates to const version)
template<typename T>
CROSS_FUN void copy(cstd::span<T> src, cstd::span<T> dst) {
    copy(cstd::span<const T>(src.data(), src.size()), dst);
}

} // namespace algorithms
