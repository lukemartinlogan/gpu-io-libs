#pragma once

#include "cstd.h"

using byte_t = cstd::byte;
using offset_t = uint64_t;
using len_t = uint64_t;
using ssize_t = ptrdiff_t;

// TODO: windows defines a macro called max :(
constexpr offset_t kUndefinedOffset = static_cast<offset_t>(-1);

#define ASSERT(cond, msg) assert((cond) && (msg))
#define UNREACHABLE(msg) assert(false && (msg))

#include "error.h"
#include "gpu_string.h"

namespace hdf5 {
    // Type aliases for string types
    using string_view = gpu_string_view;
    using string = gpu_string<>;  // Default 255 char max

    // Explicit padding type to work around NVCC ICE with implicit struct padding
    // when using cuda::std::expected with types containing hshm::priv::vector
    template<size_t N>
    struct padding {
        uint8_t _[N];
    };
}