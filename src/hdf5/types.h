#pragma once

#include "cstd.h"

using byte_t = cstd::byte;
using offset_t = uint64_t;
using len_t = uint64_t;
using ssize_t = ptrdiff_t;

constexpr offset_t kUndefinedOffset = cstd::numeric_limits<offset_t>::max();

#define ASSERT(cond, msg) assert((cond) && (msg))
#define UNREACHABLE(msg) assert(false && (msg))

#include "error.h"
#include "gpu_string.h"

namespace hdf5 {
    // Type aliases for string types
    using string_view = gpu_string_view;
    using string = gpu_string<>;  // Default 255 char max
}