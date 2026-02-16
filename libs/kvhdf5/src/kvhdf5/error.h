#pragma once

#include "defines.h"
#include "utils/gpu_string.h"
#include <cuda/std/expected>

namespace kvhdf5 {

enum class ErrorCode : uint8_t {
    InvalidArgument,
    // Add more error codes as needed during implementation
};

struct Error {
    ErrorCode code;
    gpu_string_view message;
};

template<typename T>
using expected = cstd::expected<T, Error>;

CROSS_FUN
constexpr cstd::unexpected<Error> make_error(ErrorCode code, gpu_string_view msg = {}) {
    return cstd::unexpected(Error{code, msg});
}

} // namespace kvhdf5
