#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>

using byte_t = std::byte;

using offset_t = uint64_t;
using len_t = uint64_t;

constexpr offset_t kUndefinedOffset = std::numeric_limits<offset_t>::max();