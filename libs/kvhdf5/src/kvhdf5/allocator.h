#pragma once

#include "defines.h"
#include "hermes_shm/memory/allocator/arena_allocator.h"
#include <cuda/std/concepts>

namespace kvhdf5 {

using AllocatorImpl = hshm::ipc::ArenaAllocator<false>;

template<typename A>
concept Allocator = requires {
    // for now, just check it's our allocator type
    requires cstd::same_as<A, AllocatorImpl>;
};

template<typename T, typename A = AllocatorImpl>
concept ProvidesAllocator = requires(T&& t) {
    { t.GetAllocator() } -> cstd::same_as<A&>;
} && Allocator<A>;

} // namespace kvhdf5
