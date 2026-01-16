#pragma once

#include "../cuda_compat.h"
#include "types.h"

#include "hermes_shm/memory/allocator/buddy_allocator.h"
#include "hermes_shm/data_structures/priv/vector.h"

namespace hdf5 {

    using HdfAllocator = hshm::ipc::BuddyAllocator;

    template<typename T>
    using vector = hshm::priv::vector<T, HdfAllocator>;

} // namespace hdf5

namespace iowarp {

    template<typename T>
    concept ProvidesAllocator = requires(T&& t) {
        { t.GetAllocator() } -> std::same_as<hdf5::HdfAllocator*>;
    };

} // namespace iowarp
