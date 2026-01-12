#pragma once

#include "../cuda_compat.h"

#include "hermes_shm/memory/allocator/arena_allocator.h"
#include "hermes_shm/data_structures/priv/vector.h"

namespace hdf5 {

    using HdfAllocator = hshm::ipc::ArenaAllocator<false>;

    template<typename T>
    using vector = hshm::priv::vector<T, HdfAllocator>;

} // namespace hdf5
