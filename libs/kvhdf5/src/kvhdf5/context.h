#pragma once

#include "defines.h"
#include "allocator.h"
#include "hermes_shm/data_structures/priv/vector.h"

namespace kvhdf5 {

template<typename T>
using vector = hshm::priv::vector<T, AllocatorImpl>;

struct Context {
    AllocatorImpl* allocator_;

    // TODO: maybe a way to get shared references to this?

    CROSS_FUN explicit constexpr Context(AllocatorImpl* alloc) : allocator_(alloc) {
        KVHDF5_ASSERT(allocator_ != nullptr, "Context allocator is null");
    }

    CROSS_FUN AllocatorImpl& GetAllocator() const {
        KVHDF5_ASSERT(allocator_ != nullptr, "Context allocator is null");
        return *allocator_;
    }
};

static_assert(ProvidesAllocator<Context>);

} // namespace kvhdf5
