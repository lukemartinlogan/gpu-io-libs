#pragma once

#include "cuda_compat.h"  // Must be included before CUDA headers for __nanosleep
#include "hermes_shm/memory/backend/array_backend.h"
#include "hermes_shm/memory/allocator/arena_allocator.h"
#include <cuda/std/cstring>

namespace test {

template<typename AllocatorImpl>
struct AllocatorFixture {
    static constexpr size_t kHeapSize = 64 * 1024;  // 64KB
    char* memory = nullptr;
    hshm::ipc::ArrayBackend backend;
    AllocatorImpl* allocator = nullptr;

    AllocatorFixture() {
        size_t alloc_size = kHeapSize + 3 * hshm::ipc::kBackendHeaderSize;
        memory = new char[alloc_size];
        cuda::std::memset(memory, 0, alloc_size);

        if (backend.shm_init(hshm::ipc::MemoryBackendId::GetRoot(), alloc_size, memory)) {
            allocator = backend.MakeAlloc<AllocatorImpl>();
        }
    }

    ~AllocatorFixture() {
        if (memory) {
            delete[] memory;
        }
    }

    bool IsValid() const { return allocator != nullptr; }
};

}  // namespace test
