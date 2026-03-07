#pragma once

#include "cuda_compat.h"
#include "cte_runtime.h"
#include "kvhdf5/cte_blob_store.h"
#include "kvhdf5/container.h"
#include "kvhdf5/hdf5.h"
#include "hermes_shm/memory/backend/array_backend.h"
#include "hermes_shm/memory/allocator/buddy_allocator.h"
#include <benchmark/benchmark.h>
#include <atomic>
#include <string>
#include <cuda/std/cstring>

namespace bench {

// Global atomic counter for unique CTE tag names per benchmark iteration.
inline std::atomic<uint64_t> g_tag_counter{0};

inline std::string UniqueTagName() {
    uint64_t id = g_tag_counter.fetch_add(1, std::memory_order_relaxed);
    return "bench_tag_" + std::to_string(id);
}

// Default heap size: 1MB (enough for metadata objects; override heap_size_ if more is needed).
static constexpr size_t kDefaultHeapSize = 1ULL * 1024 * 1024;

// Base fixture that manages its own allocator with configurable heap size.
// Not a template — subclass fixtures set heap_size_ in their constructor.
class CteFixtureBase : public benchmark::Fixture {
protected:
    size_t heap_size_ = kDefaultHeapSize;
    size_t alloc_size_ = 0;
    char* memory_ = nullptr;
    hshm::ipc::ArrayBackend backend_;
    kvhdf5::AllocatorImpl* allocator_ = nullptr;

public:
    void SetUp(benchmark::State&) override {
        EnsureCteRuntime();
        alloc_size_ = heap_size_ + 3 * hshm::ipc::kBackendHeaderSize;
        memory_ = new char[alloc_size_];
        cuda::std::memset(memory_, 0, alloc_size_);
        if (backend_.shm_init(hshm::ipc::MemoryBackendId::GetRoot(),
                              alloc_size_, memory_)) {
            allocator_ = backend_.MakeAlloc<kvhdf5::AllocatorImpl>();
        }
    }

    void TearDown(benchmark::State&) override {
        allocator_ = nullptr;
        if (memory_) {
            delete[] memory_;
            memory_ = nullptr;
        }
    }

    kvhdf5::AllocatorImpl* GetAllocator() { return allocator_; }

    // Re-initialize the BuddyAllocator from scratch. Call between benchmark
    // iterations (inside state.PauseTiming()) when the benchmark creates
    // many short-lived objects.
    void ResetAllocator() {
        cuda::std::memset(memory_, 0, alloc_size_);
        backend_.shm_init(hshm::ipc::MemoryBackendId::GetRoot(),
                          alloc_size_, memory_);
        allocator_ = backend_.MakeAlloc<kvhdf5::AllocatorImpl>();
    }

    kvhdf5::Container<kvhdf5::CteBlobStore> CreateContainer() {
        auto tag = UniqueTagName();
        kvhdf5::CteBlobStore store(tag);
        return kvhdf5::Container<kvhdf5::CteBlobStore>(
            std::move(store), GetAllocator());
    }

    kvhdf5::File<kvhdf5::CteBlobStore> CreateFile() {
        auto tag = UniqueTagName();
        kvhdf5::CteBlobStore store(tag);
        kvhdf5::Context ctx(GetAllocator());
        auto result = kvhdf5::File<kvhdf5::CteBlobStore>::Create(
            std::move(store), ctx);
        assert(result.has_value());
        return std::move(result.value());
    }

    void DestroyContainer(kvhdf5::Container<kvhdf5::CteBlobStore>& c) {
        c.GetBlobStore().GetStore()->Destroy();
    }

    void DestroyFile(kvhdf5::File<kvhdf5::CteBlobStore>& f) {
        f.GetContainer().GetBlobStore().GetStore()->Destroy();
    }
};

// Convenience alias for the default fixture.
using CteFixture = CteFixtureBase;

// Lightweight fixture for allocator-only benchmarks.
// Does NOT start the CTE runtime — safe for vector/allocator microbenchmarks.
class AllocatorFixture : public benchmark::Fixture {
protected:
    size_t heap_size_ = kDefaultHeapSize;
    size_t alloc_size_ = 0;
    char* memory_ = nullptr;
    hshm::ipc::ArrayBackend backend_;
    kvhdf5::AllocatorImpl* allocator_ = nullptr;

public:
    void SetUp(benchmark::State&) override {
        alloc_size_ = heap_size_ + 3 * hshm::ipc::kBackendHeaderSize;
        memory_ = new char[alloc_size_];
        cuda::std::memset(memory_, 0, alloc_size_);
        if (backend_.shm_init(hshm::ipc::MemoryBackendId::GetRoot(),
                              alloc_size_, memory_)) {
            allocator_ = backend_.MakeAlloc<kvhdf5::AllocatorImpl>();
        }
    }

    void TearDown(benchmark::State&) override {
        allocator_ = nullptr;
        if (memory_) {
            delete[] memory_;
            memory_ = nullptr;
        }
    }

    kvhdf5::AllocatorImpl* GetAllocator() { return allocator_; }

    void ResetAllocator() {
        cuda::std::memset(memory_, 0, alloc_size_);
        backend_.shm_init(hshm::ipc::MemoryBackendId::GetRoot(),
                          alloc_size_, memory_);
        allocator_ = backend_.MakeAlloc<kvhdf5::AllocatorImpl>();
    }
};

}  // namespace bench
