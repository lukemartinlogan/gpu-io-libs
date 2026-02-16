#include <catch2/catch_test_macros.hpp>
#include <kvhdf5/context.h>
#include "hermes_shm/memory/backend/array_backend.h"
#include <cuda/std/cstring>

using namespace kvhdf5;

struct AllocatorFixture {
    static constexpr size_t kHeapSize = 64 * 1024;  // 64KB
    char* memory = nullptr;
    hshm::ipc::ArrayBackend backend;
    AllocatorImpl* allocator = nullptr;

    AllocatorFixture() {
        size_t alloc_size = kHeapSize + 3 * hshm::ipc::kBackendHeaderSize;
        memory = new char[alloc_size];
        cstd::memset(memory, 0, alloc_size);

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

TEST_CASE("Context construction", "[context]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());

    SECTION("Construction with valid allocator") {
        Context ctx(fixture.allocator);
        REQUIRE(&ctx.GetAllocator() == fixture.allocator);
    }
}

TEST_CASE("Context GetAllocator", "[context]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());

    Context ctx(fixture.allocator);

    SECTION("Returns reference to the same allocator") {
        AllocatorImpl& alloc_ref = ctx.GetAllocator();
        REQUIRE(&alloc_ref == fixture.allocator);
    }

    SECTION("Reference can be used to take address") {
        AllocatorImpl* alloc_ptr = &ctx.GetAllocator();
        REQUIRE(alloc_ptr == fixture.allocator);
    }
}

TEST_CASE("Context satisfies ProvidesAllocator concept", "[context]") {
    // Compile-time check - if this compiles, the test passes
    static_assert(ProvidesAllocator<Context>);
    static_assert(ProvidesAllocator<Context, AllocatorImpl>);

    REQUIRE(true);
}
