#include <catch2/catch_test_macros.hpp>
#include <kvhdf5/context.h>
#include "../common/allocator_fixture.h"

using namespace kvhdf5;
using AllocatorFixture = test::AllocatorFixture<AllocatorImpl>;

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
