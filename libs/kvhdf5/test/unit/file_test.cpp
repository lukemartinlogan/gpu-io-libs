#include <catch2/catch_test_macros.hpp>
#include <kvhdf5/file.h>
#include <kvhdf5/memory_blob_store.h>
#include "../common/allocator_fixture.h"

using namespace kvhdf5;
using AllocatorFixture = test::AllocatorFixture<AllocatorImpl>;

TEST_CASE("File::Create succeeds", "[file]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());

    auto file = File<InMemoryBlobStore>::Create(
        InMemoryBlobStore(fixture.allocator), Context(fixture.allocator));
    REQUIRE(file.has_value());
}

TEST_CASE("File::GetContainer provides access", "[file]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());

    auto file = File<InMemoryBlobStore>::Create(
        InMemoryBlobStore(fixture.allocator), Context(fixture.allocator));
    REQUIRE(file.has_value());

    auto& container = file.value().GetContainer();
    REQUIRE(container.RootGroup().IsValid());
}

TEST_CASE("File move construction", "[file]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());

    auto file1 = File<InMemoryBlobStore>::Create(
        InMemoryBlobStore(fixture.allocator), Context(fixture.allocator));
    REQUIRE(file1.has_value());

    auto root_id = file1.value().GetContainer().RootGroup();

    File<InMemoryBlobStore> file2(std::move(file1.value()));
    REQUIRE(file2.GetContainer().RootGroup() == root_id);
}
