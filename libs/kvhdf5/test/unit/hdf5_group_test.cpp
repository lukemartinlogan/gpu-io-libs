#include <catch2/catch_test_macros.hpp>
#include <kvhdf5/hdf5_group.h>
#include <kvhdf5/memory_blob_store.h>
#include "../common/allocator_fixture.h"

using namespace kvhdf5;
using AllocatorFixture = test::AllocatorFixture<AllocatorImpl>;

TEST_CASE("File OpenRootGroup returns valid group", "[hdf5_group]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());

    auto file = File<InMemoryBlobStore>::Create(
        InMemoryBlobStore(fixture.allocator), Context(fixture.allocator));
    REQUIRE(file.has_value());

    auto root = file.value().OpenRootGroup();
    REQUIRE(root.GetId().IsValid());
}

TEST_CASE("CreateGroup and OpenGroup", "[hdf5_group]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());

    auto file = File<InMemoryBlobStore>::Create(
        InMemoryBlobStore(fixture.allocator), Context(fixture.allocator));
    REQUIRE(file.has_value());

    auto root = file.value().OpenRootGroup();

    auto created = root.CreateGroup("simulation");
    REQUIRE(created.has_value());

    auto opened = root.OpenGroup("simulation");
    REQUIRE(opened.has_value());

    REQUIRE(created.value().GetId() == opened.value().GetId());
}

TEST_CASE("OpenGroup nonexistent fails", "[hdf5_group]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());

    auto file = File<InMemoryBlobStore>::Create(
        InMemoryBlobStore(fixture.allocator), Context(fixture.allocator));
    REQUIRE(file.has_value());

    auto root = file.value().OpenRootGroup();

    auto result = root.OpenGroup("nonexistent");
    REQUIRE_FALSE(result.has_value());
}

TEST_CASE("CreateGroup duplicate name fails", "[hdf5_group]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());

    auto file = File<InMemoryBlobStore>::Create(
        InMemoryBlobStore(fixture.allocator), Context(fixture.allocator));
    REQUIRE(file.has_value());

    auto root = file.value().OpenRootGroup();

    auto first = root.CreateGroup("a");
    REQUIRE(first.has_value());

    auto second = root.CreateGroup("a");
    REQUIRE_FALSE(second.has_value());
}

TEST_CASE("Nested groups", "[hdf5_group]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());

    auto file = File<InMemoryBlobStore>::Create(
        InMemoryBlobStore(fixture.allocator), Context(fixture.allocator));
    REQUIRE(file.has_value());

    auto root = file.value().OpenRootGroup();

    auto a = root.CreateGroup("a");
    REQUIRE(a.has_value());

    auto b = a.value().CreateGroup("b");
    REQUIRE(b.has_value());

    auto c = b.value().CreateGroup("c");
    REQUIRE(c.has_value());

    // Walk back and verify
    auto a_opened = root.OpenGroup("a");
    REQUIRE(a_opened.has_value());
    REQUIRE(a_opened.value().GetId() == a.value().GetId());

    auto b_opened = a_opened.value().OpenGroup("b");
    REQUIRE(b_opened.has_value());
    REQUIRE(b_opened.value().GetId() == b.value().GetId());

    auto c_opened = b_opened.value().OpenGroup("c");
    REQUIRE(c_opened.has_value());
    REQUIRE(c_opened.value().GetId() == c.value().GetId());
}

TEST_CASE("Group::GetInfo", "[hdf5_group]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());

    auto file = File<InMemoryBlobStore>::Create(
        InMemoryBlobStore(fixture.allocator), Context(fixture.allocator));
    REQUIRE(file.has_value());

    auto root = file.value().OpenRootGroup();

    auto child1 = root.CreateGroup("child1");
    REQUIRE(child1.has_value());

    auto child2 = root.CreateGroup("child2");
    REQUIRE(child2.has_value());

    auto info = root.GetInfo();
    REQUIRE(info.has_value());
    REQUIRE(info.value().num_children == 2);
    REQUIRE(info.value().num_attributes == 0);
}

TEST_CASE("Group attributes set and get", "[hdf5_group]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());

    auto file = File<InMemoryBlobStore>::Create(
        InMemoryBlobStore(fixture.allocator), Context(fixture.allocator));
    REQUIRE(file.has_value());

    auto root = file.value().OpenRootGroup();

    int32_t value = 42;
    auto dt = Datatype::Int32();

    auto set_result = root.SetAttribute("count", dt, &value);
    REQUIRE(set_result.has_value());

    REQUIRE(root.HasAttribute("count"));
    REQUIRE_FALSE(root.HasAttribute("missing"));

    int32_t read_value = 0;
    auto get_result = root.GetAttribute("count", dt, &read_value);
    REQUIRE(get_result.has_value());
    REQUIRE(read_value == 42);
}

TEST_CASE("Group attribute overwrite", "[hdf5_group]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());

    auto file = File<InMemoryBlobStore>::Create(
        InMemoryBlobStore(fixture.allocator), Context(fixture.allocator));
    REQUIRE(file.has_value());

    auto root = file.value().OpenRootGroup();

    auto dt = Datatype::Float64();

    double val1 = 3.14;
    auto set1 = root.SetAttribute("pi", dt, &val1);
    REQUIRE(set1.has_value());

    double val2 = 2.718;
    auto set2 = root.SetAttribute("pi", dt, &val2);
    REQUIRE(set2.has_value());

    double read_value = 0.0;
    auto get_result = root.GetAttribute("pi", dt, &read_value);
    REQUIRE(get_result.has_value());
    REQUIRE(read_value == 2.718);
}
