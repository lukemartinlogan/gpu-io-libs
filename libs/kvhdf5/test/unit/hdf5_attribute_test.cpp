#include <catch2/catch_test_macros.hpp>
#include <kvhdf5/hdf5_attribute.h>
#include <kvhdf5/memory_blob_store.h>
#include "../common/allocator_fixture.h"

using namespace kvhdf5;
using AllocatorFixture = test::AllocatorFixture<AllocatorImpl>;

TEST_CASE("Attribute Write and Read on Group", "[hdf5_attribute]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());

    auto file = File<InMemoryBlobStore>::Create(
        InMemoryBlobStore(fixture.allocator), Context(fixture.allocator));
    REQUIRE(file.has_value());

    auto root = file.value().OpenRootGroup();
    auto dt = Datatype::Float32();

    float pi = 3.14f;
    auto set_result = root.SetAttribute("pi", dt, &pi);
    REQUIRE(set_result.has_value());

    float read_val = 0.0f;
    auto get_result = root.GetAttribute("pi", dt, &read_val);
    REQUIRE(get_result.has_value());
    REQUIRE(read_val == pi);
}

TEST_CASE("HasAttribute on Group", "[hdf5_attribute]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());

    auto file = File<InMemoryBlobStore>::Create(
        InMemoryBlobStore(fixture.allocator), Context(fixture.allocator));
    REQUIRE(file.has_value());

    auto root = file.value().OpenRootGroup();

    REQUIRE_FALSE(root.HasAttribute("nonexistent"));

    int32_t val = 7;
    auto set_result = root.SetAttribute("count", Datatype::Int32(), &val);
    REQUIRE(set_result.has_value());

    REQUIRE(root.HasAttribute("count"));
    REQUIRE_FALSE(root.HasAttribute("nonexistent"));
}

TEST_CASE("Dataset attributes", "[hdf5_attribute]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());

    auto file = File<InMemoryBlobStore>::Create(
        InMemoryBlobStore(fixture.allocator), Context(fixture.allocator));
    REQUIRE(file.has_value());

    auto root = file.value().OpenRootGroup();

    uint64_t dims[] = {4};
    auto space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(space.has_value());

    auto ds = root.CreateDataset("data", Datatype::Float64(), space.value());
    REQUIRE(ds.has_value());

    REQUIRE_FALSE(ds.value().HasAttribute("scale"));

    double scale = 1.5;
    auto set_result = ds.value().SetAttribute("scale", Datatype::Float64(), &scale);
    REQUIRE(set_result.has_value());

    REQUIRE(ds.value().HasAttribute("scale"));

    double read_val = 0.0;
    auto get_result = ds.value().GetAttribute("scale", Datatype::Float64(), &read_val);
    REQUIRE(get_result.has_value());
    REQUIRE(read_val == scale);
}

TEST_CASE("Attribute handle Write and Read", "[hdf5_attribute]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());

    auto file = File<InMemoryBlobStore>::Create(
        InMemoryBlobStore(fixture.allocator), Context(fixture.allocator));
    REQUIRE(file.has_value());

    auto root = file.value().OpenRootGroup();
    auto dt = Datatype::Int64();

    // Create the attribute via SetAttribute first so it exists
    int64_t initial = 42;
    auto set_result = root.SetAttribute("answer", dt, &initial);
    REQUIRE(set_result.has_value());

    // Open handle and read through it
    auto handle = root.OpenAttribute("answer");
    REQUIRE(handle.GetName() == gpu_string_view("answer"));

    int64_t read_val = 0;
    auto read_result = handle.Read(dt, &read_val);
    REQUIRE(read_result.has_value());
    REQUIRE(read_val == 42);

    // Write a new value through the handle
    int64_t updated = 99;
    auto write_result = handle.Write(dt, &updated);
    REQUIRE(write_result.has_value());

    // Verify via group's GetAttribute
    int64_t verify = 0;
    auto get_result = root.GetAttribute("answer", dt, &verify);
    REQUIRE(get_result.has_value());
    REQUIRE(verify == 99);
}

TEST_CASE("Attribute handle on Dataset", "[hdf5_attribute]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());

    auto file = File<InMemoryBlobStore>::Create(
        InMemoryBlobStore(fixture.allocator), Context(fixture.allocator));
    REQUIRE(file.has_value());

    auto root = file.value().OpenRootGroup();

    uint64_t dims[] = {8};
    auto space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(space.has_value());

    auto ds = root.CreateDataset("vec", Datatype::Uint32(), space.value());
    REQUIRE(ds.has_value());

    uint32_t tag = 0xDEADBEEF;
    auto set_result = ds.value().SetAttribute("tag", Datatype::Uint32(), &tag);
    REQUIRE(set_result.has_value());

    auto handle = ds.value().OpenAttribute("tag");
    REQUIRE(handle.GetName() == gpu_string_view("tag"));

    uint32_t read_val = 0;
    auto read_result = handle.Read(Datatype::Uint32(), &read_val);
    REQUIRE(read_result.has_value());
    REQUIRE(read_val == 0xDEADBEEF);
}
