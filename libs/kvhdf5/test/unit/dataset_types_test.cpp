#include <catch2/catch_test_macros.hpp>
#include <kvhdf5/hdf5_dataset.h>
#include <kvhdf5/memory_blob_store.h>
#include "../common/allocator_fixture.h"

using namespace kvhdf5;
using AllocatorFixture = test::AllocatorFixture<AllocatorImpl>;

TEST_CASE("Types: Int8 dataset", "[dataset_comprehensive][types]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());
    auto file = File<InMemoryBlobStore>::Create(
        InMemoryBlobStore(fixture.allocator), Context(fixture.allocator));
    REQUIRE(file.has_value());
    auto root = file.value().OpenRootGroup();

    uint64_t dims[] = {10};
    auto space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(space.has_value());
    auto type = Datatype::Int8();

    auto ds = root.CreateDataset("int8_data", type, space.value());
    REQUIRE(ds.has_value());

    int8_t buf[10];
    for (int i = 0; i < 10; i++) buf[i] = static_cast<int8_t>(-5 + i);

    auto ms = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(ms.has_value());
    auto fs = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(fs.has_value());

    REQUIRE(ds.value().Write(type, ms.value(), fs.value(), buf).has_value());

    int8_t out[10] = {};
    REQUIRE(ds.value().Read(type, ms.value(), fs.value(), out).has_value());
    for (int i = 0; i < 10; i++) REQUIRE(out[i] == static_cast<int8_t>(-5 + i));
}

TEST_CASE("Types: Int64 dataset", "[dataset_comprehensive][types]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());
    auto file = File<InMemoryBlobStore>::Create(
        InMemoryBlobStore(fixture.allocator), Context(fixture.allocator));
    REQUIRE(file.has_value());
    auto root = file.value().OpenRootGroup();

    uint64_t dims[] = {10};
    auto space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(space.has_value());
    auto type = Datatype::Int64();

    auto ds = root.CreateDataset("int64_data", type, space.value());
    REQUIRE(ds.has_value());

    int64_t buf[10];
    for (int i = 0; i < 10; i++) buf[i] = 1LL << (40 + i);

    auto ms = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(ms.has_value());
    auto fs = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(fs.has_value());

    REQUIRE(ds.value().Write(type, ms.value(), fs.value(), buf).has_value());

    int64_t out[10] = {};
    REQUIRE(ds.value().Read(type, ms.value(), fs.value(), out).has_value());
    for (int i = 0; i < 10; i++) REQUIRE(out[i] == 1LL << (40 + i));
}

TEST_CASE("Types: Float32 dataset", "[dataset_comprehensive][types]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());
    auto file = File<InMemoryBlobStore>::Create(
        InMemoryBlobStore(fixture.allocator), Context(fixture.allocator));
    REQUIRE(file.has_value());
    auto root = file.value().OpenRootGroup();

    uint64_t dims[] = {10};
    auto space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(space.has_value());
    auto type = Datatype::Float32();

    auto ds = root.CreateDataset("float32_data", type, space.value());
    REQUIRE(ds.has_value());

    float buf[10];
    for (int i = 0; i < 10; i++) buf[i] = 0.5f + static_cast<float>(i);

    auto ms = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(ms.has_value());
    auto fs = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(fs.has_value());

    REQUIRE(ds.value().Write(type, ms.value(), fs.value(), buf).has_value());

    float out[10] = {};
    REQUIRE(ds.value().Read(type, ms.value(), fs.value(), out).has_value());
    for (int i = 0; i < 10; i++) REQUIRE(out[i] == 0.5f + static_cast<float>(i));
}

TEST_CASE("Types: Float64 dataset", "[dataset_comprehensive][types]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());
    auto file = File<InMemoryBlobStore>::Create(
        InMemoryBlobStore(fixture.allocator), Context(fixture.allocator));
    REQUIRE(file.has_value());
    auto root = file.value().OpenRootGroup();

    uint64_t dims[] = {10};
    auto space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(space.has_value());
    auto type = Datatype::Float64();

    auto ds = root.CreateDataset("float64_data", type, space.value());
    REQUIRE(ds.has_value());

    double buf[10];
    for (int i = 0; i < 10; i++) buf[i] = static_cast<double>(i + 1) * 1e100;

    auto ms = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(ms.has_value());
    auto fs = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(fs.has_value());

    REQUIRE(ds.value().Write(type, ms.value(), fs.value(), buf).has_value());

    double out[10] = {};
    REQUIRE(ds.value().Read(type, ms.value(), fs.value(), out).has_value());
    for (int i = 0; i < 10; i++) REQUIRE(out[i] == static_cast<double>(i + 1) * 1e100);
}

TEST_CASE("MemSpace: read into subregion of larger buffer", "[dataset_comprehensive][memspace]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());
    auto file = File<InMemoryBlobStore>::Create(
        InMemoryBlobStore(fixture.allocator), Context(fixture.allocator));
    REQUIRE(file.has_value());
    auto root = file.value().OpenRootGroup();

    // Create a 10-element dataset and write values 0-9
    uint64_t ds_dims[] = {10};
    auto space = Dataspace::CreateSimple(cstd::span<const uint64_t>(ds_dims, 1));
    REQUIRE(space.has_value());
    auto type = Datatype::Int32();

    auto ds = root.CreateDataset("subregion_read", type, space.value());
    REQUIRE(ds.has_value());

    int32_t write_buf[10];
    for (int i = 0; i < 10; i++) write_buf[i] = i;

    auto write_ms = Dataspace::CreateSimple(cstd::span<const uint64_t>(ds_dims, 1));
    REQUIRE(write_ms.has_value());
    auto write_fs = Dataspace::CreateSimple(cstd::span<const uint64_t>(ds_dims, 1));
    REQUIRE(write_fs.has_value());
    REQUIRE(ds.value().Write(type, write_ms.value(), write_fs.value(), write_buf).has_value());

    // Allocate 20-element output buffer initialized to -1
    int32_t out[20];
    for (int i = 0; i < 20; i++) out[i] = -1;

    // mem_space: {20} with hyperslab start=5, count=10, stride=1, block=1
    uint64_t mem_dims[] = {20};
    auto mem_space = Dataspace::CreateSimple(cstd::span<const uint64_t>(mem_dims, 1));
    REQUIRE(mem_space.has_value());
    uint64_t mem_start[] = {5}, mem_stride[] = {1}, mem_count[] = {10}, mem_block[] = {1};
    REQUIRE(mem_space.value().SelectHyperslab(SelectionOp::Set,
        cstd::span<const uint64_t>(mem_start, 1),
        cstd::span<const uint64_t>(mem_stride, 1),
        cstd::span<const uint64_t>(mem_count, 1),
        cstd::span<const uint64_t>(mem_block, 1)).has_value());

    // file_space: {10} SelectAll
    auto file_space = Dataspace::CreateSimple(cstd::span<const uint64_t>(ds_dims, 1));
    REQUIRE(file_space.has_value());

    REQUIRE(ds.value().Read(type, mem_space.value(), file_space.value(), out).has_value());

    // Positions 0-4 must still be -1
    for (int i = 0; i < 5; i++) REQUIRE(out[i] == -1);
    // Positions 5-14 must have values 0-9
    for (int i = 0; i < 10; i++) REQUIRE(out[5 + i] == i);
    // Positions 15-19 must still be -1
    for (int i = 15; i < 20; i++) REQUIRE(out[i] == -1);
}

TEST_CASE("MemSpace: write from subregion of larger buffer", "[dataset_comprehensive][memspace]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());
    auto file = File<InMemoryBlobStore>::Create(
        InMemoryBlobStore(fixture.allocator), Context(fixture.allocator));
    REQUIRE(file.has_value());
    auto root = file.value().OpenRootGroup();

    // Allocate 20-element source buffer; positions 5-14 hold values 100-109
    int32_t src[20] = {};
    for (int i = 0; i < 10; i++) src[5 + i] = 100 + i;

    // Create a 10-element dataset
    uint64_t ds_dims[] = {10};
    auto space = Dataspace::CreateSimple(cstd::span<const uint64_t>(ds_dims, 1));
    REQUIRE(space.has_value());
    auto type = Datatype::Int32();

    auto ds = root.CreateDataset("subregion_write", type, space.value());
    REQUIRE(ds.has_value());

    // mem_space: {20} with hyperslab start=5, count=10, stride=1, block=1
    uint64_t mem_dims[] = {20};
    auto mem_space = Dataspace::CreateSimple(cstd::span<const uint64_t>(mem_dims, 1));
    REQUIRE(mem_space.has_value());
    uint64_t mem_start[] = {5}, mem_stride[] = {1}, mem_count[] = {10}, mem_block[] = {1};
    REQUIRE(mem_space.value().SelectHyperslab(SelectionOp::Set,
        cstd::span<const uint64_t>(mem_start, 1),
        cstd::span<const uint64_t>(mem_stride, 1),
        cstd::span<const uint64_t>(mem_count, 1),
        cstd::span<const uint64_t>(mem_block, 1)).has_value());

    // file_space: {10} SelectAll
    auto file_space = Dataspace::CreateSimple(cstd::span<const uint64_t>(ds_dims, 1));
    REQUIRE(file_space.has_value());

    REQUIRE(ds.value().Write(type, mem_space.value(), file_space.value(), src).has_value());

    // Read back all 10 elements and verify values 100-109
    int32_t out[10] = {};
    auto read_ms = Dataspace::CreateSimple(cstd::span<const uint64_t>(ds_dims, 1));
    REQUIRE(read_ms.has_value());
    auto read_fs = Dataspace::CreateSimple(cstd::span<const uint64_t>(ds_dims, 1));
    REQUIRE(read_fs.has_value());
    REQUIRE(ds.value().Read(type, read_ms.value(), read_fs.value(), out).has_value());
    for (int i = 0; i < 10; i++) REQUIRE(out[i] == 100 + i);
}
