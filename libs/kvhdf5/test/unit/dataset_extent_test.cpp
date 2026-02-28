#include <catch2/catch_test_macros.hpp>
#include <kvhdf5/hdf5_dataset.h>
#include <kvhdf5/memory_blob_store.h>
#include "../common/allocator_fixture.h"

using namespace kvhdf5;
using AllocatorFixture = test::AllocatorFixture<AllocatorImpl>;

TEST_CASE("Extent: grow 1D dataset", "[dataset_comprehensive][extent]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());
    auto file = File<InMemoryBlobStore>::Create(
        InMemoryBlobStore(fixture.allocator), Context(fixture.allocator));
    REQUIRE(file.has_value());
    auto root = file.value().OpenRootGroup();

    // Create 10-element dataset (single chunk of 10)
    uint64_t dims[] = {10};
    auto space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(space.has_value());
    auto ds = root.CreateDataset("data1d", Datatype::Int32(), space.value());
    REQUIRE(ds.has_value());

    // Write elements 0-9
    int32_t buf[10];
    for (int i = 0; i < 10; i++) buf[i] = i * 10;
    auto ms = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(ms.has_value());
    auto fs = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(fs.has_value());
    REQUIRE(ds.value().Write(Datatype::Int32(), ms.value(), fs.value(), buf).has_value());

    // SetExtent to 20
    uint64_t new_dims[] = {20};
    REQUIRE(ds.value().SetExtent(cstd::span<const uint64_t>(new_dims, 1)).has_value());

    // GetSpace should now report 20 elements
    auto sp = ds.value().GetSpace();
    REQUIRE(sp.has_value());
    REQUIRE(sp.value().GetNDims() == 1);
    REQUIRE(sp.value().GetTotalElements() == 20);

    // Read all 20 elements: first 10 preserved, last 10 zero
    int32_t out[20];
    for (int i = 0; i < 20; i++) out[i] = -1;
    uint64_t read_dims[] = {20};
    auto rms = Dataspace::CreateSimple(cstd::span<const uint64_t>(read_dims, 1));
    REQUIRE(rms.has_value());
    auto rfs = Dataspace::CreateSimple(cstd::span<const uint64_t>(read_dims, 1));
    REQUIRE(rfs.has_value());
    REQUIRE(ds.value().Read(Datatype::Int32(), rms.value(), rfs.value(), out).has_value());

    for (int i = 0; i < 10; i++) REQUIRE(out[i] == i * 10);
    for (int i = 10; i < 20; i++) REQUIRE(out[i] == 0);
}

TEST_CASE("Extent: grow 2D dataset", "[dataset_comprehensive][extent]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());
    auto file = File<InMemoryBlobStore>::Create(
        InMemoryBlobStore(fixture.allocator), Context(fixture.allocator));
    REQUIRE(file.has_value());
    auto root = file.value().OpenRootGroup();

    // Create 4x5 dataset with a single 4x5 chunk
    uint64_t dims[] = {4, 5};
    auto space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 2));
    REQUIRE(space.has_value());
    DatasetCreateProps props;
    props.chunk_dims.push_back(4);
    props.chunk_dims.push_back(5);
    auto ds = root.CreateDataset("data2d", Datatype::Int32(), space.value(), props);
    REQUIRE(ds.has_value());

    // Write all 20 values (0..19)
    float buf[20];
    for (int i = 0; i < 20; i++) buf[i] = float(i);
    auto ms = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 2));
    REQUIRE(ms.has_value());
    auto fs = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 2));
    REQUIRE(fs.has_value());
    REQUIRE(ds.value().Write(Datatype::Float32(), ms.value(), fs.value(), buf).has_value());

    // SetExtent to {8, 5}
    uint64_t new_dims[] = {8, 5};
    REQUIRE(ds.value().SetExtent(cstd::span<const uint64_t>(new_dims, 2)).has_value());

    // GetSpace should report 8x5 = 40 elements
    auto sp = ds.value().GetSpace();
    REQUIRE(sp.has_value());
    REQUIRE(sp.value().GetNDims() == 2);
    REQUIRE(sp.value().GetTotalElements() == 40);

    // Read all 40 elements: first 20 preserved (rows 0-3), last 20 zero (rows 4-7)
    float out[40];
    for (int i = 0; i < 40; i++) out[i] = -1.0f;
    uint64_t read_dims[] = {8, 5};
    auto rms = Dataspace::CreateSimple(cstd::span<const uint64_t>(read_dims, 2));
    REQUIRE(rms.has_value());
    auto rfs = Dataspace::CreateSimple(cstd::span<const uint64_t>(read_dims, 2));
    REQUIRE(rfs.has_value());
    REQUIRE(ds.value().Read(Datatype::Float32(), rms.value(), rfs.value(), out).has_value());

    // Rows 0-3 (elements 0-19) match original data
    for (int i = 0; i < 20; i++) REQUIRE(out[i] == float(i));
    // Rows 4-7 (elements 20-39) are zero — those chunks were never written
    for (int i = 20; i < 40; i++) REQUIRE(out[i] == 0.0f);
}

TEST_CASE("Extent: write to extended region", "[dataset_comprehensive][extent]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());
    auto file = File<InMemoryBlobStore>::Create(
        InMemoryBlobStore(fixture.allocator), Context(fixture.allocator));
    REQUIRE(file.has_value());
    auto root = file.value().OpenRootGroup();

    // Create 10-element 1D dataset with chunk size 10
    uint64_t dims[] = {10};
    auto space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(space.has_value());
    DatasetCreateProps props;
    props.chunk_dims.push_back(10);
    auto ds = root.CreateDataset("extwrite", Datatype::Int32(), space.value(), props);
    REQUIRE(ds.has_value());

    // Write elements 0-9
    int32_t buf_orig[10];
    for (int i = 0; i < 10; i++) buf_orig[i] = i + 1;
    auto ms0 = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(ms0.has_value());
    auto fs0 = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(fs0.has_value());
    REQUIRE(ds.value().Write(Datatype::Int32(), ms0.value(), fs0.value(), buf_orig).has_value());

    // SetExtent to 20
    uint64_t new_dims[] = {20};
    REQUIRE(ds.value().SetExtent(cstd::span<const uint64_t>(new_dims, 1)).has_value());

    // Write to elements 10-19 via hyperslab (start=10, count=10, stride=1, block=1)
    int32_t buf_ext[10];
    for (int i = 0; i < 10; i++) buf_ext[i] = (i + 10) * 100;

    uint64_t mem_dims[] = {10};
    auto ms1 = Dataspace::CreateSimple(cstd::span<const uint64_t>(mem_dims, 1));
    REQUIRE(ms1.has_value());
    auto fs1 = Dataspace::CreateSimple(cstd::span<const uint64_t>(new_dims, 1));
    REQUIRE(fs1.has_value());
    uint64_t start[] = {10}, stride[] = {1}, count[] = {10}, block[] = {1};
    REQUIRE(fs1.value().SelectHyperslab(
        SelectionOp::Set,
        cstd::span<const uint64_t>(start, 1),
        cstd::span<const uint64_t>(stride, 1),
        cstd::span<const uint64_t>(count, 1),
        cstd::span<const uint64_t>(block, 1)).has_value());
    REQUIRE(ds.value().Write(Datatype::Int32(), ms1.value(), fs1.value(), buf_ext).has_value());

    // Read all 20 and verify both regions
    int32_t out[20];
    for (int i = 0; i < 20; i++) out[i] = -1;
    auto rms = Dataspace::CreateSimple(cstd::span<const uint64_t>(new_dims, 1));
    REQUIRE(rms.has_value());
    auto rfs = Dataspace::CreateSimple(cstd::span<const uint64_t>(new_dims, 1));
    REQUIRE(rfs.has_value());
    REQUIRE(ds.value().Read(Datatype::Int32(), rms.value(), rfs.value(), out).has_value());

    // First region: original data
    for (int i = 0; i < 10; i++) REQUIRE(out[i] == i + 1);
    // Extended region: newly written data
    for (int i = 0; i < 10; i++) REQUIRE(out[10 + i] == (i + 10) * 100);
}

TEST_CASE("Extent: extended region reads as zero", "[dataset_comprehensive][extent]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());
    auto file = File<InMemoryBlobStore>::Create(
        InMemoryBlobStore(fixture.allocator), Context(fixture.allocator));
    REQUIRE(file.has_value());
    auto root = file.value().OpenRootGroup();

    // Create 5-element dataset; nothing is written
    uint64_t dims[] = {5};
    auto space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(space.has_value());
    auto ds = root.CreateDataset("unwritten", Datatype::Int32(), space.value());
    REQUIRE(ds.has_value());

    // SetExtent to 15 without writing anything
    uint64_t new_dims[] = {15};
    REQUIRE(ds.value().SetExtent(cstd::span<const uint64_t>(new_dims, 1)).has_value());

    // Read all 15 — every element should be zero
    int32_t out[15];
    for (int i = 0; i < 15; i++) out[i] = 99;
    auto rms = Dataspace::CreateSimple(cstd::span<const uint64_t>(new_dims, 1));
    REQUIRE(rms.has_value());
    auto rfs = Dataspace::CreateSimple(cstd::span<const uint64_t>(new_dims, 1));
    REQUIRE(rfs.has_value());
    REQUIRE(ds.value().Read(Datatype::Int32(), rms.value(), rfs.value(), out).has_value());

    for (int i = 0; i < 15; i++) REQUIRE(out[i] == 0);
}
