#include <catch2/catch_test_macros.hpp>
#include <kvhdf5/hdf5_dataset.h>
#include <kvhdf5/memory_blob_store.h>
#include "../common/allocator_fixture.h"

using namespace kvhdf5;
using AllocatorFixture = test::AllocatorFixture<AllocatorImpl>;

TEST_CASE("Single-chunk 1D: write all, read all", "[dataset_comprehensive][single_chunk]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());
    auto file = File<InMemoryBlobStore>::Create(
        InMemoryBlobStore(fixture.allocator), Context(fixture.allocator));
    REQUIRE(file.has_value());
    auto root = file.value().OpenRootGroup();

    // 20-element 1D dataset; no DatasetCreateProps => single chunk matching dims
    uint64_t dims[] = {20};
    auto space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(space.has_value());

    auto ds = root.CreateDataset("data1d", Datatype::Int32(), space.value());
    REQUIRE(ds.has_value());

    // Fill write buffer with sequential values 0..19
    int32_t write_buf[20];
    for (int i = 0; i < 20; i++) write_buf[i] = i;

    auto ms = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(ms.has_value());
    auto fs = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(fs.has_value());

    REQUIRE(ds.value().Write(Datatype::Int32(), ms.value(), fs.value(), write_buf).has_value());

    // Read back all 20 elements
    int32_t read_buf[20] = {};
    REQUIRE(ds.value().Read(Datatype::Int32(), ms.value(), fs.value(), read_buf).has_value());

    for (int i = 0; i < 20; i++) {
        INFO("index " << i);
        REQUIRE(read_buf[i] == i);
    }
}

TEST_CASE("Single-chunk 2D: write all, read all", "[dataset_comprehensive][single_chunk]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());
    auto file = File<InMemoryBlobStore>::Create(
        InMemoryBlobStore(fixture.allocator), Context(fixture.allocator));
    REQUIRE(file.has_value());
    auto root = file.value().OpenRootGroup();

    // 6x8 matrix; no DatasetCreateProps => single chunk covering all 48 elements
    uint64_t dims[] = {6, 8};
    auto space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 2));
    REQUIRE(space.has_value());

    auto ds = root.CreateDataset("matrix2d", Datatype::Int32(), space.value());
    REQUIRE(ds.has_value());

    // Fill write buffer: element at (row, col) = row*8 + col
    int32_t write_buf[48];
    for (int row = 0; row < 6; row++) {
        for (int col = 0; col < 8; col++) {
            write_buf[row * 8 + col] = row * 8 + col;
        }
    }

    auto ms = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 2));
    REQUIRE(ms.has_value());
    auto fs = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 2));
    REQUIRE(fs.has_value());

    REQUIRE(ds.value().Write(Datatype::Int32(), ms.value(), fs.value(), write_buf).has_value());

    // Read back entire matrix
    int32_t read_buf[48] = {};
    REQUIRE(ds.value().Read(Datatype::Int32(), ms.value(), fs.value(), read_buf).has_value());

    for (int row = 0; row < 6; row++) {
        for (int col = 0; col < 8; col++) {
            INFO("row=" << row << " col=" << col);
            REQUIRE(read_buf[row * 8 + col] == row * 8 + col);
        }
    }
}

TEST_CASE("Single-chunk: partial write via hyperslab, unwritten is zero",
          "[dataset_comprehensive][single_chunk]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());
    auto file = File<InMemoryBlobStore>::Create(
        InMemoryBlobStore(fixture.allocator), Context(fixture.allocator));
    REQUIRE(file.has_value());
    auto root = file.value().OpenRootGroup();

    // 10-element 1D dataset
    uint64_t dims[] = {10};
    auto space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(space.has_value());

    auto ds = root.CreateDataset("partial", Datatype::Int32(), space.value());
    REQUIRE(ds.has_value());

    // Write only elements 3-6 (indices 3, 4, 5, 6) via hyperslab:
    //   start=3, stride=1, count=4, block=1
    int32_t write_buf[] = {300, 400, 500, 600};
    uint64_t mem_dims[] = {4};
    auto ms = Dataspace::CreateSimple(cstd::span<const uint64_t>(mem_dims, 1));
    REQUIRE(ms.has_value());
    auto fs = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(fs.has_value());

    uint64_t start[]  = {3};
    uint64_t stride[] = {1};
    uint64_t count[]  = {4};
    uint64_t block[]  = {1};
    REQUIRE(fs.value().SelectHyperslab(
        SelectionOp::Set,
        cstd::span<const uint64_t>(start,  1),
        cstd::span<const uint64_t>(stride, 1),
        cstd::span<const uint64_t>(count,  1),
        cstd::span<const uint64_t>(block,  1)
    ).has_value());

    REQUIRE(ds.value().Write(Datatype::Int32(), ms.value(), fs.value(), write_buf).has_value());

    // Read back all 10 elements using SelectAll
    int32_t read_buf[10] = {};
    auto rms = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(rms.has_value());
    auto rfs = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(rfs.has_value());
    REQUIRE(ds.value().Read(Datatype::Int32(), rms.value(), rfs.value(), read_buf).has_value());

    // Elements 0-2 were never written: must be zero
    for (int i = 0; i < 3; i++) {
        INFO("index " << i << " should be zero");
        REQUIRE(read_buf[i] == 0);
    }

    // Elements 3-6 were written
    REQUIRE(read_buf[3] == 300);
    REQUIRE(read_buf[4] == 400);
    REQUIRE(read_buf[5] == 500);
    REQUIRE(read_buf[6] == 600);

    // Elements 7-9 were never written: must be zero
    for (int i = 7; i < 10; i++) {
        INFO("index " << i << " should be zero");
        REQUIRE(read_buf[i] == 0);
    }
}

TEST_CASE("Single-chunk: overwrite region", "[dataset_comprehensive][single_chunk]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());
    auto file = File<InMemoryBlobStore>::Create(
        InMemoryBlobStore(fixture.allocator), Context(fixture.allocator));
    REQUIRE(file.has_value());
    auto root = file.value().OpenRootGroup();

    // 10-element 1D dataset
    uint64_t dims[] = {10};
    auto space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(space.has_value());

    auto ds = root.CreateDataset("overwrite", Datatype::Int32(), space.value());
    REQUIRE(ds.has_value());

    // First: write all 10 elements with values 0..9 (scaled by 10)
    int32_t initial_buf[10];
    for (int i = 0; i < 10; i++) initial_buf[i] = i * 10;

    auto ms_all = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(ms_all.has_value());
    auto fs_all = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(fs_all.has_value());
    REQUIRE(ds.value().Write(Datatype::Int32(), ms_all.value(), fs_all.value(), initial_buf).has_value());

    // Second: overwrite elements 2-4 (indices 2, 3, 4) via hyperslab with new values
    int32_t overwrite_buf[] = {-2, -3, -4};
    uint64_t mem_dims[]  = {3};
    auto ms_hyp = Dataspace::CreateSimple(cstd::span<const uint64_t>(mem_dims, 1));
    REQUIRE(ms_hyp.has_value());
    auto fs_hyp = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(fs_hyp.has_value());

    uint64_t start[]  = {2};
    uint64_t stride[] = {1};
    uint64_t count[]  = {3};
    uint64_t block[]  = {1};
    REQUIRE(fs_hyp.value().SelectHyperslab(
        SelectionOp::Set,
        cstd::span<const uint64_t>(start,  1),
        cstd::span<const uint64_t>(stride, 1),
        cstd::span<const uint64_t>(count,  1),
        cstd::span<const uint64_t>(block,  1)
    ).has_value());

    REQUIRE(ds.value().Write(Datatype::Int32(), ms_hyp.value(), fs_hyp.value(), overwrite_buf).has_value());

    // Read back all 10 elements
    int32_t read_buf[10] = {};
    auto rms = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(rms.has_value());
    auto rfs = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(rfs.has_value());
    REQUIRE(ds.value().Read(Datatype::Int32(), rms.value(), rfs.value(), read_buf).has_value());

    // Elements 0-1: original values preserved (0, 10)
    REQUIRE(read_buf[0] == 0);
    REQUIRE(read_buf[1] == 10);

    // Elements 2-4: overwritten with new values
    REQUIRE(read_buf[2] == -2);
    REQUIRE(read_buf[3] == -3);
    REQUIRE(read_buf[4] == -4);

    // Elements 5-9: original values preserved (50, 60, 70, 80, 90)
    for (int i = 5; i < 10; i++) {
        INFO("index " << i << " should retain original value " << i * 10);
        REQUIRE(read_buf[i] == i * 10);
    }
}
