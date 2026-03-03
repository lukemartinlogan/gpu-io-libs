#include <catch2/catch_test_macros.hpp>
#include <kvhdf5/hdf5_dataset.h>
#include <kvhdf5/memory_blob_store.h>
#include "../common/allocator_fixture.h"

using namespace kvhdf5;
using AllocatorFixture = test::AllocatorFixture<AllocatorImpl>;

TEST_CASE("Hyperslab: row slice from 2D multi-chunk", "[dataset_comprehensive][hyperslab]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());
    auto file = File<InMemoryBlobStore>::Create(
        InMemoryBlobStore(fixture.allocator), Context(fixture.allocator));
    REQUIRE(file.has_value());
    auto root = file.value().OpenRootGroup();

    // 10x10 dataset with 4x4 chunks => 3x3 chunk grid
    uint64_t dims[] = {10, 10};
    auto space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 2));
    REQUIRE(space.has_value());

    DatasetCreateProps props;
    props.chunk_dims.push_back(4);
    props.chunk_dims.push_back(4);
    auto ds = root.CreateDataset("row_slice", Datatype::Int32(), space.value(), props);
    REQUIRE(ds.has_value());

    // Write all elements: value = row * 10 + col
    int32_t buf[100];
    for (int row = 0; row < 10; row++)
        for (int col = 0; col < 10; col++)
            buf[row * 10 + col] = row * 10 + col;

    auto all_space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 2));
    REQUIRE(all_space.has_value());
    REQUIRE(ds.value().Write(Datatype::Int32(), all_space.value(), all_space.value(), buf).has_value());

    // Select row 3: start={3,0}, stride={1,1}, count={1,10}, block={1,1}
    uint64_t read_dims[] = {10};
    auto ms = Dataspace::CreateSimple(cstd::span<const uint64_t>(read_dims, 1));
    REQUIRE(ms.has_value());
    auto fs = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 2));
    REQUIRE(fs.has_value());

    uint64_t start[] = {3, 0}, stride[] = {1, 1}, count[] = {1, 10}, block[] = {1, 1};
    REQUIRE(fs.value().SelectHyperslab(SelectionOp::Set,
        cstd::span<const uint64_t>(start, 2), cstd::span<const uint64_t>(stride, 2),
        cstd::span<const uint64_t>(count, 2), cstd::span<const uint64_t>(block, 2)).has_value());

    int32_t out[10] = {};
    REQUIRE(ds.value().Read(Datatype::Int32(), ms.value(), fs.value(), out).has_value());

    // Row 3: elements 30, 31, 32, ..., 39
    for (int col = 0; col < 10; col++) {
        REQUIRE(out[col] == 30 + col);
    }
}

TEST_CASE("Hyperslab: column slice from 2D multi-chunk", "[dataset_comprehensive][hyperslab]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());
    auto file = File<InMemoryBlobStore>::Create(
        InMemoryBlobStore(fixture.allocator), Context(fixture.allocator));
    REQUIRE(file.has_value());
    auto root = file.value().OpenRootGroup();

    // 10x10 dataset with 4x4 chunks
    uint64_t dims[] = {10, 10};
    auto space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 2));
    REQUIRE(space.has_value());

    DatasetCreateProps props;
    props.chunk_dims.push_back(4);
    props.chunk_dims.push_back(4);
    auto ds = root.CreateDataset("col_slice", Datatype::Int32(), space.value(), props);
    REQUIRE(ds.has_value());

    // Write all elements: value = row * 10 + col
    int32_t buf[100];
    for (int row = 0; row < 10; row++)
        for (int col = 0; col < 10; col++)
            buf[row * 10 + col] = row * 10 + col;

    auto all_space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 2));
    REQUIRE(all_space.has_value());
    REQUIRE(ds.value().Write(Datatype::Int32(), all_space.value(), all_space.value(), buf).has_value());

    // Select column 5: start={0,5}, stride={1,1}, count={10,1}, block={1,1}
    uint64_t read_dims[] = {10};
    auto ms = Dataspace::CreateSimple(cstd::span<const uint64_t>(read_dims, 1));
    REQUIRE(ms.has_value());
    auto fs = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 2));
    REQUIRE(fs.has_value());

    uint64_t start[] = {0, 5}, stride[] = {1, 1}, count[] = {10, 1}, block[] = {1, 1};
    REQUIRE(fs.value().SelectHyperslab(SelectionOp::Set,
        cstd::span<const uint64_t>(start, 2), cstd::span<const uint64_t>(stride, 2),
        cstd::span<const uint64_t>(count, 2), cstd::span<const uint64_t>(block, 2)).has_value());

    int32_t out[10] = {};
    REQUIRE(ds.value().Read(Datatype::Int32(), ms.value(), fs.value(), out).has_value());

    // Column 5: values 5, 15, 25, 35, 45, 55, 65, 75, 85, 95
    for (int row = 0; row < 10; row++) {
        REQUIRE(out[row] == row * 10 + 5);
    }
}

TEST_CASE("Hyperslab: strided 1D every Nth element", "[dataset_comprehensive][hyperslab]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());
    auto file = File<InMemoryBlobStore>::Create(
        InMemoryBlobStore(fixture.allocator), Context(fixture.allocator));
    REQUIRE(file.has_value());
    auto root = file.value().OpenRootGroup();

    // 60 elements, chunk=20 => 3 chunks
    uint64_t dims[] = {60};
    auto space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(space.has_value());

    DatasetCreateProps props;
    props.chunk_dims.push_back(20);
    auto ds = root.CreateDataset("strided_1d", Datatype::Int32(), space.value(), props);
    REQUIRE(ds.has_value());

    // Write all elements: value = index
    int32_t buf[60];
    for (int i = 0; i < 60; i++) buf[i] = i;

    auto all_space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(all_space.has_value());
    REQUIRE(ds.value().Write(Datatype::Int32(), all_space.value(), all_space.value(), buf).has_value());

    // Select every 5th element: start=0, stride=5, count=12, block=1
    // This selects: 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55
    uint64_t read_dims[] = {12};
    auto ms = Dataspace::CreateSimple(cstd::span<const uint64_t>(read_dims, 1));
    REQUIRE(ms.has_value());
    auto fs = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(fs.has_value());

    uint64_t start[] = {0}, stride[] = {5}, count[] = {12}, block[] = {1};
    REQUIRE(fs.value().SelectHyperslab(SelectionOp::Set,
        cstd::span<const uint64_t>(start, 1), cstd::span<const uint64_t>(stride, 1),
        cstd::span<const uint64_t>(count, 1), cstd::span<const uint64_t>(block, 1)).has_value());

    int32_t out[12] = {};
    REQUIRE(ds.value().Read(Datatype::Int32(), ms.value(), fs.value(), out).has_value());

    // Expected: 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55
    for (int i = 0; i < 12; i++) {
        REQUIRE(out[i] == i * 5);
    }
}

TEST_CASE("Hyperslab: rectangular block within single chunk", "[dataset_comprehensive][hyperslab]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());
    auto file = File<InMemoryBlobStore>::Create(
        InMemoryBlobStore(fixture.allocator), Context(fixture.allocator));
    REQUIRE(file.has_value());
    auto root = file.value().OpenRootGroup();

    // 10x10 dataset with a single chunk (10x10)
    uint64_t dims[] = {10, 10};
    auto space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 2));
    REQUIRE(space.has_value());

    // No props => chunk_dims defaults to dims (single chunk)
    auto ds = root.CreateDataset("rect_single_chunk", Datatype::Int32(), space.value());
    REQUIRE(ds.has_value());

    // Write all elements: value = row * 10 + col
    int32_t buf[100];
    for (int row = 0; row < 10; row++)
        for (int col = 0; col < 10; col++)
            buf[row * 10 + col] = row * 10 + col;

    auto all_space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 2));
    REQUIRE(all_space.has_value());
    REQUIRE(ds.value().Write(Datatype::Int32(), all_space.value(), all_space.value(), buf).has_value());

    // Select 3x3 block at (2,3): start={2,3}, stride={1,1}, count={3,3}, block={1,1}
    uint64_t read_dims[] = {9};
    auto ms = Dataspace::CreateSimple(cstd::span<const uint64_t>(read_dims, 1));
    REQUIRE(ms.has_value());
    auto fs = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 2));
    REQUIRE(fs.has_value());

    uint64_t start[] = {2, 3}, stride[] = {1, 1}, count[] = {3, 3}, block[] = {1, 1};
    REQUIRE(fs.value().SelectHyperslab(SelectionOp::Set,
        cstd::span<const uint64_t>(start, 2), cstd::span<const uint64_t>(stride, 2),
        cstd::span<const uint64_t>(count, 2), cstd::span<const uint64_t>(block, 2)).has_value());

    int32_t out[9] = {};
    REQUIRE(ds.value().Read(Datatype::Int32(), ms.value(), fs.value(), out).has_value());

    // The hyperslab iterates rows [2,3,4], cols [3,4,5] in row-major order
    // out[0]=buf[2*10+3]=23, out[1]=24, out[2]=25,
    // out[3]=buf[3*10+3]=33, out[4]=34, out[5]=35,
    // out[6]=buf[4*10+3]=43, out[7]=44, out[8]=45
    int expected[9] = {23, 24, 25, 33, 34, 35, 43, 44, 45};
    for (int i = 0; i < 9; i++) {
        REQUIRE(out[i] == expected[i]);
    }
}

TEST_CASE("Hyperslab: rectangular block spanning chunks", "[dataset_comprehensive][hyperslab]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());
    auto file = File<InMemoryBlobStore>::Create(
        InMemoryBlobStore(fixture.allocator), Context(fixture.allocator));
    REQUIRE(file.has_value());
    auto root = file.value().OpenRootGroup();

    // 10x10 dataset with 4x4 chunks
    uint64_t dims[] = {10, 10};
    auto space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 2));
    REQUIRE(space.has_value());

    DatasetCreateProps props;
    props.chunk_dims.push_back(4);
    props.chunk_dims.push_back(4);
    auto ds = root.CreateDataset("rect_spanning", Datatype::Int32(), space.value(), props);
    REQUIRE(ds.has_value());

    // Write all elements: value = row * 10 + col
    int32_t buf[100];
    for (int row = 0; row < 10; row++)
        for (int col = 0; col < 10; col++)
            buf[row * 10 + col] = row * 10 + col;

    auto all_space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 2));
    REQUIRE(all_space.has_value());
    REQUIRE(ds.value().Write(Datatype::Int32(), all_space.value(), all_space.value(), buf).has_value());

    // Select 4x4 block at (3,3): start={3,3}, stride={1,1}, count={4,4}, block={1,1}
    // This spans chunk (0,0), (0,1), (1,0), (1,1) in 4x4 chunk grid
    // Rows 3-6, cols 3-6
    uint64_t read_dims[] = {16};
    auto ms = Dataspace::CreateSimple(cstd::span<const uint64_t>(read_dims, 1));
    REQUIRE(ms.has_value());
    auto fs = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 2));
    REQUIRE(fs.has_value());

    uint64_t start[] = {3, 3}, stride[] = {1, 1}, count[] = {4, 4}, block[] = {1, 1};
    REQUIRE(fs.value().SelectHyperslab(SelectionOp::Set,
        cstd::span<const uint64_t>(start, 2), cstd::span<const uint64_t>(stride, 2),
        cstd::span<const uint64_t>(count, 2), cstd::span<const uint64_t>(block, 2)).has_value());

    int32_t out[16] = {};
    REQUIRE(ds.value().Read(Datatype::Int32(), ms.value(), fs.value(), out).has_value());

    // Rows 3-6, cols 3-6 in row-major order
    // out[0]=33, out[1]=34, out[2]=35, out[3]=36
    // out[4]=43, out[5]=44, out[6]=45, out[7]=46
    // out[8]=53, out[9]=54, out[10]=55, out[11]=56
    // out[12]=63, out[13]=64, out[14]=65, out[15]=66
    int n = 0;
    for (int row = 3; row < 7; row++) {
        for (int col = 3; col < 7; col++) {
            REQUIRE(out[n] == row * 10 + col);
            n++;
        }
    }
}

TEST_CASE("Hyperslab: block with stride (multiple sub-blocks)", "[dataset_comprehensive][hyperslab]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());
    auto file = File<InMemoryBlobStore>::Create(
        InMemoryBlobStore(fixture.allocator), Context(fixture.allocator));
    REQUIRE(file.has_value());
    auto root = file.value().OpenRootGroup();

    // 20 elements, chunk=10 => 2 chunks
    uint64_t dims[] = {20};
    auto space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(space.has_value());

    DatasetCreateProps props;
    props.chunk_dims.push_back(10);
    auto ds = root.CreateDataset("block_stride", Datatype::Int32(), space.value(), props);
    REQUIRE(ds.has_value());

    // Write all elements: value = index
    int32_t buf[20];
    for (int i = 0; i < 20; i++) buf[i] = i;

    auto all_space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(all_space.has_value());
    REQUIRE(ds.value().Write(Datatype::Int32(), all_space.value(), all_space.value(), buf).has_value());

    // Select with start=0, stride=6, count=3, block=2
    // Block 0: elements 0, 1  (start + 0*stride + [0..block-1] = 0,1)
    // Block 1: elements 6, 7  (start + 1*stride + [0..block-1] = 6,7)
    // Block 2: elements 12,13 (start + 2*stride + [0..block-1] = 12,13)
    // Total: {0, 1, 6, 7, 12, 13}
    uint64_t read_dims[] = {6};
    auto ms = Dataspace::CreateSimple(cstd::span<const uint64_t>(read_dims, 1));
    REQUIRE(ms.has_value());
    auto fs = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(fs.has_value());

    uint64_t start[] = {0}, stride[] = {6}, count[] = {3}, block[] = {2};
    REQUIRE(fs.value().SelectHyperslab(SelectionOp::Set,
        cstd::span<const uint64_t>(start, 1), cstd::span<const uint64_t>(stride, 1),
        cstd::span<const uint64_t>(count, 1), cstd::span<const uint64_t>(block, 1)).has_value());

    int32_t out[6] = {};
    REQUIRE(ds.value().Read(Datatype::Int32(), ms.value(), fs.value(), out).has_value());

    // Expected: 0, 1, 6, 7, 12, 13
    int expected[6] = {0, 1, 6, 7, 12, 13};
    for (int i = 0; i < 6; i++) {
        REQUIRE(out[i] == expected[i]);
    }
}
