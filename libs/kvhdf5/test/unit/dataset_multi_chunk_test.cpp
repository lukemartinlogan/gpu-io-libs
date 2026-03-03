#include <catch2/catch_test_macros.hpp>
#include <kvhdf5/hdf5_dataset.h>
#include <kvhdf5/memory_blob_store.h>
#include "../common/allocator_fixture.h"

using namespace kvhdf5;
using AllocatorFixture = test::AllocatorFixture<AllocatorImpl>;

TEST_CASE("Multi-chunk 1D: write all, read all (partial last chunk)", "[dataset_comprehensive][multi_chunk]") {
    // 50 elements, chunk=15 => chunks of (15, 15, 15, 5)
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());
    auto file = File<InMemoryBlobStore>::Create(
        InMemoryBlobStore(fixture.allocator), Context(fixture.allocator));
    REQUIRE(file.has_value());
    auto root = file.value().OpenRootGroup();

    uint64_t dims[] = {50};
    auto space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(space.has_value());

    DatasetCreateProps props;
    props.chunk_dims.push_back(15);
    auto ds = root.CreateDataset("partial_last", Datatype::Int32(), space.value(), props);
    REQUIRE(ds.has_value());

    // Fill buffer: buf[i] = i * 7
    int32_t buf[50];
    for (int i = 0; i < 50; i++) buf[i] = i * 7;

    auto ms = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(ms.has_value());
    auto fs = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(fs.has_value());

    REQUIRE(ds.value().Write(Datatype::Int32(), ms.value(), fs.value(), buf).has_value());

    // Read all back
    int32_t out[50] = {};
    auto rms = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(rms.has_value());
    auto rfs = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(rfs.has_value());
    REQUIRE(ds.value().Read(Datatype::Int32(), rms.value(), rfs.value(), out).has_value());

    // Verify all 50 elements across all 4 chunks (15, 15, 15, 5)
    for (int i = 0; i < 50; i++) {
        INFO("element " << i);
        REQUIRE(out[i] == i * 7);
    }
}

TEST_CASE("Multi-chunk 2D: write all, read all (partial edge chunks)", "[dataset_comprehensive][multi_chunk]") {
    // 7x9, chunks 4x4 => chunk grid 2x3. Last row-chunk is 3 rows, last col-chunk is 1 col.
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());
    auto file = File<InMemoryBlobStore>::Create(
        InMemoryBlobStore(fixture.allocator), Context(fixture.allocator));
    REQUIRE(file.has_value());
    auto root = file.value().OpenRootGroup();

    uint64_t dims[] = {7, 9};
    auto space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 2));
    REQUIRE(space.has_value());

    DatasetCreateProps props;
    props.chunk_dims.push_back(4);
    props.chunk_dims.push_back(4);
    auto ds = root.CreateDataset("grid_partial", Datatype::Int32(), space.value(), props);
    REQUIRE(ds.has_value());

    // Fill buffer row-major: buf[r*9 + c] = r * 100 + c
    int32_t buf[63];
    for (int r = 0; r < 7; r++)
        for (int c = 0; c < 9; c++)
            buf[r * 9 + c] = r * 100 + c;

    auto ms = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 2));
    REQUIRE(ms.has_value());
    auto fs = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 2));
    REQUIRE(fs.has_value());

    REQUIRE(ds.value().Write(Datatype::Int32(), ms.value(), fs.value(), buf).has_value());

    // Read all back
    int32_t out[63] = {};
    auto rms = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 2));
    REQUIRE(rms.has_value());
    auto rfs = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 2));
    REQUIRE(rfs.has_value());
    REQUIRE(ds.value().Read(Datatype::Int32(), rms.value(), rfs.value(), out).has_value());

    // Verify all 63 elements, including partial edge chunks
    for (int r = 0; r < 7; r++) {
        for (int c = 0; c < 9; c++) {
            INFO("element (" << r << ", " << c << ")");
            REQUIRE(out[r * 9 + c] == r * 100 + c);
        }
    }
}

TEST_CASE("Multi-chunk: write individual chunks, read all", "[dataset_comprehensive][multi_chunk]") {
    // 30 elements, chunk=10 => 3 chunks.
    // Write chunk 0 (elements 0-9) and chunk 2 (elements 20-29) via hyperslab.
    // Chunk 1 (elements 10-19) is never written => reads as zero.
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());
    auto file = File<InMemoryBlobStore>::Create(
        InMemoryBlobStore(fixture.allocator), Context(fixture.allocator));
    REQUIRE(file.has_value());
    auto root = file.value().OpenRootGroup();

    uint64_t dims[] = {30};
    auto space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(space.has_value());

    DatasetCreateProps props;
    props.chunk_dims.push_back(10);
    auto ds = root.CreateDataset("skip_middle", Datatype::Int32(), space.value(), props);
    REQUIRE(ds.has_value());

    // Write chunk 0: elements 0-9
    int32_t chunk0_buf[10];
    for (int i = 0; i < 10; i++) chunk0_buf[i] = i + 1;  // 1..10

    {
        uint64_t mem_dims[] = {10};
        auto ms = Dataspace::CreateSimple(cstd::span<const uint64_t>(mem_dims, 1));
        REQUIRE(ms.has_value());
        auto fs = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
        REQUIRE(fs.has_value());

        uint64_t start[] = {0}, stride[] = {1}, count[] = {10}, block[] = {1};
        REQUIRE(fs.value().SelectHyperslab(SelectionOp::Set,
            cstd::span<const uint64_t>(start, 1),
            cstd::span<const uint64_t>(stride, 1),
            cstd::span<const uint64_t>(count, 1),
            cstd::span<const uint64_t>(block, 1)).has_value());

        REQUIRE(ds.value().Write(Datatype::Int32(), ms.value(), fs.value(), chunk0_buf).has_value());
    }

    // Write chunk 2: elements 20-29
    int32_t chunk2_buf[10];
    for (int i = 0; i < 10; i++) chunk2_buf[i] = (i + 20) * 3;  // 60, 63, ..., 87

    {
        uint64_t mem_dims[] = {10};
        auto ms = Dataspace::CreateSimple(cstd::span<const uint64_t>(mem_dims, 1));
        REQUIRE(ms.has_value());
        auto fs = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
        REQUIRE(fs.has_value());

        uint64_t start[] = {20}, stride[] = {1}, count[] = {10}, block[] = {1};
        REQUIRE(fs.value().SelectHyperslab(SelectionOp::Set,
            cstd::span<const uint64_t>(start, 1),
            cstd::span<const uint64_t>(stride, 1),
            cstd::span<const uint64_t>(count, 1),
            cstd::span<const uint64_t>(block, 1)).has_value());

        REQUIRE(ds.value().Write(Datatype::Int32(), ms.value(), fs.value(), chunk2_buf).has_value());
    }

    // Read all 30 elements
    int32_t out[30];
    for (int i = 0; i < 30; i++) out[i] = -1;  // sentinel

    auto rms = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(rms.has_value());
    auto rfs = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(rfs.has_value());
    REQUIRE(ds.value().Read(Datatype::Int32(), rms.value(), rfs.value(), out).has_value());

    // Chunk 0: elements 0-9 should equal chunk0_buf
    for (int i = 0; i < 10; i++) {
        INFO("chunk 0 element " << i);
        REQUIRE(out[i] == chunk0_buf[i]);
    }

    // Chunk 1: elements 10-19 should be zero (never written)
    for (int i = 10; i < 20; i++) {
        INFO("chunk 1 element " << i);
        REQUIRE(out[i] == 0);
    }

    // Chunk 2: elements 20-29 should equal chunk2_buf
    for (int i = 0; i < 10; i++) {
        INFO("chunk 2 element " << (i + 20));
        REQUIRE(out[20 + i] == chunk2_buf[i]);
    }
}

TEST_CASE("Multi-chunk: write to non-contiguous chunks", "[dataset_comprehensive][multi_chunk]") {
    // 40 elements, chunk=10 => 4 chunks (indices 0, 1, 2, 3).
    // Write chunk 1 (elements 10-19) and chunk 3 (elements 30-39).
    // Chunks 0 and 2 are never written => zero.
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());
    auto file = File<InMemoryBlobStore>::Create(
        InMemoryBlobStore(fixture.allocator), Context(fixture.allocator));
    REQUIRE(file.has_value());
    auto root = file.value().OpenRootGroup();

    uint64_t dims[] = {40};
    auto space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(space.has_value());

    DatasetCreateProps props;
    props.chunk_dims.push_back(10);
    auto ds = root.CreateDataset("noncontiguous", Datatype::Int32(), space.value(), props);
    REQUIRE(ds.has_value());

    // Write chunk 1: elements 10-19
    int32_t chunk1_buf[10];
    for (int i = 0; i < 10; i++) chunk1_buf[i] = 100 + i;

    {
        uint64_t mem_dims[] = {10};
        auto ms = Dataspace::CreateSimple(cstd::span<const uint64_t>(mem_dims, 1));
        REQUIRE(ms.has_value());
        auto fs = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
        REQUIRE(fs.has_value());

        uint64_t start[] = {10}, stride[] = {1}, count[] = {10}, block[] = {1};
        REQUIRE(fs.value().SelectHyperslab(SelectionOp::Set,
            cstd::span<const uint64_t>(start, 1),
            cstd::span<const uint64_t>(stride, 1),
            cstd::span<const uint64_t>(count, 1),
            cstd::span<const uint64_t>(block, 1)).has_value());

        REQUIRE(ds.value().Write(Datatype::Int32(), ms.value(), fs.value(), chunk1_buf).has_value());
    }

    // Write chunk 3: elements 30-39
    int32_t chunk3_buf[10];
    for (int i = 0; i < 10; i++) chunk3_buf[i] = 300 + i;

    {
        uint64_t mem_dims[] = {10};
        auto ms = Dataspace::CreateSimple(cstd::span<const uint64_t>(mem_dims, 1));
        REQUIRE(ms.has_value());
        auto fs = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
        REQUIRE(fs.has_value());

        uint64_t start[] = {30}, stride[] = {1}, count[] = {10}, block[] = {1};
        REQUIRE(fs.value().SelectHyperslab(SelectionOp::Set,
            cstd::span<const uint64_t>(start, 1),
            cstd::span<const uint64_t>(stride, 1),
            cstd::span<const uint64_t>(count, 1),
            cstd::span<const uint64_t>(block, 1)).has_value());

        REQUIRE(ds.value().Write(Datatype::Int32(), ms.value(), fs.value(), chunk3_buf).has_value());
    }

    // Read all 40 elements
    int32_t out[40];
    for (int i = 0; i < 40; i++) out[i] = -1;  // sentinel

    auto rms = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(rms.has_value());
    auto rfs = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(rfs.has_value());
    REQUIRE(ds.value().Read(Datatype::Int32(), rms.value(), rfs.value(), out).has_value());

    // Chunk 0: elements 0-9 => zero (never written)
    for (int i = 0; i < 10; i++) {
        INFO("chunk 0 element " << i);
        REQUIRE(out[i] == 0);
    }

    // Chunk 1: elements 10-19 => chunk1_buf
    for (int i = 0; i < 10; i++) {
        INFO("chunk 1 element " << (i + 10));
        REQUIRE(out[10 + i] == chunk1_buf[i]);
    }

    // Chunk 2: elements 20-29 => zero (never written)
    for (int i = 20; i < 30; i++) {
        INFO("chunk 2 element " << i);
        REQUIRE(out[i] == 0);
    }

    // Chunk 3: elements 30-39 => chunk3_buf
    for (int i = 0; i < 10; i++) {
        INFO("chunk 3 element " << (i + 30));
        REQUIRE(out[30 + i] == chunk3_buf[i]);
    }
}

TEST_CASE("Multi-chunk: write across chunk boundary", "[dataset_comprehensive][multi_chunk]") {
    // 20 elements, chunk=8 => 3 chunks: [0-7], [8-15], [16-19].
    // Write elements 6-13 (spans the boundary between chunk 0 and chunk 1).
    // Elements 0-5 and 14-19 are never written => zero.
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());
    auto file = File<InMemoryBlobStore>::Create(
        InMemoryBlobStore(fixture.allocator), Context(fixture.allocator));
    REQUIRE(file.has_value());
    auto root = file.value().OpenRootGroup();

    uint64_t dims[] = {20};
    auto space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(space.has_value());

    DatasetCreateProps props;
    props.chunk_dims.push_back(8);
    auto ds = root.CreateDataset("boundary_cross", Datatype::Int32(), space.value(), props);
    REQUIRE(ds.has_value());

    // Write elements 6-13 (8 elements spanning the chunk boundary at index 8)
    int32_t write_buf[8];
    for (int i = 0; i < 8; i++) write_buf[i] = (6 + i) * 10;  // 60, 70, 80, ..., 130

    {
        uint64_t mem_dims[] = {8};
        auto ms = Dataspace::CreateSimple(cstd::span<const uint64_t>(mem_dims, 1));
        REQUIRE(ms.has_value());
        auto fs = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
        REQUIRE(fs.has_value());

        uint64_t start[] = {6}, stride[] = {1}, count[] = {8}, block[] = {1};
        REQUIRE(fs.value().SelectHyperslab(SelectionOp::Set,
            cstd::span<const uint64_t>(start, 1),
            cstd::span<const uint64_t>(stride, 1),
            cstd::span<const uint64_t>(count, 1),
            cstd::span<const uint64_t>(block, 1)).has_value());

        REQUIRE(ds.value().Write(Datatype::Int32(), ms.value(), fs.value(), write_buf).has_value());
    }

    // Read all 20 elements
    int32_t out[20];
    for (int i = 0; i < 20; i++) out[i] = -1;  // sentinel

    auto rms = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(rms.has_value());
    auto rfs = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(rfs.has_value());
    REQUIRE(ds.value().Read(Datatype::Int32(), rms.value(), rfs.value(), out).has_value());

    // Elements 0-5: zero (unwritten portion of chunk 0)
    for (int i = 0; i < 6; i++) {
        INFO("unwritten element " << i);
        REQUIRE(out[i] == 0);
    }

    // Elements 6-13: written values spanning chunk boundary [0-7] / [8-15]
    for (int i = 0; i < 8; i++) {
        INFO("written element " << (6 + i));
        REQUIRE(out[6 + i] == write_buf[i]);
    }

    // Elements 14-19: zero (unwritten portion of chunks 1 and 2)
    for (int i = 14; i < 20; i++) {
        INFO("unwritten element " << i);
        REQUIRE(out[i] == 0);
    }
}
