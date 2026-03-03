#include <catch2/catch_test_macros.hpp>
#include <kvhdf5/hdf5_dataset.h>
#include <kvhdf5/memory_blob_store.h>
#include "../common/allocator_fixture.h"

using namespace kvhdf5;
using AllocatorFixture = test::AllocatorFixture<AllocatorImpl>;

TEST_CASE("Create and write single-chunk 1D dataset", "[hdf5_dataset]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());
    auto file = File<InMemoryBlobStore>::Create(
        InMemoryBlobStore(fixture.allocator), Context(fixture.allocator));
    REQUIRE(file.has_value());
    auto root = file.value().OpenRootGroup();

    uint64_t dims[] = {10};
    auto space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(space.has_value());
    auto type = Datatype::Int32();

    auto ds = root.CreateDataset("data", type, space.value());
    REQUIRE(ds.has_value());

    // Write 10 int32s
    int32_t buf[10];
    for (int i = 0; i < 10; i++) buf[i] = i * 100;

    auto mem_space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(mem_space.has_value());
    auto file_space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(file_space.has_value());

    auto err = ds.value().Write(type, mem_space.value(), file_space.value(), buf);
    REQUIRE(err.has_value());

    // Read back
    int32_t out[10] = {};
    auto err2 = ds.value().Read(type, mem_space.value(), file_space.value(), out);
    REQUIRE(err2.has_value());
    for (int i = 0; i < 10; i++) REQUIRE(out[i] == i * 100);
}

TEST_CASE("Create and write single-chunk 2D dataset", "[hdf5_dataset]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());
    auto file = File<InMemoryBlobStore>::Create(
        InMemoryBlobStore(fixture.allocator), Context(fixture.allocator));
    REQUIRE(file.has_value());
    auto root = file.value().OpenRootGroup();

    uint64_t dims[] = {4, 5};
    auto space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 2));
    REQUIRE(space.has_value());
    auto type = Datatype::Float32();

    auto ds = root.CreateDataset("matrix", type, space.value());
    REQUIRE(ds.has_value());

    float buf[20];
    for (int i = 0; i < 20; i++) buf[i] = float(i);

    auto ms = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 2));
    REQUIRE(ms.has_value());
    auto fs = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 2));
    REQUIRE(fs.has_value());
    REQUIRE(ds.value().Write(type, ms.value(), fs.value(), buf).has_value());

    float out[20] = {};
    REQUIRE(ds.value().Read(type, ms.value(), fs.value(), out).has_value());
    for (int i = 0; i < 20; i++) REQUIRE(out[i] == float(i));
}

TEST_CASE("OpenDataset from group", "[hdf5_dataset]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());
    auto file = File<InMemoryBlobStore>::Create(
        InMemoryBlobStore(fixture.allocator), Context(fixture.allocator));
    REQUIRE(file.has_value());
    auto root = file.value().OpenRootGroup();

    uint64_t dims[] = {5};
    auto space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(space.has_value());
    auto ds = root.CreateDataset("d", Datatype::Int32(), space.value());
    REQUIRE(ds.has_value());
    auto ds2 = root.OpenDataset("d");
    REQUIRE(ds2.has_value());
    REQUIRE(ds2.value().GetId() == ds.value().GetId());
}

TEST_CASE("Dataset::GetSpace and GetType", "[hdf5_dataset]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());
    auto file = File<InMemoryBlobStore>::Create(
        InMemoryBlobStore(fixture.allocator), Context(fixture.allocator));
    REQUIRE(file.has_value());
    auto root = file.value().OpenRootGroup();

    uint64_t dims[] = {10};
    auto space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(space.has_value());
    auto ds = root.CreateDataset("d", Datatype::Float64(), space.value());
    REQUIRE(ds.has_value());

    auto sp = ds.value().GetSpace();
    REQUIRE(sp.has_value());
    REQUIRE(sp.value().GetNDims() == 1);
    REQUIRE(sp.value().GetTotalElements() == 10);

    auto tp = ds.value().GetType();
    REQUIRE(tp.IsPrimitive());
    REQUIRE(tp.GetPrimitiveKind() == PrimitiveType::Kind::Float64);
}

TEST_CASE("Multi-chunk 1D write/read all", "[hdf5_dataset][multichunk]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());
    auto file = File<InMemoryBlobStore>::Create(
        InMemoryBlobStore(fixture.allocator), Context(fixture.allocator));
    REQUIRE(file.has_value());
    auto root = file.value().OpenRootGroup();

    // 100 elements, chunks of 30 => 4 chunks (30, 30, 30, 10)
    uint64_t dims[] = {100};
    auto space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(space.has_value());
    DatasetCreateProps props;
    props.chunk_dims.push_back(30);
    auto ds = root.CreateDataset("chunked", Datatype::Int32(), space.value(), props);
    REQUIRE(ds.has_value());

    int32_t buf[100];
    for (int i = 0; i < 100; i++) buf[i] = i;

    auto ms = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(ms.has_value());
    auto fs = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(fs.has_value());
    REQUIRE(ds.value().Write(Datatype::Int32(), ms.value(), fs.value(), buf).has_value());

    int32_t out[100] = {};
    REQUIRE(ds.value().Read(Datatype::Int32(), ms.value(), fs.value(), out).has_value());
    for (int i = 0; i < 100; i++) REQUIRE(out[i] == i);
}

TEST_CASE("Multi-chunk 2D write/read all", "[hdf5_dataset][multichunk]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());
    auto file = File<InMemoryBlobStore>::Create(
        InMemoryBlobStore(fixture.allocator), Context(fixture.allocator));
    REQUIRE(file.has_value());
    auto root = file.value().OpenRootGroup();

    // 10x10, chunks of 4x4 => 3x3 chunk grid (last chunks partial)
    uint64_t dims[] = {10, 10};
    auto space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 2));
    REQUIRE(space.has_value());
    DatasetCreateProps props;
    props.chunk_dims.push_back(4);
    props.chunk_dims.push_back(4);
    auto ds = root.CreateDataset("grid", Datatype::Float32(), space.value(), props);
    REQUIRE(ds.has_value());

    float buf[100];
    for (int i = 0; i < 100; i++) buf[i] = float(i);

    auto ms = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 2));
    REQUIRE(ms.has_value());
    auto fs = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 2));
    REQUIRE(fs.has_value());
    REQUIRE(ds.value().Write(Datatype::Float32(), ms.value(), fs.value(), buf).has_value());

    float out[100] = {};
    REQUIRE(ds.value().Read(Datatype::Float32(), ms.value(), fs.value(), out).has_value());
    for (int i = 0; i < 100; i++) REQUIRE(out[i] == float(i));
}

TEST_CASE("Multi-chunk 1D partial last chunk", "[hdf5_dataset][multichunk]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());
    auto file = File<InMemoryBlobStore>::Create(
        InMemoryBlobStore(fixture.allocator), Context(fixture.allocator));
    REQUIRE(file.has_value());
    auto root = file.value().OpenRootGroup();

    // 7 elements, chunk size 3 => chunks of (3, 3, 1)
    uint64_t dims[] = {7};
    auto space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(space.has_value());
    DatasetCreateProps props;
    props.chunk_dims.push_back(3);
    auto ds = root.CreateDataset("partial", Datatype::Int32(), space.value(), props);
    REQUIRE(ds.has_value());

    int32_t buf[] = {10, 20, 30, 40, 50, 60, 70};
    auto ms = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(ms.has_value());
    auto fs = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(fs.has_value());
    REQUIRE(ds.value().Write(Datatype::Int32(), ms.value(), fs.value(), buf).has_value());

    int32_t out[7] = {};
    REQUIRE(ds.value().Read(Datatype::Int32(), ms.value(), fs.value(), out).has_value());
    for (int i = 0; i < 7; i++) REQUIRE(out[i] == buf[i]);
}

TEST_CASE("Single-chunk still works after multi-chunk changes", "[hdf5_dataset]") {
    // Re-run a basic single-chunk test to ensure no regression
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());
    auto file = File<InMemoryBlobStore>::Create(
        InMemoryBlobStore(fixture.allocator), Context(fixture.allocator));
    REQUIRE(file.has_value());
    auto root = file.value().OpenRootGroup();

    uint64_t dims[] = {5};
    auto space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(space.has_value());
    auto ds = root.CreateDataset("single", Datatype::Int32(), space.value());
    REQUIRE(ds.has_value());

    int32_t buf[] = {1, 2, 3, 4, 5};
    auto ms = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(ms.has_value());
    auto fs = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(fs.has_value());
    REQUIRE(ds.value().Write(Datatype::Int32(), ms.value(), fs.value(), buf).has_value());

    int32_t out[5] = {};
    REQUIRE(ds.value().Read(Datatype::Int32(), ms.value(), fs.value(), out).has_value());
    for (int i = 0; i < 5; i++) REQUIRE(out[i] == buf[i]);
}

TEST_CASE("Hyperslab write to single-chunk dataset", "[hdf5_dataset][hyperslab]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());
    auto file = File<InMemoryBlobStore>::Create(
        InMemoryBlobStore(fixture.allocator), Context(fixture.allocator));
    REQUIRE(file.has_value());
    auto root = file.value().OpenRootGroup();

    uint64_t dims[] = {10};
    auto space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(space.has_value());
    auto ds = root.CreateDataset("d", Datatype::Int32(), space.value());
    REQUIRE(ds.has_value());

    // Write to elements 2-5
    int32_t write_buf[] = {20, 30, 40, 50};
    uint64_t mem_dims[] = {4};
    auto ms = Dataspace::CreateSimple(cstd::span<const uint64_t>(mem_dims, 1));
    REQUIRE(ms.has_value());
    auto fs = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(fs.has_value());

    uint64_t start[] = {2}, stride[] = {1}, count[] = {4}, block[] = {1};
    REQUIRE(fs.value().SelectHyperslab(SelectionOp::Set,
        cstd::span<const uint64_t>(start, 1),
        cstd::span<const uint64_t>(stride, 1),
        cstd::span<const uint64_t>(count, 1),
        cstd::span<const uint64_t>(block, 1)).has_value());

    REQUIRE(ds.value().Write(Datatype::Int32(), ms.value(), fs.value(), write_buf).has_value());

    // Read back full dataset
    int32_t full[10] = {};
    auto fs2 = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(fs2.has_value());
    REQUIRE(ds.value().Read(Datatype::Int32(), fs2.value(), fs2.value(), full).has_value());

    REQUIRE(full[0] == 0);
    REQUIRE(full[1] == 0);
    REQUIRE(full[2] == 20);
    REQUIRE(full[3] == 30);
    REQUIRE(full[4] == 40);
    REQUIRE(full[5] == 50);
    REQUIRE(full[6] == 0);
}

TEST_CASE("Hyperslab read from multi-chunk dataset", "[hdf5_dataset][hyperslab]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());
    auto file = File<InMemoryBlobStore>::Create(
        InMemoryBlobStore(fixture.allocator), Context(fixture.allocator));
    REQUIRE(file.has_value());
    auto root = file.value().OpenRootGroup();

    // 100 elements, chunk=30, write all, then read elements 25-34 (crosses chunk boundary)
    uint64_t dims[] = {100};
    auto space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(space.has_value());
    DatasetCreateProps props;
    props.chunk_dims.push_back(30);
    auto ds = root.CreateDataset("chunked", Datatype::Int32(), space.value(), props);
    REQUIRE(ds.has_value());

    int32_t buf[100];
    for (int i = 0; i < 100; i++) buf[i] = i;
    auto all_space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(all_space.has_value());
    REQUIRE(ds.value().Write(Datatype::Int32(), all_space.value(), all_space.value(), buf).has_value());

    // Read elements 25-34
    uint64_t read_dims[] = {10};
    auto ms = Dataspace::CreateSimple(cstd::span<const uint64_t>(read_dims, 1));
    REQUIRE(ms.has_value());
    auto fs = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(fs.has_value());
    uint64_t start[] = {25}, stride[] = {1}, count[] = {10}, block[] = {1};
    REQUIRE(fs.value().SelectHyperslab(SelectionOp::Set,
        cstd::span<const uint64_t>(start, 1), cstd::span<const uint64_t>(stride, 1),
        cstd::span<const uint64_t>(count, 1), cstd::span<const uint64_t>(block, 1)).has_value());

    int32_t out[10] = {};
    REQUIRE(ds.value().Read(Datatype::Int32(), ms.value(), fs.value(), out).has_value());
    for (int i = 0; i < 10; i++) REQUIRE(out[i] == 25 + i);
}

TEST_CASE("Hyperslab 2D row slice across chunks", "[hdf5_dataset][hyperslab]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());
    auto file = File<InMemoryBlobStore>::Create(
        InMemoryBlobStore(fixture.allocator), Context(fixture.allocator));
    REQUIRE(file.has_value());
    auto root = file.value().OpenRootGroup();

    // 10x10, 4x4 chunks, write all, select row 5
    uint64_t dims[] = {10, 10};
    auto space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 2));
    REQUIRE(space.has_value());
    DatasetCreateProps props;
    props.chunk_dims.push_back(4);
    props.chunk_dims.push_back(4);
    auto ds = root.CreateDataset("grid", Datatype::Int32(), space.value(), props);
    REQUIRE(ds.has_value());

    int32_t buf[100];
    for (int i = 0; i < 100; i++) buf[i] = i;
    auto all_space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 2));
    REQUIRE(all_space.has_value());
    REQUIRE(ds.value().Write(Datatype::Int32(), all_space.value(), all_space.value(), buf).has_value());

    // Read row 5 (10 elements)
    uint64_t read_dims[] = {10};
    auto ms = Dataspace::CreateSimple(cstd::span<const uint64_t>(read_dims, 1));
    REQUIRE(ms.has_value());
    auto fs = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 2));
    REQUIRE(fs.has_value());
    uint64_t start[] = {5, 0}, stride[] = {1, 1}, count[] = {1, 10}, block[] = {1, 1};
    REQUIRE(fs.value().SelectHyperslab(SelectionOp::Set,
        cstd::span<const uint64_t>(start, 2), cstd::span<const uint64_t>(stride, 2),
        cstd::span<const uint64_t>(count, 2), cstd::span<const uint64_t>(block, 2)).has_value());

    int32_t out[10] = {};
    REQUIRE(ds.value().Read(Datatype::Int32(), ms.value(), fs.value(), out).has_value());
    // Row 5 = elements 50-59
    for (int i = 0; i < 10; i++) REQUIRE(out[i] == 50 + i);
}

TEST_CASE("Strided hyperslab across chunks", "[hdf5_dataset][hyperslab]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());
    auto file = File<InMemoryBlobStore>::Create(
        InMemoryBlobStore(fixture.allocator), Context(fixture.allocator));
    REQUIRE(file.has_value());
    auto root = file.value().OpenRootGroup();

    // 100 elements, chunk=30, select every 3rd element starting at 0
    uint64_t dims[] = {100};
    auto space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(space.has_value());
    DatasetCreateProps props;
    props.chunk_dims.push_back(30);
    auto ds = root.CreateDataset("strided", Datatype::Int32(), space.value(), props);
    REQUIRE(ds.has_value());

    int32_t buf[100];
    for (int i = 0; i < 100; i++) buf[i] = i;
    auto all_space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(all_space.has_value());
    REQUIRE(ds.value().Write(Datatype::Int32(), all_space.value(), all_space.value(), buf).has_value());

    // Read every 3rd element: 0, 3, 6, 9, ... => 34 elements (count=34, stride=3)
    uint64_t read_dims[] = {34};
    auto ms = Dataspace::CreateSimple(cstd::span<const uint64_t>(read_dims, 1));
    REQUIRE(ms.has_value());
    auto fs = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(fs.has_value());
    uint64_t start[] = {0}, s[] = {3}, count[] = {34}, block[] = {1};
    REQUIRE(fs.value().SelectHyperslab(SelectionOp::Set,
        cstd::span<const uint64_t>(start, 1), cstd::span<const uint64_t>(s, 1),
        cstd::span<const uint64_t>(count, 1), cstd::span<const uint64_t>(block, 1)).has_value());

    int32_t out[34] = {};
    REQUIRE(ds.value().Read(Datatype::Int32(), ms.value(), fs.value(), out).has_value());
    for (int i = 0; i < 34; i++) REQUIRE(out[i] == i * 3);
}

TEST_CASE("SetExtent grows dataset", "[hdf5_dataset][extent]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());
    auto file = File<InMemoryBlobStore>::Create(
        InMemoryBlobStore(fixture.allocator), Context(fixture.allocator));
    REQUIRE(file.has_value());
    auto root = file.value().OpenRootGroup();

    uint64_t dims[] = {10};
    auto space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(space.has_value());
    auto ds = root.CreateDataset("d", Datatype::Int32(), space.value());
    REQUIRE(ds.has_value());

    // Write initial data
    int32_t buf[10];
    for (int i = 0; i < 10; i++) buf[i] = i;
    auto ms = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(ms.has_value());
    auto fs = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(fs.has_value());
    REQUIRE(ds.value().Write(Datatype::Int32(), ms.value(), fs.value(), buf).has_value());

    // Extend to 20
    uint64_t new_dims[] = {20};
    REQUIRE(ds.value().SetExtent(cstd::span<const uint64_t>(new_dims, 1)).has_value());

    auto sp = ds.value().GetSpace();
    REQUIRE(sp.has_value());
    REQUIRE(sp.value().GetTotalElements() == 20);

    // Read all 20 - old data preserved, new region is zero
    // Original chunk_dims == 10 (defaulted to dims at creation).
    // After extending to 20: num_chunks = ceil(20/10) = 2.
    // Chunk 0 has elements 0-9 (written), chunk 1 has elements 10-19 (does not exist = zeros).
    uint64_t read_dims[] = {20};
    auto rms = Dataspace::CreateSimple(cstd::span<const uint64_t>(read_dims, 1));
    REQUIRE(rms.has_value());
    auto rfs = Dataspace::CreateSimple(cstd::span<const uint64_t>(read_dims, 1));
    REQUIRE(rfs.has_value());

    int32_t out[20] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
    REQUIRE(ds.value().Read(Datatype::Int32(), rms.value(), rfs.value(), out).has_value());
    for (int i = 0; i < 10; i++) REQUIRE(out[i] == i);
    for (int i = 10; i < 20; i++) REQUIRE(out[i] == 0);
}

TEST_CASE("SetExtent rank mismatch returns error", "[hdf5_dataset][extent]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());
    auto file = File<InMemoryBlobStore>::Create(
        InMemoryBlobStore(fixture.allocator), Context(fixture.allocator));
    REQUIRE(file.has_value());
    auto root = file.value().OpenRootGroup();

    uint64_t dims[] = {10};
    auto space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(space.has_value());
    auto ds = root.CreateDataset("d", Datatype::Int32(), space.value());
    REQUIRE(ds.has_value());

    // Try to set extent with wrong rank (2D on a 1D dataset)
    uint64_t bad_dims[] = {10, 10};
    auto result = ds.value().SetExtent(cstd::span<const uint64_t>(bad_dims, 2));
    REQUIRE(!result.has_value());
}

TEST_CASE("ChunkExists works after Write", "[hdf5_dataset][chunk_iter][debug]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());
    auto file = File<InMemoryBlobStore>::Create(
        InMemoryBlobStore(fixture.allocator), Context(fixture.allocator));
    REQUIRE(file.has_value());
    auto root = file.value().OpenRootGroup();

    // 10 elements, single chunk (default chunk_dims = dims = 10)
    uint64_t dims[] = {10};
    auto space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(space.has_value());
    auto ds = root.CreateDataset("dbg", Datatype::Int32(), space.value());
    REQUIRE(ds.has_value());

    int32_t buf[10];
    for (int i = 0; i < 10; i++) buf[i] = i;
    auto ms = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(ms.has_value());
    auto fs = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(fs.has_value());
    REQUIRE(ds.value().Write(Datatype::Int32(), ms.value(), fs.value(), buf).has_value());

    // Read back to verify data is there
    int32_t out[10] = {};
    REQUIRE(ds.value().Read(Datatype::Int32(), ms.value(), fs.value(), out).has_value());
    REQUIRE(out[0] == 0);
    REQUIRE(out[9] == 9);

    // Directly test ChunkExists on the container
    auto& container = file.value().GetContainer();
    uint64_t coords[] = {0};
    ChunkKey key(ds.value().GetId(), cstd::span<const uint64_t>(coords, 1));
    INFO("ChunkExists direct: " << container.ChunkExists(key));
    REQUIRE(container.ChunkExists(key));

    // Now iterate with ChunkIter - should find 1 chunk
    int count = 0;
    auto cb = [](const ChunkKey& key, uint64_t size, void* data) -> bool {
        (void)key; (void)size;
        (*static_cast<int*>(data))++;
        return true;
    };
    auto iter_result = ds.value().ChunkIter(cb, &count);
    REQUIRE(iter_result.has_value());
    REQUIRE(count == 1);
}

TEST_CASE("ChunkIter visits written chunks", "[hdf5_dataset][chunk_iter]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());
    auto file = File<InMemoryBlobStore>::Create(
        InMemoryBlobStore(fixture.allocator), Context(fixture.allocator));
    REQUIRE(file.has_value());
    auto root = file.value().OpenRootGroup();

    // 100 elements, chunks of 30 => 4 chunks (30, 30, 30, 10)
    uint64_t dims[] = {100};
    auto space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(space.has_value());
    DatasetCreateProps props;
    props.chunk_dims.push_back(30);
    auto ds = root.CreateDataset("chunked", Datatype::Int32(), space.value(), props);
    REQUIRE(ds.has_value());

    // Write all data
    int32_t buf[100];
    for (int i = 0; i < 100; i++) buf[i] = i;
    auto ms = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(ms.has_value());
    auto fs = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(fs.has_value());
    REQUIRE(ds.value().Write(Datatype::Int32(), ms.value(), fs.value(), buf).has_value());

    // Iterate chunks
    int count = 0;
    auto cb = [](const ChunkKey& key, uint64_t size, void* data) -> bool {
        (void)key; (void)size;
        (*static_cast<int*>(data))++;
        return true;
    };
    REQUIRE(ds.value().ChunkIter(cb, &count).has_value());
    REQUIRE(count == 4);  // ceil(100/30) = 4 chunks
}

TEST_CASE("ChunkIter early termination", "[hdf5_dataset][chunk_iter]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());
    auto file = File<InMemoryBlobStore>::Create(
        InMemoryBlobStore(fixture.allocator), Context(fixture.allocator));
    REQUIRE(file.has_value());
    auto root = file.value().OpenRootGroup();

    uint64_t dims[] = {100};
    auto space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(space.has_value());
    DatasetCreateProps props;
    props.chunk_dims.push_back(10);
    auto ds = root.CreateDataset("many", Datatype::Int32(), space.value(), props);
    REQUIRE(ds.has_value());

    int32_t buf[100];
    for (int i = 0; i < 100; i++) buf[i] = i;
    auto ms = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(ms.has_value());
    auto fs = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(fs.has_value());
    REQUIRE(ds.value().Write(Datatype::Int32(), ms.value(), fs.value(), buf).has_value());

    // Stop after 3 chunks
    int count = 0;
    auto cb = [](const ChunkKey& key, uint64_t size, void* data) -> bool {
        (void)key; (void)size;
        int& c = *static_cast<int*>(data);
        c++;
        return c < 3;  // stop after 3
    };
    REQUIRE(ds.value().ChunkIter(cb, &count).has_value());
    REQUIRE(count == 3);
}

TEST_CASE("Read from unwritten single-chunk dataset returns zeros", "[hdf5_dataset]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());
    auto file = File<InMemoryBlobStore>::Create(
        InMemoryBlobStore(fixture.allocator), Context(fixture.allocator));
    REQUIRE(file.has_value());
    auto root = file.value().OpenRootGroup();

    uint64_t dims[] = {5};
    auto space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(space.has_value());
    auto ds = root.CreateDataset("d", Datatype::Int32(), space.value());
    REQUIRE(ds.has_value());

    int32_t out[5] = {99, 99, 99, 99, 99};
    auto ms = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(ms.has_value());
    auto fs = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(fs.has_value());
    REQUIRE(ds.value().Read(Datatype::Int32(), ms.value(), fs.value(), out).has_value());
    for (int i = 0; i < 5; i++) REQUIRE(out[i] == 0);
}
