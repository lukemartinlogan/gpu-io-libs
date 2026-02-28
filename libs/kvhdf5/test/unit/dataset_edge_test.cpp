#include <catch2/catch_test_macros.hpp>
#include <kvhdf5/hdf5_dataset.h>
#include <kvhdf5/memory_blob_store.h>
#include "../common/allocator_fixture.h"

using namespace kvhdf5;
using AllocatorFixture = test::AllocatorFixture<AllocatorImpl>;

TEST_CASE("Edge: 1-element dataset", "[dataset_comprehensive][edge]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());
    auto file = File<InMemoryBlobStore>::Create(
        InMemoryBlobStore(fixture.allocator), Context(fixture.allocator));
    REQUIRE(file.has_value());
    auto root = file.value().OpenRootGroup();

    uint64_t dims[] = {1};
    auto space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(space.has_value());

    auto ds = root.CreateDataset("single_elem", Datatype::Int32(), space.value());
    REQUIRE(ds.has_value());

    int32_t write_val = 42;
    auto ms = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(ms.has_value());
    auto fs = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(fs.has_value());

    auto err = ds.value().Write(Datatype::Int32(), ms.value(), fs.value(), &write_val);
    REQUIRE(err.has_value());

    int32_t read_val = 0;
    auto err2 = ds.value().Read(Datatype::Int32(), ms.value(), fs.value(), &read_val);
    REQUIRE(err2.has_value());
    REQUIRE(read_val == 42);
}

TEST_CASE("Edge: 3D dataset with chunking", "[dataset_comprehensive][edge]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());
    auto file = File<InMemoryBlobStore>::Create(
        InMemoryBlobStore(fixture.allocator), Context(fixture.allocator));
    REQUIRE(file.has_value());
    auto root = file.value().OpenRootGroup();

    // dims={4,6,8}, chunks={2,3,4}, total=192 elements
    uint64_t dims[] = {4, 6, 8};
    auto space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 3));
    REQUIRE(space.has_value());

    DatasetCreateProps props;
    props.chunk_dims.push_back(2);
    props.chunk_dims.push_back(3);
    props.chunk_dims.push_back(4);

    auto ds = root.CreateDataset("tensor3d", Datatype::Int32(), space.value(), props);
    REQUIRE(ds.has_value());

    int32_t buf[192];
    for (int i = 0; i < 192; i++) buf[i] = i;

    auto ms = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 3));
    REQUIRE(ms.has_value());
    auto fs = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 3));
    REQUIRE(fs.has_value());

    REQUIRE(ds.value().Write(Datatype::Int32(), ms.value(), fs.value(), buf).has_value());

    int32_t out[192] = {};
    REQUIRE(ds.value().Read(Datatype::Int32(), ms.value(), fs.value(), out).has_value());
    for (int i = 0; i < 192; i++) REQUIRE(out[i] == i);
}

TEST_CASE("Edge: SelectNone write is no-op", "[dataset_comprehensive][edge]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());
    auto file = File<InMemoryBlobStore>::Create(
        InMemoryBlobStore(fixture.allocator), Context(fixture.allocator));
    REQUIRE(file.has_value());
    auto root = file.value().OpenRootGroup();

    uint64_t dims[] = {10};
    auto space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(space.has_value());

    auto ds = root.CreateDataset("data", Datatype::Int32(), space.value());
    REQUIRE(ds.has_value());

    // Write initial data
    int32_t original[10];
    for (int i = 0; i < 10; i++) original[i] = i * 10;

    auto ms = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(ms.has_value());
    auto fs = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(fs.has_value());
    REQUIRE(ds.value().Write(Datatype::Int32(), ms.value(), fs.value(), original).has_value());

    // Attempt write with SelectNone file_space — should be a no-op
    int32_t new_data[10];
    for (int i = 0; i < 10; i++) new_data[i] = 999;

    auto none_fs = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(none_fs.has_value());
    none_fs.value().SelectNone();
    REQUIRE(none_fs.value().GetSelectionType() == SelectionType::None);

    auto ms2 = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(ms2.has_value());
    auto err = ds.value().Write(Datatype::Int32(), ms2.value(), none_fs.value(), new_data);
    REQUIRE(err.has_value());

    // Read back — original data must be unchanged
    int32_t out[10] = {};
    auto read_ms = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(read_ms.has_value());
    auto read_fs = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(read_fs.has_value());
    REQUIRE(ds.value().Read(Datatype::Int32(), read_ms.value(), read_fs.value(), out).has_value());
    for (int i = 0; i < 10; i++) REQUIRE(out[i] == i * 10);
}

TEST_CASE("Edge: read unwritten region returns zeros", "[dataset_comprehensive][edge]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());
    auto file = File<InMemoryBlobStore>::Create(
        InMemoryBlobStore(fixture.allocator), Context(fixture.allocator));
    REQUIRE(file.has_value());
    auto root = file.value().OpenRootGroup();

    // 20 elements, chunk size 10 — two chunks
    uint64_t dims[] = {20};
    auto space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(space.has_value());

    DatasetCreateProps props;
    props.chunk_dims.push_back(10);
    auto ds = root.CreateDataset("partial", Datatype::Int32(), space.value(), props);
    REQUIRE(ds.has_value());

    // Write only chunk 0 (elements 0-9) via hyperslab
    int32_t chunk0_data[10];
    for (int i = 0; i < 10; i++) chunk0_data[i] = i + 1;  // 1..10

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

    REQUIRE(ds.value().Write(Datatype::Int32(), ms.value(), fs.value(), chunk0_data).has_value());

    // Read all 20 elements
    int32_t out[20] = {};
    auto read_ms = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(read_ms.has_value());
    auto read_fs = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(read_fs.has_value());
    REQUIRE(ds.value().Read(Datatype::Int32(), read_ms.value(), read_fs.value(), out).has_value());

    // Elements 0-9 should have written data
    for (int i = 0; i < 10; i++) REQUIRE(out[i] == i + 1);
    // Elements 10-19 (chunk 1, never written) should be zero
    for (int i = 10; i < 20; i++) REQUIRE(out[i] == 0);
}

TEST_CASE("Edge: overwrite then read", "[dataset_comprehensive][edge]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());
    auto file = File<InMemoryBlobStore>::Create(
        InMemoryBlobStore(fixture.allocator), Context(fixture.allocator));
    REQUIRE(file.has_value());
    auto root = file.value().OpenRootGroup();

    uint64_t dims[] = {10};
    auto space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(space.has_value());

    auto ds = root.CreateDataset("overwrite", Datatype::Int32(), space.value());
    REQUIRE(ds.has_value());

    // First write
    int32_t first[10];
    for (int i = 0; i < 10; i++) first[i] = i;

    auto ms1 = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(ms1.has_value());
    auto fs1 = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(fs1.has_value());
    REQUIRE(ds.value().Write(Datatype::Int32(), ms1.value(), fs1.value(), first).has_value());

    // Second write — overwrite all with new values
    int32_t second[10];
    for (int i = 0; i < 10; i++) second[i] = (i + 1) * 100;

    auto ms2 = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(ms2.has_value());
    auto fs2 = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(fs2.has_value());
    REQUIRE(ds.value().Write(Datatype::Int32(), ms2.value(), fs2.value(), second).has_value());

    // Read back — only new values should be present
    int32_t out[10] = {};
    auto read_ms = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(read_ms.has_value());
    auto read_fs = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(read_fs.has_value());
    REQUIRE(ds.value().Read(Datatype::Int32(), read_ms.value(), read_fs.value(), out).has_value());
    for (int i = 0; i < 10; i++) REQUIRE(out[i] == (i + 1) * 100);
}

TEST_CASE("Edge: partial last chunk 2D", "[dataset_comprehensive][edge]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());
    auto file = File<InMemoryBlobStore>::Create(
        InMemoryBlobStore(fixture.allocator), Context(fixture.allocator));
    REQUIRE(file.has_value());
    auto root = file.value().OpenRootGroup();

    // 5x7 dataset, chunks 3x3 — partial chunks on both dimensions
    // Chunk grid: ceil(5/3)=2 x ceil(7/3)=3 => 6 chunks total
    // Last row of chunks has 2 rows (5 mod 3=2), last col of chunks has 1 col (7 mod 3=1)
    uint64_t dims[] = {5, 7};
    auto space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 2));
    REQUIRE(space.has_value());

    DatasetCreateProps props;
    props.chunk_dims.push_back(3);
    props.chunk_dims.push_back(3);
    auto ds = root.CreateDataset("partial2d", Datatype::Int32(), space.value(), props);
    REQUIRE(ds.has_value());

    // Write all 35 values (value = flat index)
    int32_t buf[35];
    for (int i = 0; i < 35; i++) buf[i] = i;

    auto ms = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 2));
    REQUIRE(ms.has_value());
    auto fs = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 2));
    REQUIRE(fs.has_value());
    REQUIRE(ds.value().Write(Datatype::Int32(), ms.value(), fs.value(), buf).has_value());

    // Read back all 35 values
    int32_t out[35] = {};
    auto read_ms = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 2));
    REQUIRE(read_ms.has_value());
    auto read_fs = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 2));
    REQUIRE(read_fs.has_value());
    REQUIRE(ds.value().Read(Datatype::Int32(), read_ms.value(), read_fs.value(), out).has_value());
    for (int i = 0; i < 35; i++) REQUIRE(out[i] == i);
}
