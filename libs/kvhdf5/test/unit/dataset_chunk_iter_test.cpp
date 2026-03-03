#include <catch2/catch_test_macros.hpp>
#include <kvhdf5/hdf5_dataset.h>
#include <kvhdf5/memory_blob_store.h>
#include "../common/allocator_fixture.h"

using namespace kvhdf5;
using AllocatorFixture = test::AllocatorFixture<AllocatorImpl>;

TEST_CASE("ChunkIter: visits all written chunks", "[dataset_comprehensive][chunk_iter]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());
    auto file = File<InMemoryBlobStore>::Create(
        InMemoryBlobStore(fixture.allocator), Context(fixture.allocator));
    REQUIRE(file.has_value());
    auto root = file.value().OpenRootGroup();

    // 90 elements, chunk=30 => 3 chunks (all equal size)
    uint64_t dims[] = {90};
    auto space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(space.has_value());
    DatasetCreateProps props;
    props.chunk_dims.push_back(30);
    auto ds = root.CreateDataset("all_chunks", Datatype::Int32(), space.value(), props);
    REQUIRE(ds.has_value());

    // Write all 90 elements
    int32_t buf[90];
    for (int i = 0; i < 90; i++) buf[i] = i;
    auto ms = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(ms.has_value());
    auto fs = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(fs.has_value());
    REQUIRE(ds.value().Write(Datatype::Int32(), ms.value(), fs.value(), buf).has_value());

    // ChunkIter should visit all 3 chunks
    int count = 0;
    auto cb = [](const ChunkKey& key, uint64_t size, void* data) -> bool {
        (void)key; (void)size;
        (*static_cast<int*>(data))++;
        return true;
    };
    auto result = ds.value().ChunkIter(cb, &count);
    REQUIRE(result.has_value());
    REQUIRE(count == 3);
}

TEST_CASE("ChunkIter: skips unwritten chunks", "[dataset_comprehensive][chunk_iter]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());
    auto file = File<InMemoryBlobStore>::Create(
        InMemoryBlobStore(fixture.allocator), Context(fixture.allocator));
    REQUIRE(file.has_value());
    auto root = file.value().OpenRootGroup();

    // 90 elements, chunk=30 => chunks at indices [0], [1], [2]
    // Write only chunk 0 (elements 0-29) and chunk 2 (elements 60-89); skip chunk 1 (30-59)
    uint64_t dims[] = {90};
    auto space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(space.has_value());
    DatasetCreateProps props;
    props.chunk_dims.push_back(30);
    auto ds = root.CreateDataset("partial_chunks", Datatype::Int32(), space.value(), props);
    REQUIRE(ds.has_value());

    int32_t buf[30];
    for (int i = 0; i < 30; i++) buf[i] = i;

    // Write chunk 0: elements 0-29 via hyperslab
    {
        uint64_t mem_dims[] = {30};
        auto ms = Dataspace::CreateSimple(cstd::span<const uint64_t>(mem_dims, 1));
        REQUIRE(ms.has_value());
        auto fs = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
        REQUIRE(fs.has_value());
        uint64_t start[] = {0}, stride[] = {1}, count[] = {30}, block[] = {1};
        REQUIRE(fs.value().SelectHyperslab(SelectionOp::Set,
            cstd::span<const uint64_t>(start, 1),
            cstd::span<const uint64_t>(stride, 1),
            cstd::span<const uint64_t>(count, 1),
            cstd::span<const uint64_t>(block, 1)).has_value());
        REQUIRE(ds.value().Write(Datatype::Int32(), ms.value(), fs.value(), buf).has_value());
    }

    // Write chunk 2: elements 60-89 via hyperslab
    {
        uint64_t mem_dims[] = {30};
        auto ms = Dataspace::CreateSimple(cstd::span<const uint64_t>(mem_dims, 1));
        REQUIRE(ms.has_value());
        auto fs = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
        REQUIRE(fs.has_value());
        uint64_t start[] = {60}, stride[] = {1}, count[] = {30}, block[] = {1};
        REQUIRE(fs.value().SelectHyperslab(SelectionOp::Set,
            cstd::span<const uint64_t>(start, 1),
            cstd::span<const uint64_t>(stride, 1),
            cstd::span<const uint64_t>(count, 1),
            cstd::span<const uint64_t>(block, 1)).has_value());
        REQUIRE(ds.value().Write(Datatype::Int32(), ms.value(), fs.value(), buf).has_value());
    }

    // ChunkIter should visit 2 chunks (chunk 1 was never written)
    int count = 0;
    auto cb = [](const ChunkKey& key, uint64_t size, void* data) -> bool {
        (void)key; (void)size;
        (*static_cast<int*>(data))++;
        return true;
    };
    auto result = ds.value().ChunkIter(cb, &count);
    REQUIRE(result.has_value());
    REQUIRE(count == 2);
}

TEST_CASE("ChunkIter: early termination", "[dataset_comprehensive][chunk_iter]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());
    auto file = File<InMemoryBlobStore>::Create(
        InMemoryBlobStore(fixture.allocator), Context(fixture.allocator));
    REQUIRE(file.has_value());
    auto root = file.value().OpenRootGroup();

    // 100 elements, chunk=10 => 10 chunks
    uint64_t dims[] = {100};
    auto space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(space.has_value());
    DatasetCreateProps props;
    props.chunk_dims.push_back(10);
    auto ds = root.CreateDataset("many_chunks", Datatype::Int32(), space.value(), props);
    REQUIRE(ds.has_value());

    // Write all 100 elements
    int32_t buf[100];
    for (int i = 0; i < 100; i++) buf[i] = i;
    auto ms = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(ms.has_value());
    auto fs = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(fs.has_value());
    REQUIRE(ds.value().Write(Datatype::Int32(), ms.value(), fs.value(), buf).has_value());

    // Callback returns false after visiting 4 chunks, stopping iteration early
    int count = 0;
    auto cb = [](const ChunkKey& key, uint64_t size, void* data) -> bool {
        (void)key; (void)size;
        int& c = *static_cast<int*>(data);
        c++;
        return c < 4;  // continue while count < 4; stop when count reaches 4
    };
    auto result = ds.value().ChunkIter(cb, &count);
    REQUIRE(result.has_value());
    REQUIRE(count == 4);
}

TEST_CASE("ChunkIter: reports correct chunk sizes", "[dataset_comprehensive][chunk_iter]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());
    auto file = File<InMemoryBlobStore>::Create(
        InMemoryBlobStore(fixture.allocator), Context(fixture.allocator));
    REQUIRE(file.has_value());
    auto root = file.value().OpenRootGroup();

    // 25 elements, chunk=10, Int32 (4 bytes)
    // => 3 chunks: [0] has 10 elems = 40 bytes, [1] has 10 elems = 40 bytes, [2] has 5 elems = 20 bytes
    uint64_t dims[] = {25};
    auto space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(space.has_value());
    DatasetCreateProps props;
    props.chunk_dims.push_back(10);
    auto ds = root.CreateDataset("sized_chunks", Datatype::Int32(), space.value(), props);
    REQUIRE(ds.has_value());

    // Write all 25 elements
    int32_t buf[25];
    for (int i = 0; i < 25; i++) buf[i] = i;
    auto ms = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(ms.has_value());
    auto fs = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(fs.has_value());
    REQUIRE(ds.value().Write(Datatype::Int32(), ms.value(), fs.value(), buf).has_value());

    // Collect sizes reported by ChunkIter
    struct Data { uint64_t sizes[3]; int count; };
    Data user_data = {{0, 0, 0}, 0};

    auto cb = [](const ChunkKey& key, uint64_t size, void* data) -> bool {
        (void)key;
        Data& d = *static_cast<Data*>(data);
        if (d.count < 3) {
            d.sizes[d.count] = size;
        }
        d.count++;
        return true;
    };
    auto result = ds.value().ChunkIter(cb, &user_data);
    REQUIRE(result.has_value());
    REQUIRE(user_data.count == 3);

    // First two chunks: 10 Int32 elements = 10 * 4 = 40 bytes each
    REQUIRE(user_data.sizes[0] == 40);
    REQUIRE(user_data.sizes[1] == 40);
    // Last chunk: 5 Int32 elements = 5 * 4 = 20 bytes
    REQUIRE(user_data.sizes[2] == 20);
}
