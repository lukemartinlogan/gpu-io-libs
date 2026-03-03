#include <catch2/catch_test_macros.hpp>
#include <kvhdf5/hdf5_dataset.h>
#include <kvhdf5/memory_blob_store.h>
#include "../common/allocator_fixture.h"

using namespace kvhdf5;
using AllocatorFixture = test::AllocatorFixture<AllocatorImpl>;

// Large fixture for stress tests that store many/large chunks in the blob store.
// Each InMemoryBlobStore entry is backed by the fixture's allocator heap, so
// tests with high total blob-data volume need a proportionally larger heap.
struct LargeAllocatorFixture {
    // 10,000 chunks * ~400 bytes data + per-entry overhead in InMemoryBlobStore
    // (vector headers, key storage, allocator fragmentation from growth).
    static constexpr size_t kHeapSize = 64 * 1024 * 1024;  // 64 MB
    char* memory = nullptr;
    hshm::ipc::ArrayBackend backend;
    AllocatorImpl* allocator = nullptr;

    LargeAllocatorFixture() {
        size_t alloc_size = kHeapSize + 3 * hshm::ipc::kBackendHeaderSize;
        memory = new char[alloc_size];
        cuda::std::memset(memory, 0, alloc_size);
        if (backend.shm_init(hshm::ipc::MemoryBackendId::GetRoot(), alloc_size, memory)) {
            allocator = backend.MakeAlloc<AllocatorImpl>();
        }
    }

    ~LargeAllocatorFixture() {
        if (memory) {
            delete[] memory;
        }
    }

    bool IsValid() const { return allocator != nullptr; }
};

TEST_CASE("Stress: 100x100 chunk grid", "[dataset_comprehensive][stress]") {
    // 1000x1000 dataset, 10x10 chunks => 100x100 = 10,000 chunks.
    // Total elements: 1,000,000 int32 values (4 MB data).
    // Requires a large allocator heap since InMemoryBlobStore stores all blob
    // data inside the fixture allocator.
    LargeAllocatorFixture fixture;
    REQUIRE(fixture.IsValid());

    auto file = File<InMemoryBlobStore>::Create(
        InMemoryBlobStore(fixture.allocator), Context(fixture.allocator));
    REQUIRE(file.has_value());
    auto root = file.value().OpenRootGroup();

    uint64_t dims[] = {1000, 1000};
    auto space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 2));
    REQUIRE(space.has_value());

    DatasetCreateProps props;
    props.chunk_dims.push_back(10);
    props.chunk_dims.push_back(10);

    auto ds = root.CreateDataset("grid1000", Datatype::Int32(), space.value(), props);
    REQUIRE(ds.has_value());

    // Allocate write and read buffers on the heap (too large for stack).
    const int kTotal = 1000 * 1000;
    int32_t* write_buf = new int32_t[kTotal];
    int32_t* read_buf  = new int32_t[kTotal];

    for (int i = 0; i < kTotal; ++i) write_buf[i] = i;

    // Write all 1,000,000 elements via SelectAll.
    auto ms = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 2));
    REQUIRE(ms.has_value());
    auto fs = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 2));
    REQUIRE(fs.has_value());
    REQUIRE(ds.value().Write(Datatype::Int32(), ms.value(), fs.value(), write_buf).has_value());

    // Read all back.
    auto ms2 = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 2));
    REQUIRE(ms2.has_value());
    auto fs2 = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 2));
    REQUIRE(fs2.has_value());
    REQUIRE(ds.value().Read(Datatype::Int32(), ms2.value(), fs2.value(), read_buf).has_value());

    bool all_correct = true;
    for (int i = 0; i < kTotal; ++i) {
        if (read_buf[i] != write_buf[i]) {
            all_correct = false;
            break;
        }
    }
    REQUIRE(all_correct);

    delete[] write_buf;
    delete[] read_buf;
}

TEST_CASE("Stress: large contiguous write across many chunks", "[dataset_comprehensive][stress]") {
    // 10000 elements 1D, chunk=100 => 100 chunks.
    // Total blob data: 100 * 100 * 4 = 40,000 bytes, but InMemoryBlobStore
    // overhead per entry exceeds the default 64 KB heap.
    LargeAllocatorFixture fixture;
    REQUIRE(fixture.IsValid());

    auto file = File<InMemoryBlobStore>::Create(
        InMemoryBlobStore(fixture.allocator), Context(fixture.allocator));
    REQUIRE(file.has_value());
    auto root = file.value().OpenRootGroup();

    uint64_t dims[] = {10000};
    auto space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(space.has_value());

    DatasetCreateProps props;
    props.chunk_dims.push_back(100);

    auto ds = root.CreateDataset("large1d", Datatype::Int32(), space.value(), props);
    REQUIRE(ds.has_value());

    // Heap-allocate buffers: 10000 * 4 = 40 KB each, safe on stack but use heap
    // for clarity and to avoid accidental stack overflow on constrained platforms.
    const int kTotal = 10000;
    int32_t* write_buf = new int32_t[kTotal];
    int32_t* read_buf  = new int32_t[kTotal];

    for (int i = 0; i < kTotal; ++i) write_buf[i] = i * 3;

    auto ms = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(ms.has_value());
    auto fs = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(fs.has_value());
    REQUIRE(ds.value().Write(Datatype::Int32(), ms.value(), fs.value(), write_buf).has_value());

    auto ms2 = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(ms2.has_value());
    auto fs2 = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(fs2.has_value());
    REQUIRE(ds.value().Read(Datatype::Int32(), ms2.value(), fs2.value(), read_buf).has_value());

    bool all_correct = true;
    for (int i = 0; i < kTotal; ++i) {
        if (read_buf[i] != write_buf[i]) {
            all_correct = false;
            break;
        }
    }
    REQUIRE(all_correct);

    delete[] write_buf;
    delete[] read_buf;
}

TEST_CASE("Stress: random-order chunk writes", "[dataset_comprehensive][stress]") {
    // 100 elements 1D, chunk=10 => 10 chunks.
    // Write chunks in reverse order (chunk 9, 8, ..., 0) using hyperslabs.
    // Read all back and verify correct flat ordering.
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

    auto ds = root.CreateDataset("reverse_chunks", Datatype::Int32(), space.value(), props);
    REQUIRE(ds.has_value());

    // Write each chunk individually via hyperslab, in reverse chunk order
    // (chunk 9 first, then 8, ..., then 0).
    const int kChunkSize = 10;
    const int kNumChunks = 10;
    int32_t chunk_data[kChunkSize];

    for (int chunk = kNumChunks - 1; chunk >= 0; --chunk) {
        uint64_t chunk_start = static_cast<uint64_t>(chunk * kChunkSize);

        // Fill the chunk buffer with values that match the expected flat indices.
        for (int j = 0; j < kChunkSize; ++j) {
            chunk_data[j] = static_cast<int32_t>(chunk_start + j);
        }

        // Memory space: kChunkSize elements.
        uint64_t mem_dims[] = {static_cast<uint64_t>(kChunkSize)};
        auto ms = Dataspace::CreateSimple(cstd::span<const uint64_t>(mem_dims, 1));
        REQUIRE(ms.has_value());

        // File space: full dataset shape, select the chunk's hyperslab.
        auto fs = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
        REQUIRE(fs.has_value());

        uint64_t start[]  = {chunk_start};
        uint64_t stride[] = {1};
        uint64_t count[]  = {static_cast<uint64_t>(kChunkSize)};
        uint64_t block[]  = {1};
        REQUIRE(fs.value().SelectHyperslab(
            SelectionOp::Set,
            cstd::span<const uint64_t>(start,  1),
            cstd::span<const uint64_t>(stride, 1),
            cstd::span<const uint64_t>(count,  1),
            cstd::span<const uint64_t>(block,  1)).has_value());

        REQUIRE(ds.value().Write(Datatype::Int32(), ms.value(), fs.value(), chunk_data).has_value());
    }

    // Read all 100 elements back via SelectAll and verify.
    int32_t read_buf[100] = {};
    auto ms_all = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(ms_all.has_value());
    auto fs_all = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(fs_all.has_value());
    REQUIRE(ds.value().Read(Datatype::Int32(), ms_all.value(), fs_all.value(), read_buf).has_value());

    for (int i = 0; i < 100; ++i) {
        REQUIRE(read_buf[i] == i);
    }
}
