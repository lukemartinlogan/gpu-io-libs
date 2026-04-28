#if HSHM_ENABLE_CUDA

#include <catch2/catch_test_macros.hpp>
#include "gpu_cte_fixture.h"
#include "kvhdf5/gpu_cte_blob_store.h"
#include "kvhdf5/container.h"
#include "kvhdf5/types/chunk_key.h"
#include "hermes_shm/memory/backend/array_backend.h"
#include "gpu_container_helpers.h"
#include <cstring>
#include <thread>
#include <chrono>

using namespace kvhdf5;

// ---------------------------------------------------------------------------
// ContainerTestResult — pinned result struct for polling from host.
//
// status:    1 = pass, negative = step-specific error, 0 = not yet done.
// data_match: 1 = payload matched (used by kernel D).
// ---------------------------------------------------------------------------

struct ContainerTestResult {
    int status;
    int data_match;
};

// ===========================================================================
// KERNELS
//
// DESIGN NOTE — clang-18 NVPTX codegen bug (isspacep.shared):
//
//   hshm::priv::vector<T> uses a Small Vector Optimization (SVO) with an
//   inline char svo_[] buffer.  IsUsingSvo() at line 589 of vector.h compares
//   data_.ptr_ == svo_data(), where svo_data() is a typed reinterpret_cast<T*>.
//   clang-18's NVPTX backend emits llvm.nvvm.isspacep.shared with that typed
//   pointer instead of the expected i8*, which ptxas 12.x rejects:
//     "Call parameter type does not match function signature /
//      Broken function found, compilation aborted!"
//
//   The bug fires at COMPILE TIME whenever clang-18 device-instantiates any
//   CROSS_FUN (__host__ __device__) function that touches hshm::priv::vector<T>:
//     - GroupMetadata / DatasetMetadata constructors (both CROSS_FUN)
//     - GroupMetadata::Serialize / DatasetMetadata::Serialize (iterate vectors)
//     - GroupMetadata::Deserialize / DatasetMetadata::Deserialize (local vectors)
//     - Container::PutGroup / Container::PutDataset (call Serialize)
//     - Container::GetGroup / Container::GetDataset (call Deserialize)
//
//   Device-instantiation happens when ANY of these CROSS_FUN functions is
//   called from a .cu translation unit — even in #if !HSHM_IS_GPU host-only
//   blocks.  clang-18 does ODR-based eager device instantiation.
//
//   FIX STRATEGY:
//     1. No GroupMetadata / DatasetMetadata construction in this .cu file.
//        All such construction and all PutGroup / PutDataset calls are in
//        gpu_container_helpers.cc (a plain CXX translation unit that nvcc
//        never device-compiles).
//     2. Device kernels use only: AllocateId, GroupExists, DatasetExists,
//        DeleteGroup, RootGroup, PutChunk, GetChunk.  None of these paths
//        touch hshm::priv::vector<T>.
//     3. Host TEST_CASE code does "put" via kvhdf5::test::HostPutGroup /
//        HostPutDataset from the helper TU, then launches a kernel that
//        verifies via GroupExists / DatasetExists.
// ===========================================================================

// ---------------------------------------------------------------------------
// Kernel A — AllocateId returns distinct, monotonically increasing IDs.
//
// AllocateId touches only cstd::atomic<uint64_t>, no vectors.
// ObjectId::operator< is constexpr __host__ (from <=>), so raw .id fields
// are compared directly.
// ---------------------------------------------------------------------------

__global__ void kernel_allocate_id(
    chi::IpcManagerGpuInfo gpu_info,
    Container<GpuCteBlobStore>* container,
    ContainerTestResult* d_result)
{
    d_result->status    = 0;
    d_result->data_match = 0;

    CHIMAERA_GPU_INIT(gpu_info);
    if (threadIdx.x != 0) return;

    ObjectId id1 = container->AllocateId();
    ObjectId id2 = container->AllocateId();
    ObjectId id3 = container->AllocateId();

    if (!id1.IsValid()) { d_result->status = -1; return; }
    if (!id2.IsValid()) { d_result->status = -2; return; }
    if (!id3.IsValid()) { d_result->status = -3; return; }

    if (id1.id == id2.id) { d_result->status = -4; return; }
    if (id2.id == id3.id) { d_result->status = -5; return; }
    if (id1.id == id3.id) { d_result->status = -6; return; }

    if (!(id1.id < id2.id)) { d_result->status = -7; return; }
    if (!(id2.id < id3.id)) { d_result->status = -8; return; }

    d_result->data_match = 1;
    d_result->status     = 1;
}

// ---------------------------------------------------------------------------
// Kernel B — GroupExists after a host-side PutGroup.
//
// The host TEST_CASE calls HostPutGroup (in gpu_container_helpers.cc — pure
// CXX, no device compilation) before launching this kernel.  The kernel only
// calls GroupExists, which is a key-index lookup with no vector
// deserialisation.
// ---------------------------------------------------------------------------

__global__ void kernel_group_exists(
    chi::IpcManagerGpuInfo gpu_info,
    Container<GpuCteBlobStore>* container,
    GroupId gid,
    ContainerTestResult* d_result)
{
    d_result->status    = 0;
    d_result->data_match = 0;

    CHIMAERA_GPU_INIT(gpu_info);
    if (threadIdx.x != 0) return;

    if (!container->GroupExists(gid)) { d_result->status = -1; return; }

    d_result->data_match = 1;
    d_result->status     = 1;
}

// ---------------------------------------------------------------------------
// Kernel C — DatasetExists after a host-side PutDataset.
//
// Same rationale as kernel_group_exists.  DatasetExists is a key-index lookup.
// ---------------------------------------------------------------------------

__global__ void kernel_dataset_exists(
    chi::IpcManagerGpuInfo gpu_info,
    Container<GpuCteBlobStore>* container,
    DatasetId did,
    ContainerTestResult* d_result)
{
    d_result->status    = 0;
    d_result->data_match = 0;

    CHIMAERA_GPU_INIT(gpu_info);
    if (threadIdx.x != 0) return;

    if (!container->DatasetExists(did)) { d_result->status = -1; return; }

    d_result->data_match = 1;
    d_result->status     = 1;
}

// ---------------------------------------------------------------------------
// Kernel D — PutChunk / GetChunk with ChunkKey (fully on device).
//
// Raw byte blob — no GroupMetadata / DatasetMetadata, no hshm::priv::vector
// involved.  The isspacep.shared codegen bug is not triggered.
// ---------------------------------------------------------------------------

__global__ void kernel_chunk_roundtrip(
    chi::IpcManagerGpuInfo gpu_info,
    Container<GpuCteBlobStore>* container,
    ContainerTestResult* d_result)
{
    d_result->status    = 0;
    d_result->data_match = 0;

    CHIMAERA_GPU_INIT(gpu_info);
    if (threadIdx.x != 0) return;

    DatasetId did(uint64_t{42});
    cstd::array<uint64_t, 2> coords_arr = {uint64_t{3}, uint64_t{5}};
    cstd::span<const uint64_t> coords_span(coords_arr.data(), 2);
    ChunkKey key(did, coords_span);

    cstd::array<byte_t, 16> payload = {
        byte_t{0x01}, byte_t{0x02}, byte_t{0x03}, byte_t{0x04},
        byte_t{0x05}, byte_t{0x06}, byte_t{0x07}, byte_t{0x08},
        byte_t{0x09}, byte_t{0x0A}, byte_t{0x0B}, byte_t{0x0C},
        byte_t{0x0D}, byte_t{0x0E}, byte_t{0x0F}, byte_t{0x10}
    };

    bool put_ok = container->PutChunk(
        key, cstd::span<const byte_t>(payload.data(), payload.size()));
    if (!put_ok) { d_result->status = -1; return; }

    cstd::array<byte_t, 16> out_buf;
    auto result = container->GetChunk(
        key, cstd::span<byte_t>(out_buf.data(), out_buf.size()));
    if (!result.has_value()) { d_result->status = -2; return; }
    if (result->size() != 16) { d_result->status = -3; return; }

    d_result->data_match = 1;
    for (size_t i = 0; i < 16; ++i) {
        if (out_buf[i] != payload[i]) {
            d_result->data_match = 0;
            d_result->status     = -4;
            return;
        }
    }

    d_result->status = 1;
}

// ---------------------------------------------------------------------------
// Kernel E — GroupExists / DeleteGroup / GroupExists on device.
//
// The host TEST_CASE calls HostPutGroup before launching this kernel.
// Kernel flow: GroupExists(pre-delete) → DeleteGroup → GroupExists(post-delete).
// GroupExists and DeleteGroup are pure key-index ops, no vector involved.
// ---------------------------------------------------------------------------

__global__ void kernel_exists_delete(
    chi::IpcManagerGpuInfo gpu_info,
    Container<GpuCteBlobStore>* container,
    GroupId gid,
    ContainerTestResult* d_result)
{
    d_result->status    = 0;
    d_result->data_match = 0;

    CHIMAERA_GPU_INIT(gpu_info);
    if (threadIdx.x != 0) return;

    // 1. Group must exist (was put by host before kernel launch)
    if (!container->GroupExists(gid))  { d_result->status = -1; return; }

    // 2. DeleteGroup
    bool del_ok = container->DeleteGroup(gid);
    if (!del_ok)                        { d_result->status = -2; return; }

    // 3. Group must not exist after Delete
    if (container->GroupExists(gid))   { d_result->status = -3; return; }

    d_result->data_match = 1;
    d_result->status     = 1;
}

// ---------------------------------------------------------------------------
// Kernel F — RootGroup() + GroupExists() on device.
//
// RootGroup() is a simple field accessor.
// GroupExists() is a key-index lookup.
// Neither touches hshm::priv::vector<T>.
// ---------------------------------------------------------------------------

__global__ void kernel_root_group(
    chi::IpcManagerGpuInfo gpu_info,
    Container<GpuCteBlobStore>* container,
    ContainerTestResult* d_result)
{
    d_result->status    = 0;
    d_result->data_match = 0;

    CHIMAERA_GPU_INIT(gpu_info);
    if (threadIdx.x != 0) return;

    GroupId root_gid = container->RootGroup();

    if (!root_gid.IsValid())                { d_result->status = -1; return; }
    if (!container->GroupExists(root_gid))  { d_result->status = -2; return; }

    d_result->data_match = 1;
    d_result->status     = 1;
}

// ---------------------------------------------------------------------------
// Host-only section: setup helpers, run helpers, and TEST_CASEs.
// ---------------------------------------------------------------------------

#if !HSHM_IS_GPU

// ---------------------------------------------------------------------------
// ManagedAllocFixture
//
// Allocates a BuddyAllocator heap in cudaMallocManaged so it is accessible
// from both host and device.
//
// WHY: AllocatorFixture (common test header) uses `new char[]` (host only).
// hshm::priv::vector uses the allocator's managed heap for its backing store
// when the SVO overflows — that path must be reachable from device code.
// ---------------------------------------------------------------------------

struct ManagedAllocFixture {
    static constexpr size_t kHeapSize = 128 * 1024;

    char*                    memory    = nullptr;
    hshm::ipc::ArrayBackend  backend;
    AllocatorImpl*           allocator = nullptr;

    bool Setup() {
        size_t alloc_size = kHeapSize + 3 * hshm::ipc::kBackendHeaderSize;

        auto* gpu_ipc = CHI_CPU_IPC->GetGpuIpcManager();
        gpu_ipc->PauseGpuOrchestrator();
        cudaError_t err = cudaMallocManaged(
            reinterpret_cast<void**>(&memory), alloc_size);
        gpu_ipc->ResumeGpuOrchestrator();
        if (err != cudaSuccess) return false;

        memset(memory, 0, alloc_size);

        if (!backend.shm_init(hshm::ipc::MemoryBackendId::GetRoot(),
                              alloc_size, memory)) {
            return false;
        }

        allocator = backend.MakeAlloc<AllocatorImpl>();
        return allocator != nullptr;
    }

    void Teardown() {
        if (memory) {
            auto* gpu_ipc = CHI_CPU_IPC->GetGpuIpcManager();
            gpu_ipc->PauseGpuOrchestrator();
            cudaFree(memory);
            gpu_ipc->ResumeGpuOrchestrator();
            memory    = nullptr;
            allocator = nullptr;
        }
    }
};

// ---------------------------------------------------------------------------
// ManagedContainerBox
//
// Places Container<GpuCteBlobStore> in cudaMallocManaged so the UVA address
// is identical on host and device.
//
// WHY: Container<B> holds BlobStoreImpl raw_store_ and BlobStore<B> store_
// (which holds BlobStoreImpl* → &raw_store_).  CUDA parameter passing is a
// shallow memcpy — passing Container by value would make store_.store_ point
// at the host copy of raw_store_, causing device faults.  Passing a pointer
// to a managed allocation keeps the address stable on both sides.
// ---------------------------------------------------------------------------

struct ManagedContainerBox {
    using ContainerT = Container<GpuCteBlobStore>;

    ContainerT* ptr = nullptr;

    bool Setup(GpuCteBlobStore blob_store, AllocatorImpl* alloc) {
        void* raw = nullptr;
        auto* gpu_ipc = CHI_CPU_IPC->GetGpuIpcManager();
        gpu_ipc->PauseGpuOrchestrator();
        cudaError_t err = cudaMallocManaged(&raw, sizeof(ContainerT));
        gpu_ipc->ResumeGpuOrchestrator();
        if (err != cudaSuccess) return false;

        // Placement-new: ctor calls AllocateId + PutGroup for root group
        // on the host side (CPU CTE path).
        ptr = new (raw) ContainerT(std::move(blob_store), alloc);
        return ptr != nullptr;
    }

    void Teardown() {
        if (ptr) {
            ptr->~ContainerT();
            auto* gpu_ipc = CHI_CPU_IPC->GetGpuIpcManager();
            gpu_ipc->PauseGpuOrchestrator();
            cudaFree(ptr);
            gpu_ipc->ResumeGpuOrchestrator();
            ptr = nullptr;
        }
    }
};

// ---------------------------------------------------------------------------
// RunContainerTest — SimpleKernel launcher.
//
// Mirrors RunTypedBlobTest from gpu_blob_store_typed_test.cu.
// ---------------------------------------------------------------------------

using SimpleKernel = void (*)(chi::IpcManagerGpuInfo,
                              Container<GpuCteBlobStore>*,
                              ContainerTestResult*);

static ContainerTestResult RunContainerTest(
    SimpleKernel kernel,
    Container<GpuCteBlobStore>* container_ptr)
{
    auto* gpu_ipc   = CHI_CPU_IPC->GetGpuIpcManager();
    chi::IpcManagerGpuInfo gpu_info = gpu_ipc->GetClientGpuInfo(0);

    gpu_ipc->PauseGpuOrchestrator();

    volatile ContainerTestResult* d_result;
    cudaMallocHost(const_cast<ContainerTestResult**>(&d_result),
                   sizeof(ContainerTestResult));
    d_result->status     = 0;
    d_result->data_match = 0;

    cudaGetLastError();
    void* stream = hshm::GpuApi::CreateStream();
    kernel<<<1, 32, 0, static_cast<cudaStream_t>(stream)>>>(
        gpu_info, container_ptr, const_cast<ContainerTestResult*>(d_result));

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        gpu_ipc->ResumeGpuOrchestrator();
        hshm::GpuApi::DestroyStream(stream);
        return {-201, 0};
    }

    gpu_ipc->ResumeGpuOrchestrator();

    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (d_result->status == 0 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }

    ContainerTestResult result{d_result->status, d_result->data_match};
    if (result.status == 0) result.status = -300;
    // cudaFreeHost performs device-wide synchronization which deadlocks against
    // the persistent GPU orchestrator kernel. Pause around the host cleanup.
    gpu_ipc->PauseGpuOrchestrator();
    cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
    hshm::GpuApi::DestroyStream(stream);
    cudaFreeHost(const_cast<ContainerTestResult*>(d_result));
    gpu_ipc->ResumeGpuOrchestrator();
    return result;
}

// ---------------------------------------------------------------------------
// RunGroupExistsKernel — launches kernel_group_exists or kernel_exists_delete.
// ---------------------------------------------------------------------------

using GroupIdKernel = void (*)(chi::IpcManagerGpuInfo,
                               Container<GpuCteBlobStore>*,
                               GroupId,
                               ContainerTestResult*);

static ContainerTestResult RunGroupIdKernel(
    GroupIdKernel kernel,
    Container<GpuCteBlobStore>* container_ptr,
    GroupId gid)
{
    auto* gpu_ipc   = CHI_CPU_IPC->GetGpuIpcManager();
    chi::IpcManagerGpuInfo gpu_info = gpu_ipc->GetClientGpuInfo(0);

    gpu_ipc->PauseGpuOrchestrator();

    volatile ContainerTestResult* d_result;
    cudaMallocHost(const_cast<ContainerTestResult**>(&d_result),
                   sizeof(ContainerTestResult));
    d_result->status     = 0;
    d_result->data_match = 0;

    cudaGetLastError();
    void* stream = hshm::GpuApi::CreateStream();
    kernel<<<1, 32, 0, static_cast<cudaStream_t>(stream)>>>(
        gpu_info, container_ptr, gid,
        const_cast<ContainerTestResult*>(d_result));

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        gpu_ipc->ResumeGpuOrchestrator();
        hshm::GpuApi::DestroyStream(stream);
        return {-201, 0};
    }

    gpu_ipc->ResumeGpuOrchestrator();

    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (d_result->status == 0 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }

    ContainerTestResult result{d_result->status, d_result->data_match};
    if (result.status == 0) result.status = -300;
    // cudaFreeHost performs device-wide synchronization which deadlocks against
    // the persistent GPU orchestrator kernel. Pause around the host cleanup.
    gpu_ipc->PauseGpuOrchestrator();
    cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
    hshm::GpuApi::DestroyStream(stream);
    cudaFreeHost(const_cast<ContainerTestResult*>(d_result));
    gpu_ipc->ResumeGpuOrchestrator();
    return result;
}

// ---------------------------------------------------------------------------
// RunDatasetIdKernel — launches kernel_dataset_exists.
// ---------------------------------------------------------------------------

using DatasetIdKernel = void (*)(chi::IpcManagerGpuInfo,
                                 Container<GpuCteBlobStore>*,
                                 DatasetId,
                                 ContainerTestResult*);

static ContainerTestResult RunDatasetIdKernel(
    DatasetIdKernel kernel,
    Container<GpuCteBlobStore>* container_ptr,
    DatasetId did)
{
    auto* gpu_ipc   = CHI_CPU_IPC->GetGpuIpcManager();
    chi::IpcManagerGpuInfo gpu_info = gpu_ipc->GetClientGpuInfo(0);

    gpu_ipc->PauseGpuOrchestrator();

    volatile ContainerTestResult* d_result;
    cudaMallocHost(const_cast<ContainerTestResult**>(&d_result),
                   sizeof(ContainerTestResult));
    d_result->status     = 0;
    d_result->data_match = 0;

    cudaGetLastError();
    void* stream = hshm::GpuApi::CreateStream();
    kernel<<<1, 32, 0, static_cast<cudaStream_t>(stream)>>>(
        gpu_info, container_ptr, did,
        const_cast<ContainerTestResult*>(d_result));

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        gpu_ipc->ResumeGpuOrchestrator();
        hshm::GpuApi::DestroyStream(stream);
        return {-201, 0};
    }

    gpu_ipc->ResumeGpuOrchestrator();

    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (d_result->status == 0 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }

    ContainerTestResult result{d_result->status, d_result->data_match};
    if (result.status == 0) result.status = -300;
    // cudaFreeHost performs device-wide synchronization which deadlocks against
    // the persistent GPU orchestrator kernel. Pause around the host cleanup.
    gpu_ipc->PauseGpuOrchestrator();
    cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
    hshm::GpuApi::DestroyStream(stream);
    cudaFreeHost(const_cast<ContainerTestResult*>(d_result));
    gpu_ipc->ResumeGpuOrchestrator();
    return result;
}

// ---------------------------------------------------------------------------
// Per-test fixture.
// ---------------------------------------------------------------------------

struct ContainerTestFixture {
    ManagedAllocFixture  alloc_fixture;
    ManagedContainerBox  container_box;

    bool Setup(const char* tag_name) {
        auto tag_task = g_gpu_cte_client->AsyncGetOrCreateTag(tag_name);
        tag_task.Wait();
        wrp_cte::core::TagId tag_id = tag_task->tag_id_;

        GpuCteBlobStore store = GpuCteBlobStore::Create(tag_id, g_gpu_cte_pool_id);
        if (!store.IsValid()) return false;

        if (!alloc_fixture.Setup()) return false;

        if (!container_box.Setup(std::move(store), alloc_fixture.allocator))
            return false;

        return true;
    }

    void Teardown() {
        container_box.Teardown();
        alloc_fixture.Teardown();
    }

    Container<GpuCteBlobStore>* ContainerPtr() { return container_box.ptr; }
    AllocatorImpl&              Allocator()     { return *alloc_fixture.allocator; }
};

// ---------------------------------------------------------------------------
// TEST_CASE A — AllocateId returns distinct monotonic IDs (fully on device)
// ---------------------------------------------------------------------------

TEST_CASE("Container<GpuCteBlobStore> - AllocateId returns distinct monotonic IDs",
          "[gpu_container_allocate_id]")
{
    EnsureGpuCteRuntime();

    ContainerTestFixture fx;
    REQUIRE(fx.Setup("gpu_container_allocate_id"));

    auto result = RunContainerTest(kernel_allocate_id, fx.ContainerPtr());
    fx.Teardown();

    INFO("kernel status: " << result.status);
    REQUIRE(result.status == 1);
    REQUIRE(result.data_match == 1);
}

// ---------------------------------------------------------------------------
// TEST_CASE B — PutGroup (host) / GroupExists (device)
//
// PutGroup is called on the host via HostPutGroup (gpu_container_helpers.cc)
// to avoid device-instantiation of GroupMetadata constructors and Serialize
// (which trigger the clang-18 NVPTX isspacep.shared IR bug).
// The device kernel verifies the blob is visible via GroupExists.
// ---------------------------------------------------------------------------

TEST_CASE("Container<GpuCteBlobStore> - PutGroup on host, GroupExists on device",
          "[gpu_container_group_roundtrip]")
{
    EnsureGpuCteRuntime();

    ContainerTestFixture fx;
    REQUIRE(fx.Setup("gpu_container_group_roundtrip"));

    // AllocateId is safe from .cu — touches only atomic, no vectors.
    GroupId gid = GroupId(fx.ContainerPtr()->AllocateId());

    // PutGroup via helper in pure CXX TU — avoids device instantiation.
    bool put_ok = kvhdf5::test::HostPutGroup(
        fx.ContainerPtr(), gid, fx.Allocator());
    REQUIRE(put_ok);

    // Kernel: GroupExists(gid) — key-index lookup, no deserialization.
    auto result = RunGroupIdKernel(kernel_group_exists, fx.ContainerPtr(), gid);
    fx.Teardown();

    INFO("kernel status: " << result.status);
    REQUIRE(result.status == 1);
    REQUIRE(result.data_match == 1);
}

// ---------------------------------------------------------------------------
// TEST_CASE C — PutDataset (host) / DatasetExists (device)
//
// Same split-phase rationale as TEST_CASE B.
// DatasetMetadata holds vector<Attribute> — same isspacep.shared trigger.
// DatasetShape::Create uses std::initializer_list (not CROSS_FUN); the helper
// builds the shape via field assignment.
// ---------------------------------------------------------------------------

TEST_CASE("Container<GpuCteBlobStore> - PutDataset on host, DatasetExists on device",
          "[gpu_container_dataset_roundtrip]")
{
    EnsureGpuCteRuntime();

    ContainerTestFixture fx;
    REQUIRE(fx.Setup("gpu_container_dataset_roundtrip"));

    DatasetId did = DatasetId(fx.ContainerPtr()->AllocateId());

    // PutDataset via helper in pure CXX TU — avoids device instantiation.
    bool put_ok = kvhdf5::test::HostPutDataset(
        fx.ContainerPtr(), did, fx.Allocator());
    REQUIRE(put_ok);

    // Kernel: DatasetExists(did) — key-index lookup, no deserialization.
    auto result = RunDatasetIdKernel(kernel_dataset_exists, fx.ContainerPtr(), did);
    fx.Teardown();

    INFO("kernel status: " << result.status);
    REQUIRE(result.status == 1);
    REQUIRE(result.data_match == 1);
}

// ---------------------------------------------------------------------------
// TEST_CASE D — PutChunk / GetChunk with ChunkKey (fully on device)
// ---------------------------------------------------------------------------

TEST_CASE("Container<GpuCteBlobStore> - PutChunk / GetChunk with ChunkKey",
          "[gpu_container_chunk_roundtrip]")
{
    EnsureGpuCteRuntime();

    ContainerTestFixture fx;
    REQUIRE(fx.Setup("gpu_container_chunk_roundtrip"));

    auto result = RunContainerTest(kernel_chunk_roundtrip, fx.ContainerPtr());
    fx.Teardown();

    INFO("kernel status: " << result.status);
    REQUIRE(result.status == 1);
    REQUIRE(result.data_match == 1);
}

// ---------------------------------------------------------------------------
// TEST_CASE E — GroupExists and DeleteGroup on device (after host PutGroup)
//
// Host puts via HostPutGroup, then kernel confirms existence and deletes.
// GroupExists / DeleteGroup are pure key-index ops — safe on device.
// ---------------------------------------------------------------------------

TEST_CASE("Container<GpuCteBlobStore> - GroupExists and DeleteGroup on device",
          "[gpu_container_exists_delete]")
{
    EnsureGpuCteRuntime();

    ContainerTestFixture fx;
    REQUIRE(fx.Setup("gpu_container_exists_delete"));

    GroupId gid = GroupId(fx.ContainerPtr()->AllocateId());

    // Put on host (avoids device instantiation of GroupMetadata).
    bool put_ok = kvhdf5::test::HostPutGroup(
        fx.ContainerPtr(), gid, fx.Allocator());
    REQUIRE(put_ok);

    // Kernel: GroupExists(pre) → DeleteGroup → GroupExists(post).
    auto result = RunGroupIdKernel(kernel_exists_delete, fx.ContainerPtr(), gid);
    fx.Teardown();

    INFO("kernel status: " << result.status);
    REQUIRE(result.status == 1);
    REQUIRE(result.data_match == 1);
}

// ---------------------------------------------------------------------------
// TEST_CASE F — RootGroup is valid and accessible on device
//
// The Container constructor puts the root group via PutGroup (host-side,
// inside ManagedContainerBox::Setup).  The kernel verifies:
//   RootGroup().IsValid() && GroupExists(RootGroup())
// Both are pure field/index ops — no vector involved.
// ---------------------------------------------------------------------------

TEST_CASE("Container<GpuCteBlobStore> - RootGroup is valid and accessible on device",
          "[gpu_container_root]")
{
    EnsureGpuCteRuntime();

    ContainerTestFixture fx;
    REQUIRE(fx.Setup("gpu_container_root"));

    auto result = RunContainerTest(kernel_root_group, fx.ContainerPtr());

    // Host-side sanity: RootGroup() accessor (simple field return, no IO)
    // and GroupExists (key-index, no deserialization).
    GroupId root_gid = fx.ContainerPtr()->RootGroup();
    REQUIRE(root_gid.IsValid());
    REQUIRE(fx.ContainerPtr()->GroupExists(root_gid));

    fx.Teardown();

    INFO("kernel status: " << result.status);
    REQUIRE(result.status == 1);
    REQUIRE(result.data_match == 1);
}

#endif  // !HSHM_IS_GPU

#endif  // HSHM_ENABLE_CUDA
