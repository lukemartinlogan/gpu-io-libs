#if HSHM_ENABLE_CUDA

#include <catch2/catch_test_macros.hpp>
#include "gpu_cte_fixture.h"
#include "kvhdf5/gpu_cte_blob_store.h"
#include "kvhdf5/container.h"
#include "kvhdf5/hdf5_group.h"
#include "kvhdf5/hdf5_dataset.h"
#include "kvhdf5/hdf5_datatype.h"
#include "kvhdf5/dataspace.h"
#include "kvhdf5/ref.h"
#include "hermes_shm/memory/backend/array_backend.h"
#include "gpu_container_helpers.h"
#include <thread>
#include <chrono>

using namespace kvhdf5;

// ---------------------------------------------------------------------------
// GroupTestResult — pinned result struct for polling from host.
// status:  1 = pass, negative = step error, 0 = not yet done
// extra1/extra2 free fields for tests to communicate counts back.
// ---------------------------------------------------------------------------

struct GroupTestResult {
    int status;
    int extra1;
    int extra2;
};

// ===========================================================================
// KERNELS
//
// All Group GPU tests follow the gpu_dataset_test.cu fixture pattern: the
// host creates the Container in cudaMallocManaged. Container's constructor
// seeds the root group via PutGroup on the host, so kernels can fetch it
// with container->RootGroup() and call Group<B>::CreateGroup / OpenGroup /
// SetAttribute / GetAttribute / GetInfo / CreateDataset directly.
//
// All kernels are <<<1, 1>>> after CHIMAERA_GPU_INIT (lane-0-only).
//
// All round-trips happen inside a single kernel launch — separate kernels
// reading metadata that an earlier kernel wrote can hit the BufferDeserializer
// assert documented in agents/gpu-api-concerns.md
// ("Kernel-side DatasetMetadata roundtrip across kernel launches"). Group
// metadata uses the same vector<Attribute>/vector<GroupEntry> pattern, so
// we follow the same single-kernel discipline here.
// ===========================================================================

// ---------------------------------------------------------------------------
// Kernel G1: CreateGroup + OpenGroup round-trip.
// ---------------------------------------------------------------------------

__global__ void kernel_group_create_open_roundtrip(
    chi::IpcManagerGpuInfo gpu_info,
    Container<GpuCteBlobStore>* container,
    GroupTestResult* d_result)
{
    d_result->status = 0;
    d_result->extra1 = 0;
    d_result->extra2 = 0;

    CHIMAERA_GPU_INIT(gpu_info);
    if (threadIdx.x != 0) return;

    Group<GpuCteBlobStore> root(container->RootGroup(),
                                Ref<Container<GpuCteBlobStore>>(*container));

    auto created = root.CreateGroup("simulation");
    if (!created.has_value()) { d_result->status = -1; return; }

    auto opened = root.OpenGroup("simulation");
    if (!opened.has_value()) { d_result->status = -2; return; }

    if (!(created.value().GetId() == opened.value().GetId())) {
        d_result->status = -3;
        return;
    }

    d_result->status = 1;
}

// ---------------------------------------------------------------------------
// Kernel G2: OpenGroup of a missing name returns an error.
// ---------------------------------------------------------------------------

__global__ void kernel_group_open_nonexistent(
    chi::IpcManagerGpuInfo gpu_info,
    Container<GpuCteBlobStore>* container,
    GroupTestResult* d_result)
{
    d_result->status = 0;

    CHIMAERA_GPU_INIT(gpu_info);
    if (threadIdx.x != 0) return;

    Group<GpuCteBlobStore> root(container->RootGroup(),
                                Ref<Container<GpuCteBlobStore>>(*container));

    auto r = root.OpenGroup("nonexistent");
    if (r.has_value()) { d_result->status = -1; return; }

    d_result->status = 1;
}

// ---------------------------------------------------------------------------
// Kernel G3: CreateGroup with a duplicate name returns an error.
// ---------------------------------------------------------------------------

__global__ void kernel_group_create_duplicate(
    chi::IpcManagerGpuInfo gpu_info,
    Container<GpuCteBlobStore>* container,
    GroupTestResult* d_result)
{
    d_result->status = 0;

    CHIMAERA_GPU_INIT(gpu_info);
    if (threadIdx.x != 0) return;

    Group<GpuCteBlobStore> root(container->RootGroup(),
                                Ref<Container<GpuCteBlobStore>>(*container));

    auto first = root.CreateGroup("a");
    if (!first.has_value()) { d_result->status = -1; return; }

    auto second = root.CreateGroup("a");
    if (second.has_value()) { d_result->status = -2; return; }

    d_result->status = 1;
}

// ---------------------------------------------------------------------------
// Kernel G4: GetInfo reflects host-seeded children + attributes.
// Host pre-populates: 2 child groups, 1 int32 attribute on root. Kernel
// reads root.GetInfo() and reports counts back through extra1/extra2.
// ---------------------------------------------------------------------------

__global__ void kernel_group_get_info_seeded(
    chi::IpcManagerGpuInfo gpu_info,
    Container<GpuCteBlobStore>* container,
    GroupTestResult* d_result)
{
    d_result->status = 0;
    d_result->extra1 = -1;
    d_result->extra2 = -1;

    CHIMAERA_GPU_INIT(gpu_info);
    if (threadIdx.x != 0) return;

    Group<GpuCteBlobStore> root(container->RootGroup(),
                                Ref<Container<GpuCteBlobStore>>(*container));

    auto info_r = root.GetInfo();
    if (!info_r.has_value()) { d_result->status = -1; return; }

    d_result->extra1 = static_cast<int>(info_r.value().num_children);
    d_result->extra2 = static_cast<int>(info_r.value().num_attributes);

    if (info_r.value().num_children != 2)   { d_result->status = -2; return; }
    if (info_r.value().num_attributes != 1) { d_result->status = -3; return; }

    d_result->status = 1;
}

// ---------------------------------------------------------------------------
// Kernel G5: SetAttribute + GetAttribute round-trip + overwrite (single kernel).
// ---------------------------------------------------------------------------

__global__ void kernel_group_attribute_roundtrip(
    chi::IpcManagerGpuInfo gpu_info,
    Container<GpuCteBlobStore>* container,
    GroupTestResult* d_result)
{
    d_result->status = 0;
    d_result->extra1 = 0;
    d_result->extra2 = 0;

    CHIMAERA_GPU_INIT(gpu_info);
    if (threadIdx.x != 0) return;

    Group<GpuCteBlobStore> root(container->RootGroup(),
                                Ref<Container<GpuCteBlobStore>>(*container));

    auto dt = Datatype::Int32();

    int32_t v1 = 42;
    if (!root.SetAttribute("count", dt, &v1).has_value()) {
        d_result->status = -1; return;
    }

    if (!root.HasAttribute("count")) { d_result->status = -2; return; }
    if (root.HasAttribute("missing")) { d_result->status = -3; return; }

    int32_t r1 = 0;
    if (!root.GetAttribute("count", dt, &r1).has_value()) {
        d_result->status = -4; return;
    }
    d_result->extra1 = r1;
    if (r1 != 42) { d_result->status = -5; return; }

    // Overwrite same name with new value
    int32_t v2 = 99;
    if (!root.SetAttribute("count", dt, &v2).has_value()) {
        d_result->status = -6; return;
    }

    int32_t r2 = 0;
    if (!root.GetAttribute("count", dt, &r2).has_value()) {
        d_result->status = -7; return;
    }
    d_result->extra2 = r2;
    if (r2 != 99) { d_result->status = -8; return; }

    d_result->status = 1;
}

// ---------------------------------------------------------------------------
// Kernel G6: CreateDataset registers a Dataset child entry on the parent.
// After CreateDataset, root.GetInfo().num_children must be 1 and
// OpenDataset("data") must succeed.
// ---------------------------------------------------------------------------

__global__ void kernel_group_create_dataset_registers_child(
    chi::IpcManagerGpuInfo gpu_info,
    Container<GpuCteBlobStore>* container,
    GroupTestResult* d_result)
{
    d_result->status = 0;
    d_result->extra1 = -1;

    CHIMAERA_GPU_INIT(gpu_info);
    if (threadIdx.x != 0) return;

    Group<GpuCteBlobStore> root(container->RootGroup(),
                                Ref<Container<GpuCteBlobStore>>(*container));

    uint64_t dims[1] = {4};
    auto sp_r = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    if (!sp_r.has_value()) { d_result->status = -1; return; }

    auto ds_r = root.CreateDataset("data", Datatype::Int32(), sp_r.value());
    if (!ds_r.has_value()) { d_result->status = -2; return; }

    auto info_r = root.GetInfo();
    if (!info_r.has_value()) { d_result->status = -3; return; }
    d_result->extra1 = static_cast<int>(info_r.value().num_children);
    if (info_r.value().num_children != 1) { d_result->status = -4; return; }

    auto open_r = root.OpenDataset("data");
    if (!open_r.has_value()) { d_result->status = -5; return; }
    if (!(open_r.value().GetId() == ds_r.value().GetId())) {
        d_result->status = -6; return;
    }

    d_result->status = 1;
}

// ---------------------------------------------------------------------------
// Host-only: fixtures, launcher, TEST_CASEs.
// ---------------------------------------------------------------------------

#if !HSHM_IS_GPU

// Mirror the DsManagedAllocFixture / DsManagedContainerBox pattern from
// gpu_dataset_test.cu, prefixed Gp* to keep symbol names distinct in the
// linked test binary.
struct GpManagedAllocFixture {
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

struct GpManagedContainerBox {
    using ContainerT = Container<GpuCteBlobStore>;

    ContainerT* ptr = nullptr;

    bool Setup(GpuCteBlobStore blob_store, AllocatorImpl* alloc) {
        void* raw = nullptr;
        auto* gpu_ipc = CHI_CPU_IPC->GetGpuIpcManager();
        gpu_ipc->PauseGpuOrchestrator();
        cudaError_t err = cudaMallocManaged(&raw, sizeof(ContainerT));
        gpu_ipc->ResumeGpuOrchestrator();
        if (err != cudaSuccess) return false;

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

struct GroupTestFixture {
    GpManagedAllocFixture  alloc_fixture;
    GpManagedContainerBox  container_box;

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
// RunGroupKernel — Pause/Resume + pinned-status-poll launcher matching
// RunDatasetKernel / RunContainerTest. Single-thread launch; see
// gpu_dataset_test.cu for the local-memory rationale.
// ---------------------------------------------------------------------------

using GroupKernelFn = void (*)(chi::IpcManagerGpuInfo,
                                Container<GpuCteBlobStore>*,
                                GroupTestResult*);

static GroupTestResult RunGroupKernel(
    GroupKernelFn kernel,
    Container<GpuCteBlobStore>* container_ptr)
{
    auto* gpu_ipc = CHI_CPU_IPC->GetGpuIpcManager();
    chi::IpcManagerGpuInfo gpu_info = gpu_ipc->GetClientGpuInfo(0);

    gpu_ipc->PauseGpuOrchestrator();

    volatile GroupTestResult* d_result;
    cudaMallocHost(const_cast<GroupTestResult**>(&d_result),
                   sizeof(GroupTestResult));
    d_result->status = 0;
    d_result->extra1 = 0;
    d_result->extra2 = 0;

    cudaGetLastError();
    void* stream = hshm::GpuApi::CreateStream();
    kernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(
        gpu_info, container_ptr,
        const_cast<GroupTestResult*>(d_result));

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        fprintf(stderr,
                "[gpu_group_test] kernel launch failed: %s (cuda %d)\n",
                cudaGetErrorString(launch_err),
                static_cast<int>(launch_err));
        hshm::GpuApi::DestroyStream(stream);
        cudaFreeHost(const_cast<GroupTestResult*>(d_result));
        gpu_ipc->ResumeGpuOrchestrator();
        return {-201, 0, 0};
    }

    gpu_ipc->ResumeGpuOrchestrator();

    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (d_result->status == 0
           && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }

    GroupTestResult result{d_result->status, d_result->extra1, d_result->extra2};
    if (result.status == 0) result.status = -300;

    gpu_ipc->PauseGpuOrchestrator();
    cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
    hshm::GpuApi::DestroyStream(stream);
    cudaFreeHost(const_cast<GroupTestResult*>(d_result));
    gpu_ipc->ResumeGpuOrchestrator();
    return result;
}

// ---------------------------------------------------------------------------
// TEST_CASE G1 — Create+Open round-trip.
// ---------------------------------------------------------------------------

TEST_CASE("Group GPU - CreateGroup + OpenGroup round-trip",
          "[unit][gpu_group][create_open]")
{
    EnsureGpuCteRuntime();

    GroupTestFixture fx;
    REQUIRE(fx.Setup("gpu_group_create_open"));

    auto result = RunGroupKernel(
        kernel_group_create_open_roundtrip, fx.ContainerPtr());
    fx.Teardown();

    INFO("kernel status: " << result.status);
    REQUIRE(result.status == 1);
}

// ---------------------------------------------------------------------------
// TEST_CASE G2 — OpenGroup nonexistent fails.
// ---------------------------------------------------------------------------

TEST_CASE("Group GPU - OpenGroup nonexistent returns error",
          "[unit][gpu_group][open_missing]")
{
    EnsureGpuCteRuntime();

    GroupTestFixture fx;
    REQUIRE(fx.Setup("gpu_group_open_missing"));

    auto result = RunGroupKernel(
        kernel_group_open_nonexistent, fx.ContainerPtr());
    fx.Teardown();

    INFO("kernel status: " << result.status);
    REQUIRE(result.status == 1);
}

// ---------------------------------------------------------------------------
// TEST_CASE G3 — CreateGroup duplicate name fails.
// ---------------------------------------------------------------------------

TEST_CASE("Group GPU - CreateGroup duplicate name returns error",
          "[unit][gpu_group][duplicate]")
{
    EnsureGpuCteRuntime();

    GroupTestFixture fx;
    REQUIRE(fx.Setup("gpu_group_duplicate"));

    auto result = RunGroupKernel(
        kernel_group_create_duplicate, fx.ContainerPtr());
    fx.Teardown();

    INFO("kernel status: " << result.status);
    REQUIRE(result.status == 1);
}

// ---------------------------------------------------------------------------
// TEST_CASE G4 — GetInfo reads host-seeded child + attribute counts.
// ---------------------------------------------------------------------------

TEST_CASE("Group GPU - GetInfo reflects host-seeded counts",
          "[unit][gpu_group][get_info]")
{
    EnsureGpuCteRuntime();

    GroupTestFixture fx;
    REQUIRE(fx.Setup("gpu_group_get_info"));

    GroupId root_gid = fx.ContainerPtr()->RootGroup();
    kvhdf5::test::HostAddChildGroup(
        fx.ContainerPtr(), root_gid, "child1", fx.Allocator());
    kvhdf5::test::HostAddChildGroup(
        fx.ContainerPtr(), root_gid, "child2", fx.Allocator());
    REQUIRE(kvhdf5::test::HostAddInt32Attribute(
        fx.ContainerPtr(), root_gid, "tag", 7, fx.Allocator()));

    auto result = RunGroupKernel(
        kernel_group_get_info_seeded, fx.ContainerPtr());
    fx.Teardown();

    INFO("kernel status: " << result.status
         << " num_children: " << result.extra1
         << " num_attributes: " << result.extra2);
    REQUIRE(result.status == 1);
    REQUIRE(result.extra1 == 2);
    REQUIRE(result.extra2 == 1);
}

// ---------------------------------------------------------------------------
// TEST_CASE G5 — SetAttribute / GetAttribute round-trip + overwrite.
// ---------------------------------------------------------------------------

TEST_CASE("Group GPU - SetAttribute / GetAttribute round-trip and overwrite",
          "[unit][gpu_group][attribute]")
{
    EnsureGpuCteRuntime();

    GroupTestFixture fx;
    REQUIRE(fx.Setup("gpu_group_attribute"));

    auto result = RunGroupKernel(
        kernel_group_attribute_roundtrip, fx.ContainerPtr());
    fx.Teardown();

    INFO("kernel status: " << result.status
         << " first_read: " << result.extra1
         << " overwrite_read: " << result.extra2);
    REQUIRE(result.status == 1);
    REQUIRE(result.extra1 == 42);
    REQUIRE(result.extra2 == 99);
}

// ---------------------------------------------------------------------------
// TEST_CASE G6 — CreateDataset registers a child entry on the parent group.
// ---------------------------------------------------------------------------

TEST_CASE("Group GPU - CreateDataset registers child on parent",
          "[unit][gpu_group][create_dataset]")
{
    EnsureGpuCteRuntime();

    GroupTestFixture fx;
    REQUIRE(fx.Setup("gpu_group_create_dataset"));

    auto result = RunGroupKernel(
        kernel_group_create_dataset_registers_child, fx.ContainerPtr());
    fx.Teardown();

    INFO("kernel status: " << result.status
         << " num_children: " << result.extra1);
    REQUIRE(result.status == 1);
    REQUIRE(result.extra1 == 1);
}

#endif  // !HSHM_IS_GPU

#endif  // HSHM_ENABLE_CUDA
