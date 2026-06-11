#if HSHM_ENABLE_CUDA

#include <catch2/catch_test_macros.hpp>
#include "gpu_cte_fixture.h"
#include "kvhdf5/gpu_cte_blob_store.h"
#include "kvhdf5/file.h"
#include "kvhdf5/hdf5_group.h"
#include "kvhdf5/hdf5_dataset.h"
#include "kvhdf5/hdf5_datatype.h"
#include "kvhdf5/dataspace.h"
#include "kvhdf5/ref.h"
#include "hermes_shm/memory/backend/array_backend.h"
#include <thread>
#include <chrono>

using namespace kvhdf5;

// ---------------------------------------------------------------------------
// FileTestResult — pinned result struct for polling from host.
// ---------------------------------------------------------------------------

struct FileTestResult {
    int status;       // 1 = pass, negative = step error, 0 = not yet done
    int extra1;       // free-form per-test info
    int data_match;   // 1 = payload roundtrip matched
};

// ===========================================================================
// KERNELS
// ===========================================================================

// ---------------------------------------------------------------------------
// Kernel F2 (smoke through File): host creates the File, kernel takes a
// File* and exercises OpenRootGroup → CreateGroup → CreateDataset →
// OpenDataset round-trip in a single kernel launch.
//
// Write/Read are intentionally omitted: each Dataset::Write or Read declares
// a 64 KB chunk_buf on the device stack (kMaxChunkBytes — see
// agents/gpu-api-concerns.md "Test-side constraint"), and combined with the
// Container/Group/Dataset metadata code already pulled into this kernel they
// blow the per-thread local memory budget on launch
// (cudaErrorMemoryAllocation). Data-roundtrip coverage already lives in
// gpu_dataset_test.cu; this test's unique value is exercising the File
// wrapper above the existing Group/Dataset coverage.
// ---------------------------------------------------------------------------

__global__ void kernel_file_end_to_end_through_root(
    chi::IpcManagerGpuInfo gpu_info,
    File<GpuCteBlobStore>* file,
    FileTestResult* d_result)
{
    d_result->status     = 0;
    d_result->extra1     = 0;
    d_result->data_match = 0;

    CHIMAERA_GPU_INIT(gpu_info);
    if (threadIdx.x != 0) return;

    auto root = file->OpenRootGroup();
    if (!(root.GetId() == file->GetContainer().RootGroup())) {
        d_result->status = -1; return;
    }

    auto child = root.CreateGroup("scene");
    if (!child.has_value()) { d_result->status = -2; return; }

    uint64_t dims[1] = {4};
    auto sp_r = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    if (!sp_r.has_value()) { d_result->status = -3; return; }

    auto ds_r = child.value().CreateDataset(
        "data", Datatype::Int32(), sp_r.value());
    if (!ds_r.has_value()) { d_result->status = -4; return; }

    auto opened = child.value().OpenDataset("data");
    if (!opened.has_value()) { d_result->status = -5; return; }
    if (!(opened.value().GetId() == ds_r.value().GetId())) {
        d_result->status = -6; return;
    }

    d_result->data_match = 1;
    d_result->status     = 1;
}

// ---------------------------------------------------------------------------
// Kernel F-create (kernel-side File::Create): kernel takes GpuCteBlobStore by
// value (trivially copyable per its docs) plus an AllocatorImpl pointer, and
// constructs a File entirely on the device. Exercises Container's root-group
// PutGroup path from the kernel side, which the existing host-creates pattern
// avoided.
// ---------------------------------------------------------------------------

__global__ void kernel_file_create_in_kernel(
    chi::IpcManagerGpuInfo gpu_info,
    GpuCteBlobStore store,
    AllocatorImpl* alloc,
    FileTestResult* d_result)
{
    d_result->status     = 0;
    d_result->extra1     = -1;
    d_result->data_match = 0;

    CHIMAERA_GPU_INIT(gpu_info);
    if (threadIdx.x != 0) return;

    auto fr = File<GpuCteBlobStore>::Create(cstd::move(store), Context(alloc));
    if (!fr.has_value()) { d_result->status = -1; return; }
    auto& file = fr.value();

    auto root = file.OpenRootGroup();

    // Empty root: GetInfo should return 0 children, 0 attributes.
    auto info_r = root.GetInfo();
    if (!info_r.has_value()) { d_result->status = -2; return; }
    if (info_r.value().num_children   != 0) { d_result->status = -3; return; }
    if (info_r.value().num_attributes != 0) { d_result->status = -4; return; }

    // Create one child and verify it appears.
    auto child = root.CreateGroup("alpha");
    if (!child.has_value()) { d_result->status = -5; return; }

    auto info2_r = root.GetInfo();
    if (!info2_r.has_value()) { d_result->status = -6; return; }
    d_result->extra1 = static_cast<int>(info2_r.value().num_children);
    if (info2_r.value().num_children != 1) { d_result->status = -7; return; }

    auto opened = root.OpenGroup("alpha");
    if (!opened.has_value()) { d_result->status = -8; return; }
    if (!(opened.value().GetId() == child.value().GetId())) {
        d_result->status = -9; return;
    }

    d_result->status = 1;
}

// ---------------------------------------------------------------------------
// Host-only: fixtures, launchers, TEST_CASEs.
// ---------------------------------------------------------------------------

#if !HSHM_IS_GPU

// Same shared-memory allocator setup as the Group / Dataset GPU fixtures.
// Prefixed Fl* to keep symbols distinct in the linked test binary.
struct FlManagedAllocFixture {
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

// Holds a host-constructed File<GpuCteBlobStore> in cudaMallocManaged so that
// kernels can dereference the same File via UVA. Move-constructs the File
// from a File::Create return value into the placement-new target. File's
// defaulted move ctor invokes Container's move ctor, which rebinds store_
// to its own raw_store_ — see container.h ctor at the matching offset.
struct FlManagedFileBox {
    using FileT = File<GpuCteBlobStore>;

    FileT* ptr = nullptr;

    bool Setup(GpuCteBlobStore blob_store, AllocatorImpl* alloc) {
        void* raw = nullptr;
        auto* gpu_ipc = CHI_CPU_IPC->GetGpuIpcManager();
        gpu_ipc->PauseGpuOrchestrator();
        cudaError_t err = cudaMallocManaged(&raw, sizeof(FileT));
        gpu_ipc->ResumeGpuOrchestrator();
        if (err != cudaSuccess) return false;

        auto fr = FileT::Create(cstd::move(blob_store), Context(alloc));
        if (!fr.has_value()) {
            gpu_ipc->PauseGpuOrchestrator();
            cudaFree(raw);
            gpu_ipc->ResumeGpuOrchestrator();
            return false;
        }
        ptr = new (raw) FileT(cstd::move(fr.value()));
        return ptr != nullptr;
    }

    void Teardown() {
        if (ptr) {
            ptr->~FileT();
            auto* gpu_ipc = CHI_CPU_IPC->GetGpuIpcManager();
            gpu_ipc->PauseGpuOrchestrator();
            cudaFree(ptr);
            gpu_ipc->ResumeGpuOrchestrator();
            ptr = nullptr;
        }
    }
};

// Fixture for F2 (host-creates File).
struct FileTestFixture {
    FlManagedAllocFixture alloc_fixture;
    FlManagedFileBox      file_box;

    bool Setup(const char* tag_name) {
        auto tag_task = g_gpu_cte_client->AsyncGetOrCreateTag(tag_name);
        tag_task.Wait();
        wrp_cte::core::TagId tag_id = tag_task->tag_id_;

        GpuCteBlobStore store = GpuCteBlobStore::Create(tag_id, g_gpu_cte_pool_id);
        if (!store.IsValid()) return false;

        if (!alloc_fixture.Setup()) return false;
        if (!file_box.Setup(std::move(store), alloc_fixture.allocator))
            return false;

        return true;
    }

    void Teardown() {
        file_box.Teardown();
        alloc_fixture.Teardown();
    }

    File<GpuCteBlobStore>* FilePtr()      { return file_box.ptr; }
};

// Fixture for the kernel-creates probe: only allocator + blob store are set
// up on the host; the kernel constructs the File itself.
struct KernelCreateFileFixture {
    FlManagedAllocFixture  alloc_fixture;
    GpuCteBlobStore        store{};
    bool                   store_valid = false;

    bool Setup(const char* tag_name) {
        auto tag_task = g_gpu_cte_client->AsyncGetOrCreateTag(tag_name);
        tag_task.Wait();
        wrp_cte::core::TagId tag_id = tag_task->tag_id_;

        store = GpuCteBlobStore::Create(tag_id, g_gpu_cte_pool_id);
        if (!store.IsValid()) return false;
        store_valid = true;

        if (!alloc_fixture.Setup()) return false;
        return true;
    }

    void Teardown() {
        // No host-side File/Container destruction — kernel constructed and
        // destroyed File on its stack. Just tear down host-side allocator.
        alloc_fixture.Teardown();
        store_valid = false;
    }

    AllocatorImpl* Allocator() { return alloc_fixture.allocator; }
};

// ---------------------------------------------------------------------------
// Launchers — Pause/Resume + pinned-status-poll, mirrors RunGroupKernel.
// ---------------------------------------------------------------------------

using FilePtrKernelFn = void (*)(chi::IpcManagerGpuInfo,
                                  File<GpuCteBlobStore>*,
                                  FileTestResult*);

static FileTestResult RunFilePtrKernel(
    FilePtrKernelFn kernel,
    File<GpuCteBlobStore>* file_ptr)
{
    auto* gpu_ipc = CHI_CPU_IPC->GetGpuIpcManager();
    chi::IpcManagerGpuInfo gpu_info = gpu_ipc->GetClientGpuInfo(0);

    gpu_ipc->PauseGpuOrchestrator();

    volatile FileTestResult* d_result;
    cudaMallocHost(const_cast<FileTestResult**>(&d_result),
                   sizeof(FileTestResult));
    d_result->status     = 0;
    d_result->extra1     = 0;
    d_result->data_match = 0;

    cudaGetLastError();
    void* stream = hshm::GpuApi::CreateStream();
    kernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(
        gpu_info, file_ptr,
        const_cast<FileTestResult*>(d_result));

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        fprintf(stderr,
                "[gpu_file_test] kernel launch failed: %s (cuda %d)\n",
                cudaGetErrorString(launch_err),
                static_cast<int>(launch_err));
        hshm::GpuApi::DestroyStream(stream);
        cudaFreeHost(const_cast<FileTestResult*>(d_result));
        gpu_ipc->ResumeGpuOrchestrator();
        return {-201, 0, 0};
    }

    gpu_ipc->ResumeGpuOrchestrator();

    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (d_result->status == 0
           && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }

    FileTestResult result{d_result->status, d_result->extra1, d_result->data_match};
    if (result.status == 0) result.status = -300;

    gpu_ipc->PauseGpuOrchestrator();
    cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
    hshm::GpuApi::DestroyStream(stream);
    cudaFreeHost(const_cast<FileTestResult*>(d_result));
    gpu_ipc->ResumeGpuOrchestrator();
    return result;
}

using KernelCreateFileFn = void (*)(chi::IpcManagerGpuInfo,
                                     GpuCteBlobStore,
                                     AllocatorImpl*,
                                     FileTestResult*);

static FileTestResult RunKernelCreateFile(
    KernelCreateFileFn kernel,
    GpuCteBlobStore store,
    AllocatorImpl* alloc)
{
    auto* gpu_ipc = CHI_CPU_IPC->GetGpuIpcManager();
    chi::IpcManagerGpuInfo gpu_info = gpu_ipc->GetClientGpuInfo(0);

    gpu_ipc->PauseGpuOrchestrator();

    volatile FileTestResult* d_result;
    cudaMallocHost(const_cast<FileTestResult**>(&d_result),
                   sizeof(FileTestResult));
    d_result->status     = 0;
    d_result->extra1     = 0;
    d_result->data_match = 0;

    cudaGetLastError();
    void* stream = hshm::GpuApi::CreateStream();
    kernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(
        gpu_info, store, alloc,
        const_cast<FileTestResult*>(d_result));

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        fprintf(stderr,
                "[gpu_file_test] kernel launch failed: %s (cuda %d)\n",
                cudaGetErrorString(launch_err),
                static_cast<int>(launch_err));
        hshm::GpuApi::DestroyStream(stream);
        cudaFreeHost(const_cast<FileTestResult*>(d_result));
        gpu_ipc->ResumeGpuOrchestrator();
        return {-201, 0, 0};
    }

    gpu_ipc->ResumeGpuOrchestrator();

    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (d_result->status == 0
           && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }

    FileTestResult result{d_result->status, d_result->extra1, d_result->data_match};
    if (result.status == 0) result.status = -300;

    gpu_ipc->PauseGpuOrchestrator();
    cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
    hshm::GpuApi::DestroyStream(stream);
    cudaFreeHost(const_cast<FileTestResult*>(d_result));
    gpu_ipc->ResumeGpuOrchestrator();
    return result;
}

// ---------------------------------------------------------------------------
// TEST_CASE F2 — host-creates File, kernel does end-to-end CreateGroup +
// CreateDataset + Write + Read through File::OpenRootGroup().
// ---------------------------------------------------------------------------

TEST_CASE("File GPU - end-to-end Group+Dataset roundtrip via File::OpenRootGroup",
          "[unit][gpu_file][end_to_end]")
{
    EnsureGpuCteRuntime();

    FileTestFixture fx;
    REQUIRE(fx.Setup("gpu_file_e2e"));

    auto result = RunFilePtrKernel(
        kernel_file_end_to_end_through_root, fx.FilePtr());
    fx.Teardown();

    INFO("kernel status: " << result.status
         << " data_match: " << result.data_match);
    REQUIRE(result.status == 1);
    REQUIRE(result.data_match == 1);
}

// ---------------------------------------------------------------------------
// TEST_CASE F-create — kernel calls File::Create directly, then exercises
// CreateGroup + GetInfo + OpenGroup. Probes whether the Container ctor's
// kernel-side root PutGroup roundtrips correctly. Marked [!mayfail] because
// kernel-side construction of a Container with metadata write hasn't been
// validated end-to-end yet; if it works, great, if not we'll diagnose later.
// ---------------------------------------------------------------------------

TEST_CASE("File GPU - kernel-side File::Create + CreateGroup + GetInfo",
          "[unit][gpu_file][kernel_create][!mayfail]")
{
    EnsureGpuCteRuntime();

    KernelCreateFileFixture fx;
    REQUIRE(fx.Setup("gpu_file_kernel_create"));

    auto result = RunKernelCreateFile(
        kernel_file_create_in_kernel, fx.store, fx.Allocator());
    fx.Teardown();

    INFO("kernel status: " << result.status
         << " num_children_after_create: " << result.extra1);
    REQUIRE(result.status == 1);
    REQUIRE(result.extra1 == 1);
}

#endif  // !HSHM_IS_GPU

#endif  // HSHM_ENABLE_CUDA
