#if HSHM_ENABLE_CUDA

#include <catch2/catch_test_macros.hpp>
#include "gpu_cte_fixture.h"
#include "kvhdf5/gpu_cte_blob_store.h"
#include <cstring>
#include <thread>
#include <chrono>

using kvhdf5::byte_t;

// 256-byte blobs fit within the default FutureShm copy space so the GPU→CPU
// path can transfer results without a secondary allocation.
static constexpr size_t kKernelBlobSize = 256;

// ---- GPU backend setup -------------------------------------------------------

struct GpuBackends {
    hipc::GpuShmMmap primary;
    hipc::GpuShmMmap g2c;
    bool ok = false;
};

// Returns initialized backends with primary allocator registered.
// The GpuShmMmap objects must outlive any kernel that uses their memory.
static GpuBackends SetupGpuBackends() {
    GpuBackends b;

    hipc::MemoryBackendId primary_id(50, 0);
    if (!b.primary.shm_init(primary_id, 10 * 1024 * 1024,
                             "/kvhdf5_gpu_cte_primary", 0)) {
        return b;
    }

    hipc::MemoryBackendId g2c_id(51, 0);
    if (!b.g2c.shm_init(g2c_id, 4 * 1024 * 1024,
                         "/kvhdf5_gpu_cte_g2c", 0)) {
        return b;
    }

    CHI_CPU_IPC->GetGpuIpcManager()->RegisterGpuAllocator(
        primary_id, b.primary.data_, b.primary.data_capacity_);
    b.ok = true;
    return b;
}

static chi::IpcManagerGpuInfo BuildGpuInfo(GpuBackends &b) {
    chi::IpcManagerGpuInfo info;
    info.backend         = static_cast<hipc::MemoryBackend &>(b.primary);
    info.gpu2cpu_queue   = CHI_CPU_IPC->GetGpuQueue(0);
    info.gpu2cpu_backend = static_cast<hipc::MemoryBackend &>(b.g2c);
    return info;
}

// ---- Kernel 1: GPU backend init + AllocateBuffer ----------------------------
//
// Verifies that CHIMAERA_GPU_INIT succeeds and AllocateBuffer works from
// device code.  The kernel writes a sentinel pattern into a UVM buffer that
// the host then reads back via a separate pinned output pointer.

__global__ void kernel_alloc_and_fill(chi::IpcManagerGpu gpu_info,
                                       size_t blob_size,
                                       int *d_result) {
    *d_result = 0;
    CHIMAERA_GPU_INIT(gpu_info);

    hipc::FullPtr<char> buf = CHI_IPC->AllocateBuffer(blob_size);
    if (buf.IsNull()) {
        *d_result = -10;
        return;
    }

    // fill with sentinel pattern 0xBE
    for (size_t i = 0; i < blob_size; ++i) {
        buf.ptr_[i] = static_cast<char>(0xBE);
    }

    CHI_IPC->FreeBuffer(buf);
    *d_result = 1;
}

extern "C" int run_gpu_alloc_test() {
    // Use the runtime-managed GPU info rather than manually constructing one.
    // The persistent orchestrator kernel must be paused before any device-
    // synchronizing host API (cudaMallocHost, cudaStreamCreate) because CDP
    // deadlocks with it, and resumed once the client kernel is in flight.
    auto *gpu_ipc = CHI_CPU_IPC->GetGpuIpcManager();
    chi::IpcManagerGpuInfo gpu_info = gpu_ipc->GetClientGpuInfo(0);

    gpu_ipc->PauseGpuOrchestrator();

    int *d_result = hshm::GpuApi::Malloc<int>(sizeof(int));
    int h_result = 0;
    hshm::GpuApi::Memcpy(d_result, &h_result, sizeof(int));

    cudaGetLastError();
    void *stream = hshm::GpuApi::CreateStream();
    kernel_alloc_and_fill<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(
        gpu_info, kKernelBlobSize, d_result);

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        gpu_ipc->ResumeGpuOrchestrator();
        hshm::GpuApi::Free(d_result);
        hshm::GpuApi::DestroyStream(stream);
        return -201;
    }

    // This kernel does not submit CTE tasks, so the orchestrator does not
    // need to run concurrently. Sync the stream, then pause cleanly.
    hshm::GpuApi::Synchronize(stream);
    cudaError_t sync_err = cudaGetLastError();
    if (sync_err != cudaSuccess) {
        gpu_ipc->ResumeGpuOrchestrator();
        hshm::GpuApi::Free(d_result);
        hshm::GpuApi::DestroyStream(stream);
        return -200;
    }

    hshm::GpuApi::Memcpy(&h_result, d_result, sizeof(int));
    hshm::GpuApi::Free(d_result);
    hshm::GpuApi::DestroyStream(stream);
    gpu_ipc->ResumeGpuOrchestrator();
    return h_result;
}

// ---- Kernel 2: kernel writes pattern into host-allocated UVM buffer ---------
//
// Host allocates a managed (UVA) buffer; kernel writes a pattern into it; host
// then submits a PutBlob with that same buffer wrapped via ShmPtr::FromRaw.
// The buffer's UVA address is identical on both sides, so the server resolves
// the ShmPtr as an absolute pointer. This mirrors iowarp-core's canonical
// pattern (test_gpu_initiated_gpu.cc, test_bdev_gpucache.cc) — never have the
// kernel allocate a buffer that the host then needs to resolve, because the
// per-warp BuddyAllocator's alloc_id is not registered in the CPU-side
// gpu_alloc_map_.

__global__ void kernel_write_uvm_pattern(chi::IpcManagerGpu gpu_info,
                                          char *buf,
                                          size_t blob_size,
                                          int *d_status) {
    *d_status = 0;
    CHIMAERA_GPU_INIT(gpu_info);
    if (threadIdx.x != 0) return;

    for (size_t i = 0; i < blob_size; ++i) {
        buf[i] = static_cast<char>(i % 251);
    }
    *d_status = 1;
}

extern "C" int run_gpu_write_then_host_put_test(chi::PoolId pool_id,
                                                 wrp_cte::core::TagId tag_id) {
    auto *gpu_ipc = CHI_CPU_IPC->GetGpuIpcManager();
    chi::IpcManagerGpuInfo gpu_info = gpu_ipc->GetClientGpuInfo(0);

    gpu_ipc->PauseGpuOrchestrator();

    char *buf = nullptr;
    if (cudaMallocManaged(reinterpret_cast<void **>(&buf), kKernelBlobSize)
            != cudaSuccess) {
        gpu_ipc->ResumeGpuOrchestrator();
        return -100;
    }
    int *d_status = nullptr;
    if (cudaMallocHost(reinterpret_cast<void **>(&d_status), sizeof(int))
            != cudaSuccess) {
        cudaFree(buf);
        gpu_ipc->ResumeGpuOrchestrator();
        return -101;
    }
    *d_status = 0;

    cudaGetLastError();
    void *stream = hshm::GpuApi::CreateStream();
    kernel_write_uvm_pattern<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(
        gpu_info, buf, kKernelBlobSize, d_status);

    if (cudaGetLastError() != cudaSuccess) {
        cudaFreeHost(d_status);
        cudaFree(buf);
        gpu_ipc->ResumeGpuOrchestrator();
        hshm::GpuApi::DestroyStream(stream);
        return -201;
    }

    hshm::GpuApi::Synchronize(stream);
    if (cudaGetLastError() != cudaSuccess) {
        cudaFreeHost(d_status);
        cudaFree(buf);
        gpu_ipc->ResumeGpuOrchestrator();
        hshm::GpuApi::DestroyStream(stream);
        return -200;
    }

    if (*d_status != 1) {
        int s = *d_status;
        cudaFreeHost(d_status);
        cudaFree(buf);
        gpu_ipc->ResumeGpuOrchestrator();
        hshm::GpuApi::DestroyStream(stream);
        return s;
    }

    // Resume so the orchestrator can service the host-submitted PutBlob.
    gpu_ipc->ResumeGpuOrchestrator();

    auto *cpu_ipc = CHI_CPU_IPC;
    hipc::ShmPtr<> shm = hipc::ShmPtr<>::FromRaw(buf);
    auto put_task = cpu_ipc->template NewTask<wrp_cte::core::PutBlobTask>(
        chi::CreateTaskId(), pool_id, chi::PoolQuery::Local(),
        tag_id, "gpu_write_host_put_blob", chi::u64(0),
        chi::u64(kKernelBlobSize), shm, 1.0f,
        wrp_cte::core::Context(), chi::u32(0));
    auto put_future = cpu_ipc->Send(put_task);
    put_future.Wait();
    int rc = static_cast<int>(put_task->GetReturnCode());

    gpu_ipc->PauseGpuOrchestrator();
    cudaFreeHost(d_status);
    cudaFree(buf);
    gpu_ipc->ResumeGpuOrchestrator();
    hshm::GpuApi::DestroyStream(stream);
    return (rc == 0) ? 1 : -30;
}

// ---- Kernel 3: host PutBlob, host GetBlob, kernel verifies via UVA ----------
//
// Host allocates two managed (UVA) buffers. Pattern → src → host PutBlob →
// host GetBlob into recv → kernel reads recv (via UVA) and reports match.
// All ShmPtr-bearing tasks come from the host with FromRaw of host-allocated
// UVA pointers, so server-side resolution always succeeds.

__global__ void kernel_verify_uvm_pattern(chi::IpcManagerGpu gpu_info,
                                           const char *buf,
                                           size_t blob_size,
                                           int *d_match) {
    *d_match = 0;
    CHIMAERA_GPU_INIT(gpu_info);
    if (threadIdx.x != 0) return;

    int match = 1;
    for (size_t i = 0; i < blob_size; ++i) {
        if (static_cast<unsigned char>(buf[i])
                != static_cast<unsigned char>(i % 251)) {
            match = 0;
            break;
        }
    }
    *d_match = match;
}

extern "C" int run_host_put_gpu_get_test(chi::PoolId pool_id,
                                          wrp_cte::core::TagId tag_id,
                                          bool *data_match_out) {
    *data_match_out = false;

    auto *gpu_ipc = CHI_CPU_IPC->GetGpuIpcManager();
    chi::IpcManagerGpuInfo gpu_info = gpu_ipc->GetClientGpuInfo(0);

    gpu_ipc->PauseGpuOrchestrator();
    char *src = nullptr;
    char *recv = nullptr;
    int *d_match = nullptr;
    if (cudaMallocManaged(reinterpret_cast<void **>(&src), kKernelBlobSize)
            != cudaSuccess) {
        gpu_ipc->ResumeGpuOrchestrator();
        return -101;
    }
    if (cudaMallocManaged(reinterpret_cast<void **>(&recv), kKernelBlobSize)
            != cudaSuccess) {
        cudaFree(src);
        gpu_ipc->ResumeGpuOrchestrator();
        return -102;
    }
    if (cudaMallocHost(reinterpret_cast<void **>(&d_match), sizeof(int))
            != cudaSuccess) {
        cudaFree(src);
        cudaFree(recv);
        gpu_ipc->ResumeGpuOrchestrator();
        return -103;
    }
    *d_match = 0;
    for (size_t i = 0; i < kKernelBlobSize; ++i) {
        src[i] = static_cast<char>(i % 251);
        recv[i] = 0;
    }
    gpu_ipc->ResumeGpuOrchestrator();

    auto *cpu_ipc = CHI_CPU_IPC;

    {
        hipc::ShmPtr<> put_shm = hipc::ShmPtr<>::FromRaw(src);
        auto put_task = cpu_ipc->template NewTask<wrp_cte::core::PutBlobTask>(
            chi::CreateTaskId(), pool_id, chi::PoolQuery::Local(),
            tag_id, "host_put_gpu_get_blob", chi::u64(0),
            chi::u64(kKernelBlobSize), put_shm, 1.0f,
            wrp_cte::core::Context(), chi::u32(0));
        auto put_future = cpu_ipc->Send(put_task);
        put_future.Wait();
        if (put_task->GetReturnCode() != 0) {
            gpu_ipc->PauseGpuOrchestrator();
            cudaFreeHost(d_match);
            cudaFree(src);
            cudaFree(recv);
            gpu_ipc->ResumeGpuOrchestrator();
            return -104;
        }
    }

    {
        hipc::ShmPtr<> get_shm = hipc::ShmPtr<>::FromRaw(recv);
        auto get_task = cpu_ipc->template NewTask<wrp_cte::core::GetBlobTask>(
            chi::CreateTaskId(), pool_id, chi::PoolQuery::Local(),
            tag_id, "host_put_gpu_get_blob", chi::u64(0),
            chi::u64(kKernelBlobSize), chi::u32(0), get_shm);
        auto get_future = cpu_ipc->Send(get_task);
        get_future.Wait();
        if (get_task->GetReturnCode() != 0) {
            gpu_ipc->PauseGpuOrchestrator();
            cudaFreeHost(d_match);
            cudaFree(src);
            cudaFree(recv);
            gpu_ipc->ResumeGpuOrchestrator();
            return -30;
        }
    }

    // Pause/Resume bracket spans CreateStream → launch → Synchronize, matching
    // run_gpu_alloc_test (test 1). cudaStreamSynchronize is device-syncing and
    // deadlocks against the persistent CDP orchestrator if it is running.
    gpu_ipc->PauseGpuOrchestrator();
    cudaGetLastError();
    void *stream = hshm::GpuApi::CreateStream();

    kernel_verify_uvm_pattern<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(
        gpu_info, recv, kKernelBlobSize, d_match);

    if (cudaGetLastError() != cudaSuccess) {
        cudaFreeHost(d_match);
        cudaFree(src);
        cudaFree(recv);
        gpu_ipc->ResumeGpuOrchestrator();
        hshm::GpuApi::DestroyStream(stream);
        return -201;
    }

    hshm::GpuApi::Synchronize(stream);

    *data_match_out = (*d_match == 1);

    cudaFreeHost(d_match);
    cudaFree(src);
    cudaFree(recv);
    gpu_ipc->ResumeGpuOrchestrator();
    hshm::GpuApi::DestroyStream(stream);
    return 1;
}

// ---- Kernel 4: kernel PutBlob via GpuCteBlobStore -> host GetBlob -----------
//
// Verifies the kernel-side dual-send: when a kernel calls
// GpuCteBlobStore::PutBlob, the value is replicated to both the GPU CTE
// (PoolQuery::Local) and the CPU CTE (PoolQuery::ToLocalCpu) so a host-side
// reader can fetch it through the CPU runtime.

struct StoreCrossResult {
    int status;     // 1 = kernel finished, negative = error
};

__global__ void kernel_store_put_pattern(chi::IpcManagerGpu gpu_info,
                                          kvhdf5::GpuCteBlobStore store,
                                          StoreCrossResult *d_result) {
    d_result->status = 0;
    CHIMAERA_GPU_INIT(gpu_info);
    if (threadIdx.x != 0) return;

    cstd::array<byte_t, 4> key = {byte_t{0x91}, byte_t{0x92},
                                   byte_t{0x93}, byte_t{0x94}};
    cstd::array<byte_t, 8> value = {
        byte_t{0xCA}, byte_t{0xFE}, byte_t{0xBA}, byte_t{0xBE},
        byte_t{0x01}, byte_t{0x02}, byte_t{0x03}, byte_t{0x04}
    };

    if (!store.PutBlob(cstd::span<const byte_t>(key.data(), key.size()),
                       cstd::span<const byte_t>(value.data(), value.size()))) {
        d_result->status = -1;
        return;
    }
    d_result->status = 1;
}

__global__ void kernel_store_delete_key(chi::IpcManagerGpu gpu_info,
                                         kvhdf5::GpuCteBlobStore store,
                                         StoreCrossResult *d_result) {
    d_result->status = 0;
    CHIMAERA_GPU_INIT(gpu_info);
    if (threadIdx.x != 0) return;

    cstd::array<byte_t, 4> key = {byte_t{0xA1}, byte_t{0xA2},
                                   byte_t{0xA3}, byte_t{0xA4}};
    if (!store.DeleteBlob(cstd::span<const byte_t>(key.data(), key.size()))) {
        d_result->status = -1;
        return;
    }
    d_result->status = 1;
}

// Run a single kernel launch that invokes CTE tasks (so the orchestrator must
// run while the kernel is in flight). Polls a pinned status word instead of
// cudaStreamSynchronize, which would deadlock against the persistent CDP
// orchestrator. Mirrors RunGpuBlobTest in gpu_cte_blob_store_test.cu.
static int LaunchStoreKernel(
    void (*kernel)(chi::IpcManagerGpu, kvhdf5::GpuCteBlobStore,
                   StoreCrossResult *),
    kvhdf5::GpuCteBlobStore store) {
    auto *gpu_ipc = CHI_CPU_IPC->GetGpuIpcManager();
    chi::IpcManagerGpuInfo gpu_info = gpu_ipc->GetClientGpuInfo(0);

    gpu_ipc->PauseGpuOrchestrator();

    volatile StoreCrossResult *d_result = nullptr;
    if (cudaMallocHost(const_cast<StoreCrossResult **>(&d_result),
                       sizeof(StoreCrossResult)) != cudaSuccess) {
        gpu_ipc->ResumeGpuOrchestrator();
        return -100;
    }
    d_result->status = 0;

    cudaGetLastError();
    void *stream = hshm::GpuApi::CreateStream();
    kernel<<<1, 32, 0, static_cast<cudaStream_t>(stream)>>>(
        gpu_info, store, const_cast<StoreCrossResult *>(d_result));

    if (cudaGetLastError() != cudaSuccess) {
        hshm::GpuApi::DestroyStream(stream);
        cudaFreeHost(const_cast<StoreCrossResult *>(d_result));
        gpu_ipc->ResumeGpuOrchestrator();
        return -201;
    }

    gpu_ipc->ResumeGpuOrchestrator();

    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (d_result->status == 0
           && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    int status = d_result->status == 0 ? -300 : d_result->status;

    gpu_ipc->PauseGpuOrchestrator();
    cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
    hshm::GpuApi::DestroyStream(stream);
    cudaFreeHost(const_cast<StoreCrossResult *>(d_result));
    gpu_ipc->ResumeGpuOrchestrator();
    return status;
}

// ---- Catch2 test cases (host-only, hidden from device compilation pass) ------

#if !HSHM_IS_GPU
TEST_CASE("GPU backend init and AllocateBuffer work from kernel",
          "[integration][iowarp][cte_gpu][gpu_kernel]") {
    EnsureGpuCteRuntime();
    int rc = run_gpu_alloc_test();
    REQUIRE(rc == 1);
}

TEST_CASE("Kernel writes UVM pattern, host PutBlob succeeds",
          "[integration][iowarp][cte_gpu][gpu_kernel]") {
    EnsureGpuCteRuntime();
    int rc = run_gpu_write_then_host_put_test(g_gpu_cte_pool_id,
                                              g_gpu_pool_tag_id);
    REQUIRE(rc == 1);
}

TEST_CASE("Host PutBlob, host GetBlob, kernel verifies pattern via UVA",
          "[integration][iowarp][cte_gpu][gpu_kernel]") {
    EnsureGpuCteRuntime();
    bool data_match = false;
    int rc = run_host_put_gpu_get_test(g_gpu_cte_pool_id, g_gpu_pool_tag_id,
                                       &data_match);
    REQUIRE(rc == 1);
    REQUIRE(data_match);
}

// Cross-boundary: kernel PutBlob via GpuCteBlobStore -> host GetBlob via the
// same store. The kernel-side dual-send (PoolQuery::Local + ToLocalCpu) must
// land the value in CPU CTE so the host's CHI_CPU_IPC GetBlob succeeds.
TEST_CASE("Kernel PutBlob -> host GetBlob via GpuCteBlobStore (kernel dual-send)",
          "[integration][iowarp][cte_gpu][gpu_kernel]") {
    EnsureGpuCteRuntime();

    auto tag_task = g_gpu_cte_client->AsyncGetOrCreateTag(
        "gpu_kernel_put_host_get");
    tag_task.Wait();
    wrp_cte::core::TagId tag_id = tag_task->tag_id_;

    kvhdf5::GpuCteBlobStore store =
        kvhdf5::GpuCteBlobStore::Create(tag_id, g_gpu_cte_pool_id);
    REQUIRE(store.IsValid());

    int rc = LaunchStoreKernel(kernel_store_put_pattern, store);
    INFO("kernel status: " << rc);
    REQUIRE(rc == 1);

    cstd::array<byte_t, 4> key = {byte_t{0x91}, byte_t{0x92},
                                   byte_t{0x93}, byte_t{0x94}};
    cstd::array<byte_t, 8> expected = {
        byte_t{0xCA}, byte_t{0xFE}, byte_t{0xBA}, byte_t{0xBE},
        byte_t{0x01}, byte_t{0x02}, byte_t{0x03}, byte_t{0x04}
    };

    cstd::array<byte_t, 8> output{};
    auto get_result = store.GetBlob(
        cstd::span<const byte_t>(key.data(), key.size()),
        cstd::span<byte_t>(output.data(), output.size()));
    REQUIRE(get_result.has_value());
    REQUIRE(get_result->size() == expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        REQUIRE(output[i] == expected[i]);
    }

    store.Destroy();
}

// Cross-boundary: kernel DeleteBlob via GpuCteBlobStore -> host Exists==false.
// Host Put first (host dual-send populates both CTE runtimes), kernel Delete
// (kernel dual-send writes the tombstone to both), then host Exists must see
// the tombstone in CPU CTE.
TEST_CASE("Kernel DeleteBlob -> host Exists==false via GpuCteBlobStore",
          "[integration][iowarp][cte_gpu][gpu_kernel]") {
    EnsureGpuCteRuntime();

    auto tag_task = g_gpu_cte_client->AsyncGetOrCreateTag(
        "gpu_kernel_delete_host_exists");
    tag_task.Wait();
    wrp_cte::core::TagId tag_id = tag_task->tag_id_;

    kvhdf5::GpuCteBlobStore store =
        kvhdf5::GpuCteBlobStore::Create(tag_id, g_gpu_cte_pool_id);
    REQUIRE(store.IsValid());

    cstd::array<byte_t, 4> key = {byte_t{0xA1}, byte_t{0xA2},
                                   byte_t{0xA3}, byte_t{0xA4}};
    cstd::array<byte_t, 4> value = {byte_t{0x55}, byte_t{0x66},
                                     byte_t{0x77}, byte_t{0x88}};

    REQUIRE(store.PutBlob(
        cstd::span<const byte_t>(key.data(), key.size()),
        cstd::span<const byte_t>(value.data(), value.size())));
    REQUIRE(store.Exists(cstd::span<const byte_t>(key.data(), key.size())));

    int rc = LaunchStoreKernel(kernel_store_delete_key, store);
    INFO("kernel status: " << rc);
    REQUIRE(rc == 1);

    REQUIRE_FALSE(
        store.Exists(cstd::span<const byte_t>(key.data(), key.size())));

    cstd::array<byte_t, 4> output{};
    auto get_result = store.GetBlob(
        cstd::span<const byte_t>(key.data(), key.size()),
        cstd::span<byte_t>(output.data(), output.size()));
    REQUIRE_FALSE(get_result.has_value());
    REQUIRE(get_result.error() == kvhdf5::BlobStoreError::NotExist);

    store.Destroy();
}
#endif  // !HSHM_IS_GPU

#endif  // HSHM_ENABLE_CUDA
