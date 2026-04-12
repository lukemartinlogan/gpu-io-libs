#if HSHM_ENABLE_CUDA

#include <catch2/catch_test_macros.hpp>
#include "gpu_cte_fixture.h"
#include "kvhdf5/gpu_cte_blob_store.h"
#include "kvhdf5/blob_store.h"
#include <cstring>
#include <thread>
#include <chrono>

using namespace kvhdf5;

// ---------------------------------------------------------------------------
// GPU backend setup (uses unique IDs 60/61 to avoid conflicts with
// test_gpu_kernel.cu which uses 50/51)
// ---------------------------------------------------------------------------

struct GpuBlobBackends {
    hipc::GpuShmMmap primary;
    hipc::GpuShmMmap g2c;
    bool ok = false;
};

static GpuBlobBackends SetupGpuBlobBackends() {
    GpuBlobBackends b;

    hipc::MemoryBackendId primary_id(60, 0);
    if (!b.primary.shm_init(primary_id, 10 * 1024 * 1024,
                             "/kvhdf5_gpu_blobstore_primary", 0)) {
        return b;
    }

    hipc::MemoryBackendId g2c_id(61, 0);
    if (!b.g2c.shm_init(g2c_id, 4 * 1024 * 1024,
                         "/kvhdf5_gpu_blobstore_g2c", 0)) {
        return b;
    }

    CHI_CPU_IPC->GetGpuIpcManager()->RegisterGpuAllocator(
        primary_id, b.primary.data_, b.primary.data_capacity_);
    b.ok = true;
    return b;
}

static chi::IpcManagerGpuInfo BuildGpuBlobInfo(GpuBlobBackends &b) {
    chi::IpcManagerGpuInfo info;
    info.backend         = static_cast<hipc::MemoryBackend &>(b.primary);
    info.gpu2cpu_queue   = CHI_CPU_IPC->GetGpuQueue(0);
    info.gpu2cpu_backend = static_cast<hipc::MemoryBackend &>(b.g2c);
    return info;
}

// ---------------------------------------------------------------------------
// Result struct for passing test outcomes from kernel to host
// ---------------------------------------------------------------------------

struct BlobTestResult {
    int status;       // 1=pass, negative=error code
    int data_match;   // 1=data matched, 0=mismatch
};

// ---------------------------------------------------------------------------
// Kernel: PutBlob + GetBlob roundtrip
// ---------------------------------------------------------------------------

__global__ void kernel_put_get_roundtrip(
    chi::IpcManagerGpuInfo gpu_info,
    GpuCteBlobStore store,
    BlobTestResult *d_result)
{
    CHIMAERA_GPU_INIT(gpu_info);
    // Only lane 0 does the blob store calls — iowarp-core pattern.
    if (threadIdx.x != 0) return;

    d_result->status = 0;
    d_result->data_match = 0;

    // Key: 4 bytes
    cstd::array<byte_t, 4> key = {byte_t{1}, byte_t{2}, byte_t{3}, byte_t{4}};
    // Value: 8 bytes with known pattern
    cstd::array<byte_t, 8> value = {
        byte_t{0xDE}, byte_t{0xAD}, byte_t{0xBE}, byte_t{0xEF},
        byte_t{0xCA}, byte_t{0xFE}, byte_t{0xBA}, byte_t{0xBE}
    };

    bool put_ok = store.PutBlob(key, value);
    if (!put_ok) {
        d_result->status = -1;
        return;
    }

    cstd::array<byte_t, 8> output;
    auto result = store.GetBlob(key, output);
    if (!result.has_value()) {
        d_result->status = -2;
        return;
    }

    if (result->size() != 8) {
        d_result->status = -3;
        return;
    }

    // Check data matches
    d_result->data_match = 1;
    for (size_t i = 0; i < 8; ++i) {
        if (output[i] != value[i]) {
            d_result->data_match = 0;
            break;
        }
    }

    d_result->status = 1;
}

// ---------------------------------------------------------------------------
// Kernel: DeleteBlob (sentinel) + Exists
// ---------------------------------------------------------------------------

__global__ void kernel_delete_and_exists(
    chi::IpcManagerGpuInfo gpu_info,
    GpuCteBlobStore store,
    BlobTestResult *d_result)
{
    d_result->status = 0;
    d_result->data_match = 0;
    CHIMAERA_GPU_INIT(gpu_info);

    cstd::array<byte_t, 4> key = {byte_t{10}, byte_t{20}, byte_t{30}, byte_t{40}};
    cstd::array<byte_t, 4> value = {byte_t{1}, byte_t{2}, byte_t{3}, byte_t{4}};

    // Put
    if (!store.PutBlob(key, value)) {
        d_result->status = -1;
        return;
    }

    // Exists should be true
    if (!store.Exists(key)) {
        d_result->status = -2;
        return;
    }

    // Delete (sentinel)
    if (!store.DeleteBlob(key)) {
        d_result->status = -3;
        return;
    }

    // Exists should be false after delete
    if (store.Exists(key)) {
        d_result->status = -4;
        return;
    }

    // GetBlob should return NotExist after delete
    cstd::array<byte_t, 4> output;
    auto result = store.GetBlob(key, output);
    if (result.has_value()) {
        d_result->status = -5;
        return;
    }

    d_result->status = 1;
}

// ---------------------------------------------------------------------------
// Kernel: Multiple keys
// ---------------------------------------------------------------------------

__global__ void kernel_multiple_keys(
    chi::IpcManagerGpuInfo gpu_info,
    GpuCteBlobStore store,
    BlobTestResult *d_result)
{
    d_result->status = 0;
    d_result->data_match = 0;
    CHIMAERA_GPU_INIT(gpu_info);

    cstd::array<byte_t, 4> key1 = {byte_t{1}, byte_t{0}, byte_t{0}, byte_t{0}};
    cstd::array<byte_t, 4> key2 = {byte_t{2}, byte_t{0}, byte_t{0}, byte_t{0}};
    cstd::array<byte_t, 4> key3 = {byte_t{3}, byte_t{0}, byte_t{0}, byte_t{0}};

    cstd::array<byte_t, 4> val1 = {byte_t{0xAA}, byte_t{0xAA}, byte_t{0xAA}, byte_t{0xAA}};
    cstd::array<byte_t, 4> val2 = {byte_t{0xBB}, byte_t{0xBB}, byte_t{0xBB}, byte_t{0xBB}};
    cstd::array<byte_t, 4> val3 = {byte_t{0xCC}, byte_t{0xCC}, byte_t{0xCC}, byte_t{0xCC}};

    if (!store.PutBlob(key1, val1)) { d_result->status = -1; return; }
    if (!store.PutBlob(key2, val2)) { d_result->status = -2; return; }
    if (!store.PutBlob(key3, val3)) { d_result->status = -3; return; }

    // Read back and verify
    cstd::array<byte_t, 4> out;

    auto r1 = store.GetBlob(key1, out);
    if (!r1.has_value()) { d_result->status = -10; return; }
    for (int i = 0; i < 4; ++i) {
        if (out[i] != val1[i]) { d_result->status = -11; return; }
    }

    auto r2 = store.GetBlob(key2, out);
    if (!r2.has_value()) { d_result->status = -12; return; }
    for (int i = 0; i < 4; ++i) {
        if (out[i] != val2[i]) { d_result->status = -13; return; }
    }

    auto r3 = store.GetBlob(key3, out);
    if (!r3.has_value()) { d_result->status = -14; return; }
    for (int i = 0; i < 4; ++i) {
        if (out[i] != val3[i]) { d_result->status = -15; return; }
    }

    d_result->status = 1;
    d_result->data_match = 1;
}

// ---------------------------------------------------------------------------
// Helper: launch a test kernel and return the result
// ---------------------------------------------------------------------------

static BlobTestResult RunGpuBlobTest(
    void (*kernel)(chi::IpcManagerGpuInfo, GpuCteBlobStore, BlobTestResult*),
    GpuCteBlobStore store)
{
    // Use the runtime-managed GPU info. Kernels here call store.PutBlob/
    // GetBlob which submits CTE tasks, so the orchestrator must run
    // concurrently with the kernel.
    //
    // Mimics the iowarp-core test_gpu_initiated_gpu.cc pattern: pause
    // orchestrator → allocate pinned result → launch kernel → resume
    // orchestrator → POLL the pinned memory (do NOT cudaStreamSynchronize,
    // which blocks on the persistent kernel).
    auto *gpu_ipc = CHI_CPU_IPC->GetGpuIpcManager();
    chi::IpcManagerGpuInfo gpu_info = gpu_ipc->GetClientGpuInfo(0);

    gpu_ipc->PauseGpuOrchestrator();

    // Pinned result struct — kernel writes status=1 when done, CPU polls it
    volatile BlobTestResult *d_result;
    cudaMallocHost(const_cast<BlobTestResult **>(&d_result),
                   sizeof(BlobTestResult));
    d_result->status = 0;
    d_result->data_match = 0;

    cudaGetLastError();
    void *stream = hshm::GpuApi::CreateStream();
    // Launch a full warp (32 threads) — matches iowarp-core test pattern;
    // the GPU IPC layer assumes warp-level execution for some operations.
    kernel<<<1, 32, 0, static_cast<cudaStream_t>(stream)>>>(
        gpu_info, store, const_cast<BlobTestResult *>(d_result));

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        gpu_ipc->ResumeGpuOrchestrator();
        hshm::GpuApi::DestroyStream(stream);
        return {-201, 0};
    }

    // Resume orchestrator AFTER kernel launch — it services the CTE tasks
    // the kernel submits.
    gpu_ipc->ResumeGpuOrchestrator();

    // Poll pinned memory until kernel signals completion (status != 0) or
    // timeout. cudaStreamSynchronize would deadlock against the persistent
    // orchestrator kernel.
    constexpr int kTimeoutUs = 30 * 1000 * 1000;  // 30 s
    int elapsed_us = 0;
    while (d_result->status == 0 && elapsed_us < kTimeoutUs) {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
        elapsed_us += 100;
    }

    BlobTestResult result{d_result->status, d_result->data_match};
    if (result.status == 0) {
        result.status = -300;  // timeout sentinel
    }

    // Intentionally leak the pinned buffer and stream: cudaFreeHost and
    // cudaStreamDestroy can block on the persistent orchestrator kernel.
    return result;
}

// ---------------------------------------------------------------------------
// Catch2 tests (host-only, hidden from device compilation pass)
// ---------------------------------------------------------------------------

#if !HSHM_IS_GPU

TEST_CASE("GpuCteBlobStore - PutBlob + GetBlob roundtrip from kernel",
          "[unit][gpu_cte][blob_store]") {
    EnsureGpuCteRuntime();
    chi::PoolId pool_id = wrp_cte::core::kCtePoolId;

    auto tag_task = WRP_CTE_CLIENT->AsyncGetOrCreateTag("gpu_blob_store_roundtrip");
    tag_task.Wait();
    wrp_cte::core::TagId tag_id = tag_task->tag_id_;

    GpuCteBlobStore store = GpuCteBlobStore::Create(tag_id, pool_id);
    REQUIRE(store.IsValid());
    auto result = RunGpuBlobTest(kernel_put_get_roundtrip, store);
    store.Destroy();

    INFO("kernel status: " << result.status);
    REQUIRE(result.status == 1);
    REQUIRE(result.data_match == 1);
}

TEST_CASE("GpuCteBlobStore - DeleteBlob sentinel + Exists from kernel",
          "[unit][gpu_cte][blob_store]") {
    EnsureGpuCteRuntime();
    chi::PoolId pool_id = wrp_cte::core::kCtePoolId;

    auto tag_task = WRP_CTE_CLIENT->AsyncGetOrCreateTag("gpu_blob_store_delete");
    tag_task.Wait();
    wrp_cte::core::TagId tag_id = tag_task->tag_id_;

    GpuCteBlobStore store = GpuCteBlobStore::Create(tag_id, pool_id);
    REQUIRE(store.IsValid());
    auto result = RunGpuBlobTest(kernel_delete_and_exists, store);
    store.Destroy();

    INFO("kernel status: " << result.status);
    REQUIRE(result.status == 1);
}

TEST_CASE("GpuCteBlobStore - Multiple keys from kernel",
          "[unit][gpu_cte][blob_store]") {
    EnsureGpuCteRuntime();
    chi::PoolId pool_id = wrp_cte::core::kCtePoolId;

    auto tag_task = WRP_CTE_CLIENT->AsyncGetOrCreateTag("gpu_blob_store_multi");
    tag_task.Wait();
    wrp_cte::core::TagId tag_id = tag_task->tag_id_;

    GpuCteBlobStore store = GpuCteBlobStore::Create(tag_id, pool_id);
    REQUIRE(store.IsValid());
    auto result = RunGpuBlobTest(kernel_multiple_keys, store);
    store.Destroy();

    INFO("kernel status: " << result.status);
    REQUIRE(result.status == 1);
    REQUIRE(result.data_match == 1);
}

#endif  // !HSHM_IS_GPU

#endif  // HSHM_ENABLE_CUDA
