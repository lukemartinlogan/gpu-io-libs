#if HSHM_ENABLE_CUDA

#include <catch2/catch_test_macros.hpp>
#include "gpu_cte_fixture.h"
#include <cstring>

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

    CHI_IPC->RegisterGpuAllocator(primary_id,
                                   b.primary.data_,
                                   b.primary.data_capacity_);
    b.ok = true;
    return b;
}

static chi::IpcManagerGpuInfo BuildGpuInfo(GpuBackends &b) {
    chi::IpcManagerGpuInfo info;
    info.backend         = static_cast<hipc::MemoryBackend &>(b.primary);
    info.gpu2cpu_queue   = CHI_IPC->GetGpuQueue(0);
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
    GpuBackends b = SetupGpuBackends();
    if (!b.ok) return -100;

    chi::IpcManagerGpuInfo gpu_info = BuildGpuInfo(b);

    int *d_result = hshm::GpuApi::Malloc<int>(sizeof(int));
    int h_result = 0;
    hshm::GpuApi::Memcpy(d_result, &h_result, sizeof(int));

    cudaGetLastError();
    void *stream = hshm::GpuApi::CreateStream();
    kernel_alloc_and_fill<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(
        gpu_info, kKernelBlobSize, d_result);

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        hshm::GpuApi::Free(d_result);
        hshm::GpuApi::DestroyStream(stream);
        return -201;
    }

    hshm::GpuApi::Synchronize(stream);
    cudaError_t sync_err = cudaGetLastError();
    if (sync_err != cudaSuccess) {
        hshm::GpuApi::Free(d_result);
        hshm::GpuApi::DestroyStream(stream);
        return -200;
    }

    hshm::GpuApi::Memcpy(&h_result, d_result, sizeof(int));
    hshm::GpuApi::Free(d_result);
    hshm::GpuApi::DestroyStream(stream);
    return h_result;
}

// ---- Kernel 2: GPU writes pattern into UVM buffer, host verifies via CPU put -
//
// The kernel allocates a UVM buffer (accessible from both CPU and GPU), writes
// a pattern, then returns the ShmPtr to the host via a pinned output slot.
// The host wrapper then issues the PutBlob and GetBlob calls.

struct KernelWriteResult {
    int     status;      // 1=ok, negative=error
    hipc::ShmPtr<> shm;  // ShmPtr to the UVM buffer filled by the kernel
};

__global__ void kernel_write_uvm_pattern(chi::IpcManagerGpu gpu_info,
                                          size_t blob_size,
                                          KernelWriteResult *d_out) {
    d_out->status = 0;
    CHIMAERA_GPU_INIT(gpu_info);

    hipc::FullPtr<char> buf = CHI_IPC->AllocateBuffer(blob_size);
    if (buf.IsNull()) {
        d_out->status = -10;
        return;
    }

    // pattern: byte[i] = i % 251
    for (size_t i = 0; i < blob_size; ++i) {
        buf.ptr_[i] = static_cast<char>(i % 251);
    }

    // expose the ShmPtr so the host can issue AsyncPutBlob with it
    d_out->shm    = buf.shm_.template Cast<void>();
    d_out->status = 1;
}

extern "C" int run_gpu_write_then_host_put_test(chi::PoolId pool_id,
                                                 wrp_cte::core::TagId tag_id) {
    GpuBackends b = SetupGpuBackends();
    if (!b.ok) return -100;

    chi::IpcManagerGpuInfo gpu_info = BuildGpuInfo(b);

    // use pinned host memory so CPU can read the result without device sync
    KernelWriteResult *d_out;
    cudaMallocHost(&d_out, sizeof(KernelWriteResult));
    d_out->status = 0;
    d_out->shm    = hipc::ShmPtr<>::GetNull();

    cudaGetLastError();
    void *stream = hshm::GpuApi::CreateStream();
    kernel_write_uvm_pattern<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(
        gpu_info, kKernelBlobSize, d_out);

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        cudaFreeHost(d_out);
        hshm::GpuApi::DestroyStream(stream);
        return -201;
    }

    hshm::GpuApi::Synchronize(stream);
    cudaError_t sync_err = cudaGetLastError();
    if (sync_err != cudaSuccess) {
        cudaFreeHost(d_out);
        hshm::GpuApi::DestroyStream(stream);
        return -200;
    }

    if (d_out->status != 1) {
        int s = d_out->status;
        cudaFreeHost(d_out);
        hshm::GpuApi::DestroyStream(stream);
        return s;
    }

    // issue AsyncPutBlob from host using the UVM ShmPtr the kernel filled
    auto put_task = WRP_CTE_CLIENT->AsyncPutBlob(
        tag_id,
        "gpu_write_host_put_blob",
        0, kKernelBlobSize,
        d_out->shm,
        1.0f,
        wrp_cte::core::Context(),
        0,
        chi::PoolQuery::Local());
    put_task.Wait();
    int rc = static_cast<int>(put_task->GetReturnCode());

    cudaFreeHost(d_out);
    hshm::GpuApi::DestroyStream(stream);
    return (rc == 0) ? 1 : -30;
}

// ---- Kernel 3: Host puts data, GPU reads it back via UVM ShmPtr ---------------
//
// The host issues a PutBlob, then launches a kernel that allocates a UVM
// buffer (accessible to GetBlob on CPU), issues the read via a second
// host-side AsyncGetBlob on the same ShmPtr, and verifies the pattern.

__global__ void kernel_fill_recv_buffer(chi::IpcManagerGpu gpu_info,
                                         size_t blob_size,
                                         KernelWriteResult *d_out) {
    d_out->status = 0;
    CHIMAERA_GPU_INIT(gpu_info);

    hipc::FullPtr<char> buf = CHI_IPC->AllocateBuffer(blob_size);
    if (buf.IsNull()) {
        d_out->status = -10;
        return;
    }
    // zero out so a stale pattern doesn't fool the host-side check
    for (size_t i = 0; i < blob_size; ++i) {
        buf.ptr_[i] = 0;
    }

    d_out->shm    = buf.shm_.template Cast<void>();
    d_out->status = 1;
}

extern "C" int run_host_put_gpu_get_test(chi::PoolId pool_id,
                                          wrp_cte::core::TagId tag_id,
                                          bool *data_match_out) {
    *data_match_out = false;

    // pre-populate the blob from CPU with a known pattern
    {
        auto put_buf = CHI_IPC->AllocateBuffer(kKernelBlobSize);
        if (put_buf.IsNull()) return -101;
        for (size_t i = 0; i < kKernelBlobSize; ++i) {
            put_buf.ptr_[i] = static_cast<char>(i % 251);
        }
        hipc::ShmPtr<> put_shm = put_buf.shm_.template Cast<void>();
        auto put_task = WRP_CTE_CLIENT->AsyncPutBlob(
            tag_id, "host_put_gpu_get_blob",
            0, kKernelBlobSize, put_shm,
            1.0f, wrp_cte::core::Context(), 0,
            chi::PoolQuery::Local());
        put_task.Wait();
        CHI_IPC->FreeBuffer(put_buf);
        if (put_task->GetReturnCode() != 0) return -102;
    }

    GpuBackends b = SetupGpuBackends();
    if (!b.ok) return -100;

    chi::IpcManagerGpuInfo gpu_info = BuildGpuInfo(b);

    // launch kernel to allocate a UVM receive buffer
    KernelWriteResult *d_out;
    cudaMallocHost(&d_out, sizeof(KernelWriteResult));
    d_out->status = 0;
    d_out->shm    = hipc::ShmPtr<>::GetNull();

    cudaGetLastError();
    void *stream = hshm::GpuApi::CreateStream();
    kernel_fill_recv_buffer<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(
        gpu_info, kKernelBlobSize, d_out);

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        cudaFreeHost(d_out);
        hshm::GpuApi::DestroyStream(stream);
        return -201;
    }

    hshm::GpuApi::Synchronize(stream);
    cudaError_t sync_err = cudaGetLastError();
    if (sync_err != cudaSuccess) {
        cudaFreeHost(d_out);
        hshm::GpuApi::DestroyStream(stream);
        return -200;
    }

    if (d_out->status != 1) {
        int s = d_out->status;
        cudaFreeHost(d_out);
        hshm::GpuApi::DestroyStream(stream);
        return s;
    }

    // host issues AsyncGetBlob into the UVM buffer the kernel allocated
    auto get_task = WRP_CTE_CLIENT->AsyncGetBlob(
        tag_id, "host_put_gpu_get_blob",
        0, kKernelBlobSize, 0,
        d_out->shm,
        chi::PoolQuery::Local());
    get_task.Wait();
    int rc = static_cast<int>(get_task->GetReturnCode());
    if (rc != 0) {
        cudaFreeHost(d_out);
        hshm::GpuApi::DestroyStream(stream);
        return -30;
    }

    // resolve the ShmPtr so we can read back the data
    auto full_ptr = CHI_IPC->ToFullPtr<char>(d_out->shm.template Cast<char>());
    bool match = true;
    for (size_t i = 0; i < kKernelBlobSize; ++i) {
        if (static_cast<unsigned char>(full_ptr.ptr_[i]) !=
            static_cast<unsigned char>(i % 251)) {
            match = false;
            break;
        }
    }
    *data_match_out = match;

    cudaFreeHost(d_out);
    hshm::GpuApi::DestroyStream(stream);
    return 1;
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
    chi::PoolId pool_id = wrp_cte::core::kCtePoolId;
    int rc = run_gpu_write_then_host_put_test(pool_id, g_gpu_cte_tag_id);
    REQUIRE(rc == 1);
}

TEST_CASE("Host PutBlob, kernel allocates receive buffer, host GetBlob into it",
          "[integration][iowarp][cte_gpu][gpu_kernel]") {
    EnsureGpuCteRuntime();
    chi::PoolId pool_id = wrp_cte::core::kCtePoolId;
    bool data_match = false;
    int rc = run_host_put_gpu_get_test(pool_id, g_gpu_cte_tag_id, &data_match);
    REQUIRE(rc == 1);
    REQUIRE(data_match);
}
#endif  // !HSHM_IS_GPU

#endif  // HSHM_ENABLE_CUDA
