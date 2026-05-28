#pragma once

// Iowarp Gray-Scott step kernel + host helpers, factored out so both the
// example and the benchmark fixture can call them. Behavior is unchanged
// from the original kernel and LaunchAndPoll that lived directly in
// gray_scott_gpu.cu — only the location moved.

#include "kvhdf5/container.h"
#include "kvhdf5/gpu_cte_blob_store.h"
#include "kvhdf5/hdf5_dataset.h"

#include <wrp_cte/core/core_client.h>
#include <chimaera/chimaera.h>
#include "hermes_shm/memory/backend/array_backend.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>

namespace gs_iowarp {

namespace cfg {
inline constexpr unsigned kN              = 32;
inline constexpr size_t   kCellsPerGrid   = static_cast<size_t>(kN) * kN;
inline constexpr size_t   kBytesPerGrid   = kCellsPerGrid * sizeof(float);
} // namespace cfg

struct GrayScottParams {
    float Du;
    float Dv;
    float F;
    float k;
    float dt;
};

// Step kernel — definition lives in iowarp_step.cu. Single-thread (<<<1,1>>>)
// per grid; reads u_curr/v_curr through Dataset<>, writes u_next/v_next.
__global__ void IowarpKernelGrayScottStep(
    chi::IpcManagerGpuInfo gpu_info,
    kvhdf5::Container<kvhdf5::GpuCteBlobStore>* container,
    kvhdf5::DatasetId u_curr, kvhdf5::DatasetId v_curr,
    kvhdf5::DatasetId u_next, kvhdf5::DatasetId v_next,
    GrayScottParams params,
    int* d_status);

#if !HSHM_IS_GPU

// Host launcher: pause orchestrator, alloc pinned status word, launch,
// resume, poll status word. Polling (instead of cudaStreamSynchronize) is
// required because the persistent CDP orchestrator deadlocks against
// device-syncing host APIs.
//
// fn is a callable taking (IpcManagerGpuInfo, cudaStream_t, int* d_status).
template <typename Fn>
inline int LaunchAndPoll(Fn&& fn) {
    auto* gpu_ipc = CHI_CPU_IPC->GetGpuIpcManager();
    chi::IpcManagerGpuInfo gpu_info = gpu_ipc->GetClientGpuInfo(0);

    gpu_ipc->PauseGpuOrchestrator();

    volatile int* d_status = nullptr;
    if (cudaMallocHost(const_cast<int**>(&d_status), sizeof(int))
            != cudaSuccess) {
        gpu_ipc->ResumeGpuOrchestrator();
        return -100;
    }
    *d_status = 0;

    cudaGetLastError();
    void* stream_v = hshm::GpuApi::CreateStream();
    auto  stream   = static_cast<cudaStream_t>(stream_v);

    fn(gpu_info, stream, const_cast<int*>(d_status));

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        std::fprintf(stderr, "kernel launch failed: %s (cuda %d)\n",
                     cudaGetErrorString(launch_err),
                     static_cast<int>(launch_err));
        hshm::GpuApi::DestroyStream(stream_v);
        cudaFreeHost(const_cast<int*>(d_status));
        gpu_ipc->ResumeGpuOrchestrator();
        return -201;
    }

    gpu_ipc->ResumeGpuOrchestrator();

    auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (*d_status == 0
           && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    int status = (*d_status == 0) ? -300 : *d_status;

    gpu_ipc->PauseGpuOrchestrator();
    cudaStreamSynchronize(stream);
    hshm::GpuApi::DestroyStream(stream_v);
    cudaFreeHost(const_cast<int*>(d_status));
    gpu_ipc->ResumeGpuOrchestrator();
    return status;
}

// Managed-memory allocator + container glue, lifted verbatim from
// gray_scott_gpu.cu. Lives here so the bench fixture can reuse them.
struct ManagedAllocBox {
    static constexpr size_t kHeapSize = 1ULL * 1024 * 1024;  // 1 MiB

    char*                    memory    = nullptr;
    hshm::ipc::ArrayBackend  backend;
    kvhdf5::AllocatorImpl*   allocator = nullptr;

    bool Setup() {
        size_t total = kHeapSize + 3 * hshm::ipc::kBackendHeaderSize;
        auto* gpu_ipc = CHI_CPU_IPC->GetGpuIpcManager();
        gpu_ipc->PauseGpuOrchestrator();
        cudaError_t err = cudaMallocManaged(
            reinterpret_cast<void**>(&memory), total);
        gpu_ipc->ResumeGpuOrchestrator();
        if (err != cudaSuccess) return false;
        std::memset(memory, 0, total);
        if (!backend.shm_init(hshm::ipc::MemoryBackendId::GetRoot(),
                              total, memory)) return false;
        allocator = backend.MakeAlloc<kvhdf5::AllocatorImpl>();
        return allocator != nullptr;
    }
    void Teardown() {
        if (memory) {
            auto* gpu_ipc = CHI_CPU_IPC->GetGpuIpcManager();
            gpu_ipc->PauseGpuOrchestrator();
            cudaFree(memory);
            gpu_ipc->ResumeGpuOrchestrator();
            memory = nullptr;
        }
    }
};

struct ManagedContainerBox {
    kvhdf5::Container<kvhdf5::GpuCteBlobStore>* ptr = nullptr;
    bool Setup(kvhdf5::GpuCteBlobStore store,
               kvhdf5::AllocatorImpl* alloc) {
        void* raw = nullptr;
        auto* gpu_ipc = CHI_CPU_IPC->GetGpuIpcManager();
        gpu_ipc->PauseGpuOrchestrator();
        cudaError_t err = cudaMallocManaged(
            &raw, sizeof(kvhdf5::Container<kvhdf5::GpuCteBlobStore>));
        gpu_ipc->ResumeGpuOrchestrator();
        if (err != cudaSuccess) return false;
        ptr = new (raw) kvhdf5::Container<kvhdf5::GpuCteBlobStore>(
            std::move(store), alloc);
        return ptr != nullptr;
    }
    void Teardown() {
        if (ptr) {
            ptr->~Container();
            auto* gpu_ipc = CHI_CPU_IPC->GetGpuIpcManager();
            gpu_ipc->PauseGpuOrchestrator();
            cudaFree(ptr);
            gpu_ipc->ResumeGpuOrchestrator();
            ptr = nullptr;
        }
    }
};

#endif // !HSHM_IS_GPU

} // namespace gs_iowarp
