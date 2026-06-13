#pragma once

// Device-facing handle for the iowarp GPU producer path.
//
// A GpuDatasetHandle is a trivially-copyable POD built on the host by
// GpuCteDataset and passed BY VALUE into the user's compute kernel. Inside the
// kernel the user fills Data() and then calls Write() (or Read()) to submit the
// pre-built PutBlob/GetBlob task — the I/O is fused into the user's launch
// instead of being orchestrated from the host.
//
// Contract: the kernel MUST run CHIMAERA_GPU_INIT(handle.info_, nullptr) at
// block scope before calling Write()/Read(). That macro declares a *kernel-
// local* g_ipc_manager_ptr and does the block-wide ClientInitGpu + __syncthreads
// — so Write()/Read() can't see it and instead re-fetch the per-block IpcManager
// via GetBlockIpcManager() (the same accessor the macro uses).

#include "../defines.h"

#include <clio_runtime/types.h>
#include <clio_runtime/gpu/future.h>
#include <clio_runtime/gpu/gpu_info.h>
#include <clio_runtime/gpu/gpu_ipc_manager.h>
#include <clio_cte/core/core_tasks.h>

#include <cstdint>

namespace kvhdf5 {

namespace cte = clio::cte::core;

struct GpuDatasetHandle {
    chi::IpcManagerGpuInfo info_;
    ctp::ipc::FullPtr<cte::PutBlobTask> put_fp_;
    ctp::ipc::FullPtr<cte::GetBlobTask> get_fp_;
    byte_t* data_ = nullptr;     // registered device blob buffer
    uint64_t size_ = 0;

#if CTP_IS_GPU_COMPILER
    __device__ byte_t* Data() const { return data_; }
    __device__ uint64_t Size() const { return size_; }

    // Submit the pre-built PutBlob task and wait. Thread-0 of the block enqueues;
    // all other threads no-op (mirrors iowarp's threadIdx==0 producer guard).
    __device__ void Write() const { Submit(put_fp_); }
    __device__ void Read() const { Submit(get_fp_); }

private:
    template<typename TaskT>
    __device__ void Submit(const ctp::ipc::FullPtr<TaskT>& fp) const {
        auto* ipc = chi::gpu::IpcManager::GetBlockIpcManager();
        if (chi::gpu::IpcManager::GetGpuThreadId() != 0) return;
        auto fut = ipc->Send(fp);
        fut.Wait();
    }
#endif  // CTP_IS_GPU_COMPILER
};

// No is_trivially_copyable / serde::IsPOD static_assert here on purpose: iowarp's
// FullPtr<T> declares user-provided copy/move ctors, so neither trait holds — yet
// the reference passes FullPtr and IpcManagerGpuInfo BY VALUE straight into a
// __global__ kernel and it works. This handle just bundles those same proven
// kernel-arg types, so the guarantee is the runtime round-trip, not a trait.

}  // namespace kvhdf5
