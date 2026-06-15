#pragma once

// Device-facing handle for the iowarp GPU producer path.
//
// A GpuDatasetHandle is a small, by-value POD built on the host by GpuCteDataset
// and passed BY VALUE into the user's compute kernel. It describes a dataset of
// N chunks: the per-chunk Put/Get tasks + data buffers live in a device-resident
// ChunkDesc array (chunks_), and the handle carries only a pointer to it + the
// count. This keeps the kernel-arg size constant regardless of N (inlining N
// FullPtr pairs by value would hit the CUDA kernel-parameter ceiling).
//
// Inside the kernel the user fills Data(c) for a chunk and calls Write(c) (or
// Read(c)) to submit that chunk's pre-built task — the I/O is fused into the
// user's launch instead of being orchestrated from the host. A block can own one
// chunk (chunk_id = blockIdx) or grid-stride over a range of chunks; both index
// the same chunks_ array, so one-block-per-chunk is just the gridDim >= N case of
// the grid-stride form. The no-arg Data()/Write()/... overloads target chunk 0
// (the single-chunk specialization).
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

// One element of the device-resident per-chunk array. Built on the host
// (GpuCteDataset stamps the prototypes and fills these), copied to the device
// once, and read by the kernel. The FullPtrs address that chunk's pre-built
// PutBlob/GetBlob task slots; data/size address its distinct device buffer.
struct ChunkDesc {
    ctp::ipc::FullPtr<cte::PutBlobTask> put_fp;
    ctp::ipc::FullPtr<cte::GetBlobTask> get_fp;
    byte_t* data = nullptr;     // this chunk's registered device blob buffer
    uint64_t size = 0;
};

struct GpuDatasetHandle {
    chi::IpcManagerGpuInfo info_;
    ChunkDesc* chunks_ = nullptr;   // device array, count_ entries
    uint32_t count_ = 0;

#if CTP_IS_GPU_COMPILER
    __device__ uint32_t Count() const { return count_; }
    __device__ byte_t* Data(uint32_t c) const { return chunks_[c].data; }
    __device__ uint64_t Size(uint32_t c) const { return chunks_[c].size; }

    // Submit chunk c's pre-built Put/Get task and wait. Thread-0 of the block
    // enqueues; all other threads no-op (iowarp's threadIdx==0 producer guard).
    __device__ void Write(uint32_t c) const { Submit(chunks_[c].put_fp); }
    __device__ void Read(uint32_t c) const { Submit(chunks_[c].get_fp); }

    // Single-chunk convenience: target chunk 0.
    __device__ byte_t* Data() const { return chunks_[0].data; }
    __device__ uint64_t Size() const { return chunks_[0].size; }
    __device__ void Write() const { Submit(chunks_[0].put_fp); }
    __device__ void Read() const { Submit(chunks_[0].get_fp); }

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
// __global__ kernel and it works. ChunkDesc bundles those same proven types and
// is read from a device array (never copy-constructed on device); the handle
// itself stays small (info_ + pointer + count), so the guarantee is the runtime
// round-trip, not a trait.

}  // namespace kvhdf5
