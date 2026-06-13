#pragma once

// Host control plane for one GPU-resident, single-chunk dataset on the new
// iowarp producer-only CTE model. Lifts the proven mechanics of the integration
// reference (test_cte_devmem_putget): preallocate registered kDeviceMem backends
// for the PutBlob/GetBlob tasks (+ co-located FutureShm) and the blob data
// buffer, stamp the task prototypes onto the device once, and hand out a
// trivially-copyable GpuDatasetHandle the user's kernel submits.
//
// This is NOT a BlobBackend — its surface is Handle(), not WriteChunk/ReadChunk.
// It owns two device allocations, so it is move-only and frees them in the dtor.

#include "../defines.h"
#include "chunking.h"
#include "gpu_dataset_handle.h"

#include <clio_runtime/ipc_manager.h>
#include <clio_runtime/types.h>
#include <clio_runtime/gpu/gpu_info.h>
#include <clio_runtime/gpu/gpu_ipc_manager.h>
#include <clio_runtime/gpu/future.h>
#include <clio_ctp/util/gpu_api.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>

#include <cstring>
#include <new>
#include <stdexcept>

// Host-only control plane: guarded out of the nvcc device pass (kernels need
// only GpuDatasetHandle, included above). Mirrors how the reference guards its
// host bring-up class.
#if !CTP_IS_DEVICE_PASS

namespace kvhdf5 {

class GpuCteDataset {
    chi::IpcManager* ipc_ = nullptr;
    uint32_t gpu_id_ = 0;

    ctp::ipc::AllocatorId task_alloc_{};  // task slots + co-located futures
    ctp::ipc::AllocatorId data_alloc_{};  // blob data buffer
    byte_t* task_base_ = nullptr;
    byte_t* data_base_ = nullptr;
    uint64_t bytes_ = 0;

    GpuDatasetHandle handle_{};

    static constexpr uint32_t kPutSlot =
        sizeof(cte::PutBlobTask) + sizeof(chi::gpu::FutureShm);
    static constexpr uint32_t kGetSlot =
        sizeof(cte::GetBlobTask) + sizeof(chi::gpu::FutureShm);

    using MemKind = chi::gpu::IpcManager::MemKind;

public:
    // `name` must be a NUL-terminated chunk-blob name (<= kMaxBlobNameLen);
    // `bytes` is the chunk's raw byte count. Throws on any iowarp failure.
    GpuCteDataset(chi::IpcManager* ipc, chi::IpcManagerGpuInfo info,
                  uint32_t gpu_id, cte::TagId tag, const char* name,
                  uint64_t bytes)
        : ipc_(ipc), gpu_id_(gpu_id), bytes_(bytes) {
        char name_buf[chunking::kMaxBlobNameLen + 1];
        std::strncpy(name_buf, name, sizeof(name_buf));
        if (name_buf[chunking::kMaxBlobNameLen] != '\0')
            throw std::runtime_error("GpuCteDataset: blob name too long");

        // iowarp hands back the registered base as char*; we keep raw-byte
        // buffers as byte_t* (the codebase convention; names stay char*).
        char* task_base = nullptr;
        const uint32_t task_bytes = kPutSlot + kGetSlot + 64;
        task_alloc_ = ipc_->AllocateAndRegisterGpuBackend(
            gpu_id_, MemKind::kDeviceMem, task_bytes, &task_base);
        task_base_ = reinterpret_cast<byte_t*>(task_base);
        if (task_alloc_.IsNull() || task_base_ == nullptr)
            throw std::runtime_error("GpuCteDataset: task backend alloc failed");

        char* data_base = nullptr;
        data_alloc_ = ipc_->AllocateAndRegisterGpuBackend(
            gpu_id_, MemKind::kDeviceMem, bytes_, &data_base);
        data_base_ = reinterpret_cast<byte_t*>(data_base);
        if (data_alloc_.IsNull() || data_base_ == nullptr) {
            ipc_->FreeGpuBackend(gpu_id_, task_alloc_);
            throw std::runtime_error("GpuCteDataset: data backend alloc failed");
        }

        StampTasks(tag, name_buf);
        handle_ = {info, MakeFullPtr<cte::PutBlobTask>(task_base_),
                   MakeFullPtr<cte::GetBlobTask>(task_base_ + kPutSlot),
                   data_base_, bytes_};
    }

    ~GpuCteDataset() { Free(); }

    GpuCteDataset(const GpuCteDataset&) = delete;
    GpuCteDataset& operator=(const GpuCteDataset&) = delete;

    GpuCteDataset(GpuCteDataset&& o) noexcept { MoveFrom(o); }
    GpuCteDataset& operator=(GpuCteDataset&& o) noexcept {
        if (this != &o) { Free(); MoveFrom(o); }
        return *this;
    }

    [[nodiscard]] GpuDatasetHandle Handle() const { return handle_; }
    [[nodiscard]] byte_t* DeviceData() const { return data_base_; }
    [[nodiscard]] uint64_t Bytes() const { return bytes_; }

private:
    // Placement-new each task prototype (+ its FutureShm) on the host and copy
    // it into the registered device slot. shm.off_ carries the raw device data
    // pointer with a null alloc_id (the kernel reads it as an absolute address).
    void StampTasks(cte::TagId tag, const char* name) {
        ctp::ipc::ShmPtr<> blob_shm;
        blob_shm.alloc_id_.SetNull();
        blob_shm.off_ = reinterpret_cast<chi::u64>(data_base_);

        alignas(64) byte_t put_proto[kPutSlot];
        std::memset(put_proto, 0, sizeof(put_proto));
        auto* put = new (put_proto) cte::PutBlobTask(
            chi::CreateTaskId(), cte::kCtePoolId, chi::PoolQuery::ToLocalCpu(),
            tag, name, /*offset=*/chi::u64(0), bytes_, blob_shm,
            /*score=*/-1.0f, cte::Context(), /*flags=*/chi::u32(0));
        put->pod_size_ = sizeof(cte::PutBlobTask);
        new (put_proto + sizeof(cte::PutBlobTask)) chi::gpu::FutureShm();
        ctp::GpuApi::Memcpy(task_base_, put_proto, sizeof(put_proto));

        alignas(64) byte_t get_proto[kGetSlot];
        std::memset(get_proto, 0, sizeof(get_proto));
        auto* get = new (get_proto) cte::GetBlobTask(
            chi::CreateTaskId(), cte::kCtePoolId, chi::PoolQuery::ToLocalCpu(),
            tag, name, /*offset=*/chi::u64(0), bytes_, /*flags=*/chi::u32(0),
            blob_shm);
        get->pod_size_ = sizeof(cte::GetBlobTask);
        new (get_proto + sizeof(cte::GetBlobTask)) chi::gpu::FutureShm();
        ctp::GpuApi::Memcpy(task_base_ + kPutSlot, get_proto, sizeof(get_proto));
    }

    template<typename TaskT>
    static ctp::ipc::FullPtr<TaskT> MakeFullPtr(byte_t* device_addr) {
        ctp::ipc::FullPtr<TaskT> fp;
        fp.shm_.alloc_id_.SetNull();
        fp.shm_.off_ = reinterpret_cast<chi::u64>(device_addr);
        fp.ptr_ = reinterpret_cast<TaskT*>(device_addr);
        return fp;
    }

    void Free() {
        if (!data_alloc_.IsNull()) ipc_->FreeGpuBackend(gpu_id_, data_alloc_);
        if (!task_alloc_.IsNull()) ipc_->FreeGpuBackend(gpu_id_, task_alloc_);
    }

    void MoveFrom(GpuCteDataset& o) {
        ipc_ = o.ipc_;
        gpu_id_ = o.gpu_id_;
        task_alloc_ = o.task_alloc_;
        data_alloc_ = o.data_alloc_;
        task_base_ = o.task_base_;
        data_base_ = o.data_base_;
        bytes_ = o.bytes_;
        handle_ = o.handle_;
        o.task_alloc_.SetNull();
        o.data_alloc_.SetNull();
        o.task_base_ = nullptr;
        o.data_base_ = nullptr;
    }
};

}  // namespace kvhdf5

#endif  // !CTP_IS_DEVICE_PASS
