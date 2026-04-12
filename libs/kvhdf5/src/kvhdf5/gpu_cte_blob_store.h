#pragma once

#include "../defines.h"
#include "blob_store.h"
#include <chimaera/chimaera.h>
#include <wrp_cte/core/core_client.h>
#include <wrp_cte/core/core_tasks.h>
#include <hermes_shm/util/gpu_api.h>

#if !HSHM_IS_GPU
#include <cuda_runtime.h>
#endif

namespace kvhdf5 {

/**
 * GPU-callable CTE blob store.
 *
 * Satisfies the RawBlobStore concept. All data-plane methods are CROSS_FUN
 * and callable from GPU kernels after CHIMAERA_GPU_INIT.
 *
 * ### Scratch buffer model
 *
 * Each store owns a host-allocated `cudaMallocManaged` scratch region that is
 * used as the data buffer for PutBlob and GetBlob. The scratch is UVM memory,
 * so the same raw pointer is valid on both CPU and GPU and reads / writes
 * from either side are coherent after a CUDA fence.
 *
 * This is the approach iowarp-core's own tests use
 * (see test_gpu_initiated_gpu.cc): allocate via cudaMallocManaged, pass the
 * raw pointer to the kernel, wrap it with hipc::ShmPtr<>::FromRaw() so the
 * CPU side of ToFullPtr() resolves it as an absolute UVA pointer.
 *
 * Attempts to allocate the data buffer from GPU code via
 * CHI_IPC->AllocateBuffer instead failed: the per-warp BuddyAllocator's
 * alloc_id is not registered in the CPU-side gpu_alloc_map_, so the task
 * handler could not resolve the ShmPtr back to a readable host pointer.
 *
 * ### Construction
 *
 * Use GpuCteBlobStore::Create(tag_id, pool_id, scratch_size) on the host.
 * The returned object is trivially copyable — pass by value into kernels.
 * Call Destroy() on the host-side original after all kernel usage is done
 * to free the scratch buffer (and optionally the tag).
 *
 * ### Concurrency
 *
 * The scratch buffer is shared across all calls on a single store instance,
 * so PutBlob / GetBlob / DeleteBlob / Exists MUST NOT be called concurrently
 * on the same store. Single-thread (lane 0) usage from a kernel is the
 * expected pattern.
 *
 * ### Delete / Exists sentinel
 *
 * CTE does not truncate on overwrite and has no GPU-callable DelBlob. Delete
 * is implemented by writing a zero-size sentinel (a blob whose 8-byte size
 * prefix is 0). Exists reads the prefix and returns true iff real_size != 0.
 */
class GpuCteBlobStore {
    wrp_cte::core::TagId tag_id_;
    chi::PoolId pool_id_;

    // Managed (UVM) scratch buffer. Allocated on host via cudaMallocManaged,
    // freed by Destroy(). Both CPU and GPU access the same virtual address.
    char *scratch_buf_ = nullptr;
    size_t scratch_size_ = 0;

    // Size prefix stored before every blob value (same as CpuCteBlobStore).
    static constexpr size_t kPrefixSize = sizeof(uint64_t);

    // Max hex-encoded key length: 64-byte key -> 128 hex chars + null.
    // Covers ChunkKey (the largest key type at ~80 bytes serialized).
    static constexpr size_t kMaxKeyHexBuf = 256;

public:
    /** Default-constructible so the struct stays trivially copyable. */
    HSHM_CROSS_FUN GpuCteBlobStore() = default;

#if !HSHM_IS_GPU
    /**
     * Host-only factory. Allocates a managed scratch buffer sized to hold
     * the largest expected blob (plus the 8-byte size prefix).
     *
     * Pauses / resumes the GPU orchestrator around cudaMallocManaged because
     * the allocation is device-synchronizing and would otherwise deadlock
     * against the persistent CDP kernel (same reasoning as test_gpu_initiated
     * in iowarp-core).
     *
     * @param tag_id CTE tag this store writes into (already created).
     * @param pool_id CTE core pool to route tasks to.
     * @param scratch_size Scratch buffer capacity in bytes. Must exceed
     *        max(put_value_size, get_value_size) + 8. Default 1 MiB.
     */
    static GpuCteBlobStore Create(wrp_cte::core::TagId tag_id,
                                   chi::PoolId pool_id,
                                   size_t scratch_size = 1ull << 20) {
        GpuCteBlobStore store;
        store.tag_id_ = tag_id;
        store.pool_id_ = pool_id;
        store.scratch_size_ = scratch_size;

        auto *gpu_ipc = CHI_CPU_IPC->GetGpuIpcManager();
        gpu_ipc->PauseGpuOrchestrator();
        // Pinned host memory: page-locked in CPU DRAM, GPU accesses via
        // zero-copy PCIe. With UVA, the same pointer is valid on both sides.
        // GPU PCIe DMA writes participate in cache snooping on x86, so
        // system-scope fences (already issued by IpcGpu2Cpu::ClientSend)
        // ensure CPU reads see the GPU writes — unlike cudaMallocManaged
        // on this consumer-GPU configuration where page migration can
        // interfere with coherency.
        void *raw = nullptr;
        cudaError_t err = cudaMallocHost(&raw, scratch_size);
        gpu_ipc->ResumeGpuOrchestrator();

        if (err != cudaSuccess) {
            store.scratch_buf_ = nullptr;
            store.scratch_size_ = 0;
            return store;
        }
        store.scratch_buf_ = reinterpret_cast<char *>(raw);
        return store;
    }

    /**
     * Host-only cleanup. Frees the managed scratch buffer. Safe to call
     * multiple times; subsequent calls are no-ops. Pauses / resumes the
     * orchestrator around cudaFree for the same reason as Create.
     */
    void Destroy() {
        if (scratch_buf_ != nullptr) {
            auto *gpu_ipc = CHI_CPU_IPC->GetGpuIpcManager();
            gpu_ipc->PauseGpuOrchestrator();
            cudaFreeHost(scratch_buf_);
            gpu_ipc->ResumeGpuOrchestrator();
            scratch_buf_ = nullptr;
            scratch_size_ = 0;
        }
    }
#endif

    /** Whether the store has a valid scratch buffer. */
    HSHM_CROSS_FUN bool IsValid() const {
        return scratch_buf_ != nullptr && scratch_size_ > 0;
    }

    CROSS_FUN bool PutBlob(cstd::span<const byte_t> key,
                           cstd::span<const byte_t> value) {
        if (scratch_buf_ == nullptr) return false;

        // 1. Hex-encode key
        char hex_buf[kMaxKeyHexBuf];
        size_t hex_len = KeyToHex(key, hex_buf, kMaxKeyHexBuf);
        hex_buf[hex_len] = '\0';

        // 2. Check the scratch buffer is large enough for [prefix][value]
        uint64_t real_size = value.size();
        size_t total = kPrefixSize + real_size;
        if (total > scratch_size_) return false;

        // 3. Write size prefix + data into the managed scratch buffer.
        //    Writes are visible to the CPU after the IPC fence in ClientSend.
        cstd::memcpy(scratch_buf_, &real_size, kPrefixSize);
        if (real_size > 0) {
            cstd::memcpy(scratch_buf_ + kPrefixSize, value.data(), real_size);
        }

        // 4. Submit put task directly with a raw-UVA ShmPtr.
        hipc::ShmPtr<> shm = hipc::ShmPtr<>::FromRaw(scratch_buf_);
        auto *ipc = CHI_IPC;
        auto task = ipc->template NewTask<wrp_cte::core::PutBlobTask>(
            chi::CreateTaskId(), pool_id_, chi::PoolQuery::Local(),
            tag_id_, hex_buf, chi::u64(0), chi::u64(total),
            shm, -1.0f, wrp_cte::core::Context(), chi::u32(0));
        if (task.IsNull()) return false;
        auto future = ipc->Send(task);

        future.Wait();
        int rc = static_cast<int>(task->GetReturnCode());
        return rc == 0;
    }

    CROSS_FUN cstd::expected<cstd::span<byte_t>, BlobStoreError>
    GetBlob(cstd::span<const byte_t> key, cstd::span<byte_t> value_out) {
        if (scratch_buf_ == nullptr) {
            return cstd::unexpected<BlobStoreError>(BlobStoreError::NotExist);
        }

        // 1. Hex-encode key
        char hex_buf[kMaxKeyHexBuf];
        size_t hex_len = KeyToHex(key, hex_buf, kMaxKeyHexBuf);
        hex_buf[hex_len] = '\0';

        // 2. Ensure scratch holds [prefix][value_out]
        size_t request_size = kPrefixSize + value_out.size();
        if (request_size > scratch_size_) {
            return cstd::unexpected<BlobStoreError>(BlobStoreError::NotEnoughSpace);
        }

        // Zero the scratch so we can detect empty reads.
        cstd::memset(scratch_buf_, 0, request_size);

        // 3. Submit get task with raw-UVA ShmPtr.
        hipc::ShmPtr<> shm = hipc::ShmPtr<>::FromRaw(scratch_buf_);
        auto *ipc = CHI_IPC;
        auto task = ipc->template NewTask<wrp_cte::core::GetBlobTask>(
            chi::CreateTaskId(), pool_id_, chi::PoolQuery::Local(),
            tag_id_, hex_buf, chi::u64(0), chi::u64(request_size),
            chi::u32(0), shm);
        if (task.IsNull()) {
            return cstd::unexpected<BlobStoreError>(BlobStoreError::NotExist);
        }
        auto future = ipc->Send(task);

        future.Wait();
        int rc = static_cast<int>(task->GetReturnCode());
        if (rc != 0) {
            return cstd::unexpected<BlobStoreError>(BlobStoreError::NotExist);
        }

        // 4. Read size prefix
        uint64_t real_size;
        cstd::memcpy(&real_size, scratch_buf_, kPrefixSize);

        // Sentinel: real_size == 0 means "deleted"
        if (real_size == 0) {
            return cstd::unexpected<BlobStoreError>(BlobStoreError::NotExist);
        }

        if (value_out.size() < real_size) {
            return cstd::unexpected<BlobStoreError>(BlobStoreError::NotEnoughSpace);
        }

        // 5. Copy data into output
        if (real_size > 0) {
            cstd::memcpy(value_out.data(), scratch_buf_ + kPrefixSize, real_size);
        }

        return cstd::span<byte_t>(value_out.data(), real_size);
    }

    CROSS_FUN bool DeleteBlob(cstd::span<const byte_t> key) {
        if (scratch_buf_ == nullptr) return false;

        // Sentinel delete: overwrite blob with real_size=0
        char hex_buf[kMaxKeyHexBuf];
        size_t hex_len = KeyToHex(key, hex_buf, kMaxKeyHexBuf);
        hex_buf[hex_len] = '\0';

        if (kPrefixSize > scratch_size_) return false;

        uint64_t zero = 0;
        cstd::memcpy(scratch_buf_, &zero, kPrefixSize);

        hipc::ShmPtr<> shm = hipc::ShmPtr<>::FromRaw(scratch_buf_);
        auto *ipc = CHI_IPC;
        auto task = ipc->template NewTask<wrp_cte::core::PutBlobTask>(
            chi::CreateTaskId(), pool_id_, chi::PoolQuery::Local(),
            tag_id_, hex_buf, chi::u64(0), chi::u64(kPrefixSize),
            shm, -1.0f, wrp_cte::core::Context(), chi::u32(0));
        if (task.IsNull()) return false;
        auto future = ipc->Send(task);

        future.Wait();
        int rc = static_cast<int>(task->GetReturnCode());
        return rc == 0;
    }

    CROSS_FUN bool Exists(cstd::span<const byte_t> key) {
        if (scratch_buf_ == nullptr) return false;

        // Read just the size prefix to check existence
        char hex_buf[kMaxKeyHexBuf];
        size_t hex_len = KeyToHex(key, hex_buf, kMaxKeyHexBuf);
        hex_buf[hex_len] = '\0';

        if (kPrefixSize > scratch_size_) return false;

        cstd::memset(scratch_buf_, 0, kPrefixSize);

        hipc::ShmPtr<> shm = hipc::ShmPtr<>::FromRaw(scratch_buf_);
        auto *ipc = CHI_IPC;
        auto task = ipc->template NewTask<wrp_cte::core::GetBlobTask>(
            chi::CreateTaskId(), pool_id_, chi::PoolQuery::Local(),
            tag_id_, hex_buf, chi::u64(0), chi::u64(kPrefixSize),
            chi::u32(0), shm);
        if (task.IsNull()) return false;
        auto future = ipc->Send(task);

        future.Wait();
        int rc = static_cast<int>(task->GetReturnCode());
        if (rc != 0) return false;

        uint64_t real_size;
        cstd::memcpy(&real_size, scratch_buf_, kPrefixSize);
        return real_size != 0;
    }

private:
    /**
     * Encode a byte key as hex into a caller-provided buffer.
     * Empty key encodes as "_" (same convention as CpuCteBlobStore).
     * Returns the number of characters written (not counting null terminator).
     */
    CROSS_FUN static size_t KeyToHex(cstd::span<const byte_t> key,
                                      char* out, size_t out_size) {
        if (key.empty()) {
            out[0] = '_';
            return 1;
        }

        const char hex_chars[] = "0123456789abcdef";
        size_t len = 0;
        for (size_t i = 0; i < key.size() && len + 2 < out_size; ++i) {
            uint8_t byte = static_cast<uint8_t>(key[i]);
            out[len++] = hex_chars[byte >> 4];
            out[len++] = hex_chars[byte & 0x0F];
        }
        return len;
    }
};

static_assert(RawBlobStore<GpuCteBlobStore>);

}  // namespace kvhdf5
