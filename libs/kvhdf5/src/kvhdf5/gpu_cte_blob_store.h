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

    // Encoded blob name: exactly kNameLen null-free characters plus a trailing
    // '\0'. See KeyToName() below for why the length must stay <= 10.
    //
    // Root-cause note: iowarp-core's GPU-side PutBlob / GetBlob / DelBlob
    // handler builds a "compound key" via
    //   chi::priv::string::reserve(22 + blob_name_len)
    // (see core_runtime_gpu.cc: MakeCompoundKey). When the reserved capacity
    // exceeds the chi::priv::string SSO size (32), reserve() calls
    // TransitionToHeap(), which invokes the per-warp PrivateBuddyAllocator.
    // That device-side heap allocation path faults with CUDA Error 700 in
    // this runtime configuration, so we must keep blob_name_len <= 10 to
    // stay inside SSO: 22 + 10 = 32 <= SSO capacity 32.
    //
    // A hex encoding (2 chars per byte) would need <=5-byte keys, which is
    // too small for real use (8-byte IDs, 80-byte ChunkKey). We instead
    // hash the key with FNV-1a (64-bit) and encode the 64-bit digest in 10
    // Z85 characters, giving a fixed-width null-free name regardless of
    // input key size. Z85's alphabet is ASCII-printable and excludes NUL,
    // so the name survives PutBlobTask's null-terminated C-string path.
    //
    // Hash collision risk is birthday-paradox 64-bit (~2^32 distinct keys
    // before a ~50% collision chance). Acceptable for this runtime's
    // expected working set; document and revisit if a user-facing CTE
    // scans by prefix or exposes names externally.
    static constexpr size_t kNameLen = 10;
    static constexpr size_t kMaxKeyNameBuf = kNameLen + 1;  // +null

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

        // 1. Encode key as a fixed-width 10-char null-free name.
        char name_buf[kMaxKeyNameBuf];
        KeyToName(key, name_buf);

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
#if HSHM_IS_GPU
        auto *ipc = CHI_IPC;
#else
        auto *ipc = CHI_CPU_IPC;
#endif
        auto task = ipc->template NewTask<wrp_cte::core::PutBlobTask>(
            chi::CreateTaskId(), pool_id_, chi::PoolQuery::Local(),
            tag_id_, name_buf, chi::u64(0), chi::u64(total),
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

        char name_buf[kMaxKeyNameBuf];
        KeyToName(key, name_buf);

        // The server-side GetBlob handler rejects requests that read past the
        // end of a blob (ReadData returns non-zero rc). Since the caller's
        // value_out may be larger than the stored value, do a two-phase read:
        // first fetch only the 8-byte size prefix to learn real_size, then
        // fetch exactly kPrefixSize + real_size bytes.
        if (kPrefixSize > scratch_size_) {
            return cstd::unexpected<BlobStoreError>(BlobStoreError::NotEnoughSpace);
        }
        cstd::memset(scratch_buf_, 0, kPrefixSize);

        hipc::ShmPtr<> shm = hipc::ShmPtr<>::FromRaw(scratch_buf_);
#if HSHM_IS_GPU
        auto *ipc = CHI_IPC;
#else
        auto *ipc = CHI_CPU_IPC;
#endif

        auto probe_task = ipc->template NewTask<wrp_cte::core::GetBlobTask>(
            chi::CreateTaskId(), pool_id_, chi::PoolQuery::Local(),
            tag_id_, name_buf, chi::u64(0), chi::u64(kPrefixSize),
            chi::u32(0), shm);
        if (probe_task.IsNull()) {
            return cstd::unexpected<BlobStoreError>(BlobStoreError::NotExist);
        }
        auto probe_future = ipc->Send(probe_task);
        probe_future.Wait();
        if (static_cast<int>(probe_task->GetReturnCode()) != 0) {
            return cstd::unexpected<BlobStoreError>(BlobStoreError::NotExist);
        }

        uint64_t real_size;
        cstd::memcpy(&real_size, scratch_buf_, kPrefixSize);

        // Sentinel: real_size == 0 means "deleted"
        if (real_size == 0) {
            return cstd::unexpected<BlobStoreError>(BlobStoreError::NotExist);
        }

        if (value_out.size() < real_size) {
            return cstd::unexpected<BlobStoreError>(BlobStoreError::NotEnoughSpace);
        }

        size_t fetch_size = kPrefixSize + real_size;
        if (fetch_size > scratch_size_) {
            return cstd::unexpected<BlobStoreError>(BlobStoreError::NotEnoughSpace);
        }

        cstd::memset(scratch_buf_, 0, fetch_size);

        auto fetch_task = ipc->template NewTask<wrp_cte::core::GetBlobTask>(
            chi::CreateTaskId(), pool_id_, chi::PoolQuery::Local(),
            tag_id_, name_buf, chi::u64(0), chi::u64(fetch_size),
            chi::u32(0), shm);
        if (fetch_task.IsNull()) {
            return cstd::unexpected<BlobStoreError>(BlobStoreError::NotExist);
        }
        auto fetch_future = ipc->Send(fetch_task);
        fetch_future.Wait();
        if (static_cast<int>(fetch_task->GetReturnCode()) != 0) {
            return cstd::unexpected<BlobStoreError>(BlobStoreError::NotExist);
        }

        cstd::memcpy(value_out.data(), scratch_buf_ + kPrefixSize, real_size);

        return cstd::span<byte_t>(value_out.data(), real_size);
    }

    CROSS_FUN bool DeleteBlob(cstd::span<const byte_t> key) {
        if (scratch_buf_ == nullptr) return false;

        // Sentinel delete: overwrite blob with real_size=0
        char name_buf[kMaxKeyNameBuf];
        KeyToName(key, name_buf);

        if (kPrefixSize > scratch_size_) return false;

        uint64_t zero = 0;
        cstd::memcpy(scratch_buf_, &zero, kPrefixSize);

        hipc::ShmPtr<> shm = hipc::ShmPtr<>::FromRaw(scratch_buf_);
#if HSHM_IS_GPU
        auto *ipc = CHI_IPC;
#else
        auto *ipc = CHI_CPU_IPC;
#endif
        auto task = ipc->template NewTask<wrp_cte::core::PutBlobTask>(
            chi::CreateTaskId(), pool_id_, chi::PoolQuery::Local(),
            tag_id_, name_buf, chi::u64(0), chi::u64(kPrefixSize),
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
        char name_buf[kMaxKeyNameBuf];
        KeyToName(key, name_buf);

        if (kPrefixSize > scratch_size_) return false;

        cstd::memset(scratch_buf_, 0, kPrefixSize);

        hipc::ShmPtr<> shm = hipc::ShmPtr<>::FromRaw(scratch_buf_);
#if HSHM_IS_GPU
        auto *ipc = CHI_IPC;
#else
        auto *ipc = CHI_CPU_IPC;
#endif
        auto task = ipc->template NewTask<wrp_cte::core::GetBlobTask>(
            chi::CreateTaskId(), pool_id_, chi::PoolQuery::Local(),
            tag_id_, name_buf, chi::u64(0), chi::u64(kPrefixSize),
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
     * Encode an arbitrary-length byte key into a fixed-width, null-free name
     * of exactly kNameLen characters (plus a trailing '\0').
     *
     * Strategy: FNV-1a 64-bit hash over the key bytes, then emit the digest
     * in base-85 using a null-free printable alphabet. 10 base-85 digits
     * cover the full 64-bit space (85^10 > 2^64), so the encoding is
     * bijective on the hash output.
     *
     * An empty key hashes to the FNV-1a offset basis, yielding a fixed
     * sentinel name — consistent with CpuCteBlobStore's empty-key handling.
     *
     * The caller's buffer must be >= kMaxKeyNameBuf bytes.
     */
    CROSS_FUN static void KeyToName(cstd::span<const byte_t> key, char* out) {
        // FNV-1a 64-bit
        uint64_t h = 0xcbf29ce484222325ULL;
        for (size_t i = 0; i < key.size(); ++i) {
            h ^= static_cast<uint64_t>(static_cast<uint8_t>(key[i]));
            h *= 0x100000001b3ULL;
        }
        // Fold the key length in so that two keys whose bytes are a prefix
        // of one another still produce different names.
        h ^= static_cast<uint64_t>(key.size());
        h *= 0x100000001b3ULL;

        // Base-85 digits (null-free, printable ASCII, ~Z85 alphabet).
        static const char alphabet[86] =
            "0123456789"
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            ".-:+=^!/*?&<>()[]{}@%$#";
        for (size_t i = 0; i < kNameLen; ++i) {
            out[kNameLen - 1 - i] = alphabet[h % 85];
            h /= 85;
        }
        out[kNameLen] = '\0';
    }
};

static_assert(RawBlobStore<GpuCteBlobStore>);

}  // namespace kvhdf5
