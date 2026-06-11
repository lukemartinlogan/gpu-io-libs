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
 * ### Buffer model
 *
 * The host and kernel paths use different buffer strategies because they
 * have different correctness/performance tradeoffs:
 *
 *  - Kernel side: each PutBlob / GetBlob / DeleteBlob / Exists call
 *    allocates a fresh buffer via CHI_IPC->AllocateBuffer and frees it on
 *    return through a small RAII guard. Per-call allocation avoids a
 *    cross-call staleness class of bugs in the GPU runtime's task-handler
 *    source-pointer resolution that can leave stale bytes in a reused
 *    buffer (manifesting as a metadata-deserialization assertion failure
 *    many calls later).
 *
 *  - Host side: a single per-store cudaMallocHost'd scratch buffer is
 *    shared across all calls. The staleness bug doesn't apply on the host
 *    (host calls are synchronous and the CPU IPC path uses a different
 *    resolver). Per-call host allocation is much slower in practice than
 *    a single persistent allocation, so we keep the original mechanism.
 *
 * ### Construction
 *
 * Use GpuCteBlobStore::Create(tag_id, pool_id, scratch_size) on the host.
 * scratch_size sets the host scratch buffer capacity (default 1 MiB).
 * The returned object is trivially copyable — pass by value into kernels.
 * Call Destroy() on the host-side original after all kernel usage is done
 * to free the scratch buffer.
 *
 * ### Concurrency
 *
 * Kernel-side calls are independent (each owns its own buffer), so
 * concurrent calls from different warps are safe in principle. Host-side
 * calls share scratch_buf_, so they must not be called concurrently on the
 * same store instance.
 *
 * ### Delete / Exists sentinel
 *
 * CTE does not truncate on overwrite and has no GPU-callable DelBlob. Delete
 * is implemented by writing a tombstone sentinel: an 8-byte size prefix of
 * UINT64_MAX (kTombstoneSize). Real-value sizes can never reach UINT64_MAX
 * (the scratch buffer caps at sub-petabyte), so the sentinel is unambiguous
 * even when a stored blob legitimately has zero payload bytes. Exists reads
 * the prefix and returns true iff real_size != kTombstoneSize.
 */
// Templated RAII guard for IPC-allocated buffers. Frees the buffer on scope
// exit. Single ownership; no copy/move.
template <typename Ipc>
struct GpuCteBufferGuard {
    Ipc* ipc_;
    hipc::FullPtr<char> buf_;

    HSHM_INLINE_CROSS_FUN
    GpuCteBufferGuard(Ipc* ipc, hipc::FullPtr<char> buf)
        : ipc_(ipc), buf_(buf) {}

    HSHM_INLINE_CROSS_FUN
    ~GpuCteBufferGuard() {
        if (ipc_ && !buf_.IsNull()) ipc_->FreeBuffer(buf_);
    }

    GpuCteBufferGuard(const GpuCteBufferGuard&) = delete;
    GpuCteBufferGuard& operator=(const GpuCteBufferGuard&) = delete;
};

class GpuCteBlobStore {
    wrp_cte::core::TagId tag_id_;
    chi::PoolId pool_id_;

    // Host-side persistent scratch buffer (cudaMallocHost). Used by host
    // PutBlob / GetBlob / DeleteBlob / Exists. Unused by the kernel paths,
    // which allocate per-call via CHI_IPC->AllocateBuffer.
    char* scratch_buf_ = nullptr;
    size_t scratch_size_ = 0;

    // Size prefix stored before every blob value (same as CpuCteBlobStore).
    static constexpr size_t kPrefixSize = sizeof(uint64_t);

    // Tombstone sentinel: a size_prefix of UINT64_MAX marks a deleted entry.
    // Distinct from a real zero-size value, which stores prefix == 0.
    static constexpr uint64_t kTombstoneSize = ~uint64_t{0};

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
     * Host-only factory. Allocates the host-side scratch buffer (used by
     * PutBlob/GetBlob/DeleteBlob/Exists when called from host code).
     * Kernel-side calls don't touch scratch_buf_ — they allocate per call
     * via CHI_IPC->AllocateBuffer.
     *
     * Pauses / resumes the GPU orchestrator around cudaMallocHost because
     * the allocation is device-synchronizing and would otherwise deadlock
     * against the persistent CDP kernel.
     */
    static GpuCteBlobStore Create(wrp_cte::core::TagId tag_id,
                                   chi::PoolId pool_id,
                                   size_t scratch_size = 1ull << 20) {
        GpuCteBlobStore store;
        store.tag_id_ = tag_id;
        store.pool_id_ = pool_id;
        store.scratch_size_ = scratch_size;

        auto* gpu_ipc = CHI_CPU_IPC->GetGpuIpcManager();
        gpu_ipc->PauseGpuOrchestrator();
        void* raw = nullptr;
        cudaError_t err = cudaMallocHost(&raw, scratch_size);
        gpu_ipc->ResumeGpuOrchestrator();

        if (err != cudaSuccess) {
            store.scratch_buf_ = nullptr;
            store.scratch_size_ = 0;
            return store;
        }
        store.scratch_buf_ = reinterpret_cast<char*>(raw);
        return store;
    }

    /**
     * Host-only cleanup. Frees the host scratch buffer. Safe to call
     * multiple times; subsequent calls are no-ops.
     */
    void Destroy() {
        if (scratch_buf_ != nullptr) {
            auto* gpu_ipc = CHI_CPU_IPC->GetGpuIpcManager();
            gpu_ipc->PauseGpuOrchestrator();
            cudaFreeHost(scratch_buf_);
            gpu_ipc->ResumeGpuOrchestrator();
            scratch_buf_ = nullptr;
            scratch_size_ = 0;
        }
    }
#endif

    /** Whether the store is configured. Host requires a scratch buffer too. */
    HSHM_CROSS_FUN bool IsValid() const {
#if HSHM_IS_GPU
        return !tag_id_.IsNull() && !pool_id_.IsNull();
#else
        return scratch_buf_ != nullptr && scratch_size_ > 0
               && !tag_id_.IsNull() && !pool_id_.IsNull();
#endif
    }

    CROSS_FUN bool PutBlob(cstd::span<const byte_t> key,
                           cstd::span<const byte_t> value) {
        char name_buf[kMaxKeyNameBuf];
        KeyToName(key, name_buf);

        uint64_t real_size = value.size();
        size_t total = kPrefixSize + real_size;

#if HSHM_IS_GPU
        // Kernel: allocate per call via the GPU IPC manager and free on
        // return through the RAII guard.
        auto *ipc = CHI_IPC;
        auto buf = ipc->AllocateBuffer(total);
        if (buf.IsNull()) return false;
        GpuCteBufferGuard guard(ipc, buf);

        cstd::memcpy(buf.ptr_, &real_size, kPrefixSize);
        if (real_size > 0) {
            cstd::memcpy(buf.ptr_ + kPrefixSize, value.data(), real_size);
        }
        hipc::ShmPtr<> shm = buf.shm_.template Cast<void>();

        // Kernel-side dual-send: kernel PoolQuery::Local() reaches GPU CTE
        // only; ToLocalCpu() also populates CPU CTE so host readers see the
        // same blob.
        auto gpu_task = ipc->template NewTask<wrp_cte::core::PutBlobTask>(
            chi::CreateTaskId(), pool_id_, chi::PoolQuery::Local(),
            tag_id_, name_buf, chi::u64(0), chi::u64(total),
            shm, -1.0f, wrp_cte::core::Context(), chi::u32(0));
        if (gpu_task.IsNull()) return false;
        auto gpu_future = ipc->Send(gpu_task);
        gpu_future.Wait();
        if (static_cast<int>(gpu_task->GetReturnCode()) != 0) return false;

        auto cpu_task = ipc->template NewTask<wrp_cte::core::PutBlobTask>(
            chi::CreateTaskId(), pool_id_, chi::PoolQuery::ToLocalCpu(),
            tag_id_, name_buf, chi::u64(0), chi::u64(total),
            shm, -1.0f, wrp_cte::core::Context(), chi::u32(0));
        if (cpu_task.IsNull()) return false;
        auto cpu_future = ipc->Send(cpu_task);
        cpu_future.Wait();
        return static_cast<int>(cpu_task->GetReturnCode()) == 0;
#else
        // Host: persistent scratch_buf_ — synchronous, no staleness concern.
        if (scratch_buf_ == nullptr) return false;
        if (total > scratch_size_) return false;

        cstd::memcpy(scratch_buf_, &real_size, kPrefixSize);
        if (real_size > 0) {
            cstd::memcpy(scratch_buf_ + kPrefixSize, value.data(), real_size);
        }
        hipc::ShmPtr<> shm = hipc::ShmPtr<>::FromRaw(scratch_buf_);
        auto *ipc = CHI_CPU_IPC;

        // Host dual-send: Local() populates CPU CTE; LocalGpuBcast()
        // populates GPU CTE. Both are needed because the CPU and GPU
        // runtimes hold independent blob_map_ / targets_ state.
        auto cpu_task = ipc->template NewTask<wrp_cte::core::PutBlobTask>(
            chi::CreateTaskId(), pool_id_, chi::PoolQuery::Local(),
            tag_id_, name_buf, chi::u64(0), chi::u64(total),
            shm, -1.0f, wrp_cte::core::Context(), chi::u32(0));
        if (cpu_task.IsNull()) return false;
        auto cpu_future = ipc->Send(cpu_task);
        cpu_future.Wait();
        if (static_cast<int>(cpu_task->GetReturnCode()) != 0) return false;

        auto gpu_task = ipc->template NewTask<wrp_cte::core::PutBlobTask>(
            chi::CreateTaskId(), pool_id_, chi::PoolQuery::LocalGpuBcast(),
            tag_id_, name_buf, chi::u64(0), chi::u64(total),
            shm, -1.0f, wrp_cte::core::Context(), chi::u32(0));
        if (gpu_task.IsNull()) return false;
        auto gpu_future = ipc->Send(gpu_task);
        gpu_future.Wait();
        return static_cast<int>(gpu_task->GetReturnCode()) == 0;
#endif
    }

    CROSS_FUN cstd::expected<cstd::span<byte_t>, BlobStoreError>
    GetBlob(cstd::span<const byte_t> key, cstd::span<byte_t> value_out) {
        char name_buf[kMaxKeyNameBuf];
        KeyToName(key, name_buf);

#if HSHM_IS_GPU
        // Kernel: per-call buffers for probe and fetch phases.
        auto *ipc = CHI_IPC;
        uint64_t real_size = 0;
        {
            auto probe_buf = ipc->AllocateBuffer(kPrefixSize);
            if (probe_buf.IsNull()) {
                return cstd::unexpected<BlobStoreError>(BlobStoreError::NotEnoughSpace);
            }
            GpuCteBufferGuard probe_guard(ipc, probe_buf);
            cstd::memset(probe_buf.ptr_, 0, kPrefixSize);

            hipc::ShmPtr<> shm = probe_buf.shm_.template Cast<void>();
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
            cstd::memcpy(&real_size, probe_buf.ptr_, kPrefixSize);
        }

        if (real_size == kTombstoneSize) {
            return cstd::unexpected<BlobStoreError>(BlobStoreError::NotExist);
        }
        if (value_out.size() < real_size) {
            return cstd::unexpected<BlobStoreError>(BlobStoreError::NotEnoughSpace);
        }
        if (real_size == 0) {
            return cstd::span<byte_t>(value_out.data(), 0);
        }

        size_t fetch_size = kPrefixSize + real_size;
        auto fetch_buf = ipc->AllocateBuffer(fetch_size);
        if (fetch_buf.IsNull()) {
            return cstd::unexpected<BlobStoreError>(BlobStoreError::NotEnoughSpace);
        }
        GpuCteBufferGuard fetch_guard(ipc, fetch_buf);
        cstd::memset(fetch_buf.ptr_, 0, fetch_size);

        hipc::ShmPtr<> fetch_shm = fetch_buf.shm_.template Cast<void>();
        auto fetch_task = ipc->template NewTask<wrp_cte::core::GetBlobTask>(
            chi::CreateTaskId(), pool_id_, chi::PoolQuery::Local(),
            tag_id_, name_buf, chi::u64(0), chi::u64(fetch_size),
            chi::u32(0), fetch_shm);
        if (fetch_task.IsNull()) {
            return cstd::unexpected<BlobStoreError>(BlobStoreError::NotExist);
        }
        auto fetch_future = ipc->Send(fetch_task);
        fetch_future.Wait();
        if (static_cast<int>(fetch_task->GetReturnCode()) != 0) {
            return cstd::unexpected<BlobStoreError>(BlobStoreError::NotExist);
        }

        cstd::memcpy(value_out.data(), fetch_buf.ptr_ + kPrefixSize, real_size);
        return cstd::span<byte_t>(value_out.data(), real_size);
#else
        // Host: persistent scratch_buf_, two-phase probe + fetch in place.
        if (scratch_buf_ == nullptr) {
            return cstd::unexpected<BlobStoreError>(BlobStoreError::NotExist);
        }
        if (kPrefixSize > scratch_size_) {
            return cstd::unexpected<BlobStoreError>(BlobStoreError::NotEnoughSpace);
        }
        cstd::memset(scratch_buf_, 0, kPrefixSize);

        hipc::ShmPtr<> shm = hipc::ShmPtr<>::FromRaw(scratch_buf_);
        auto *ipc = CHI_CPU_IPC;

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
        if (real_size == kTombstoneSize) {
            return cstd::unexpected<BlobStoreError>(BlobStoreError::NotExist);
        }
        if (value_out.size() < real_size) {
            return cstd::unexpected<BlobStoreError>(BlobStoreError::NotEnoughSpace);
        }
        if (real_size == 0) {
            return cstd::span<byte_t>(value_out.data(), 0);
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
#endif
    }

    CROSS_FUN bool DeleteBlob(cstd::span<const byte_t> key) {
        char name_buf[kMaxKeyNameBuf];
        KeyToName(key, name_buf);
        uint64_t tombstone = kTombstoneSize;

#if HSHM_IS_GPU
        auto *ipc = CHI_IPC;
        auto buf = ipc->AllocateBuffer(kPrefixSize);
        if (buf.IsNull()) return false;
        GpuCteBufferGuard guard(ipc, buf);
        cstd::memcpy(buf.ptr_, &tombstone, kPrefixSize);
        hipc::ShmPtr<> shm = buf.shm_.template Cast<void>();

        auto gpu_task = ipc->template NewTask<wrp_cte::core::PutBlobTask>(
            chi::CreateTaskId(), pool_id_, chi::PoolQuery::Local(),
            tag_id_, name_buf, chi::u64(0), chi::u64(kPrefixSize),
            shm, -1.0f, wrp_cte::core::Context(), chi::u32(0));
        if (gpu_task.IsNull()) return false;
        auto gpu_future = ipc->Send(gpu_task);
        gpu_future.Wait();
        if (static_cast<int>(gpu_task->GetReturnCode()) != 0) return false;

        auto cpu_task = ipc->template NewTask<wrp_cte::core::PutBlobTask>(
            chi::CreateTaskId(), pool_id_, chi::PoolQuery::ToLocalCpu(),
            tag_id_, name_buf, chi::u64(0), chi::u64(kPrefixSize),
            shm, -1.0f, wrp_cte::core::Context(), chi::u32(0));
        if (cpu_task.IsNull()) return false;
        auto cpu_future = ipc->Send(cpu_task);
        cpu_future.Wait();
        return static_cast<int>(cpu_task->GetReturnCode()) == 0;
#else
        if (scratch_buf_ == nullptr) return false;
        if (kPrefixSize > scratch_size_) return false;
        cstd::memcpy(scratch_buf_, &tombstone, kPrefixSize);
        hipc::ShmPtr<> shm = hipc::ShmPtr<>::FromRaw(scratch_buf_);
        auto *ipc = CHI_CPU_IPC;

        auto cpu_task = ipc->template NewTask<wrp_cte::core::PutBlobTask>(
            chi::CreateTaskId(), pool_id_, chi::PoolQuery::Local(),
            tag_id_, name_buf, chi::u64(0), chi::u64(kPrefixSize),
            shm, -1.0f, wrp_cte::core::Context(), chi::u32(0));
        if (cpu_task.IsNull()) return false;
        auto cpu_future = ipc->Send(cpu_task);
        cpu_future.Wait();
        if (static_cast<int>(cpu_task->GetReturnCode()) != 0) return false;

        auto gpu_task = ipc->template NewTask<wrp_cte::core::PutBlobTask>(
            chi::CreateTaskId(), pool_id_, chi::PoolQuery::LocalGpuBcast(),
            tag_id_, name_buf, chi::u64(0), chi::u64(kPrefixSize),
            shm, -1.0f, wrp_cte::core::Context(), chi::u32(0));
        if (gpu_task.IsNull()) return false;
        auto gpu_future = ipc->Send(gpu_task);
        gpu_future.Wait();
        return static_cast<int>(gpu_task->GetReturnCode()) == 0;
#endif
    }

    CROSS_FUN bool Exists(cstd::span<const byte_t> key) {
        char name_buf[kMaxKeyNameBuf];
        KeyToName(key, name_buf);

#if HSHM_IS_GPU
        auto *ipc = CHI_IPC;
        auto buf = ipc->AllocateBuffer(kPrefixSize);
        if (buf.IsNull()) return false;
        GpuCteBufferGuard guard(ipc, buf);
        cstd::memset(buf.ptr_, 0, kPrefixSize);
        hipc::ShmPtr<> shm = buf.shm_.template Cast<void>();
        char* read_ptr = buf.ptr_;
#else
        if (scratch_buf_ == nullptr) return false;
        if (kPrefixSize > scratch_size_) return false;
        cstd::memset(scratch_buf_, 0, kPrefixSize);
        hipc::ShmPtr<> shm = hipc::ShmPtr<>::FromRaw(scratch_buf_);
        auto *ipc = CHI_CPU_IPC;
        char* read_ptr = scratch_buf_;
#endif

        auto task = ipc->template NewTask<wrp_cte::core::GetBlobTask>(
            chi::CreateTaskId(), pool_id_, chi::PoolQuery::Local(),
            tag_id_, name_buf, chi::u64(0), chi::u64(kPrefixSize),
            chi::u32(0), shm);
        if (task.IsNull()) return false;
        auto future = ipc->Send(task);
        future.Wait();
        if (static_cast<int>(task->GetReturnCode()) != 0) return false;

        uint64_t real_size;
        cstd::memcpy(&real_size, read_ptr, kPrefixSize);
        return real_size != kTombstoneSize;
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
