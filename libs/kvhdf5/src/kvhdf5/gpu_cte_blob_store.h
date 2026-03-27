#pragma once

#include "../defines.h"
#include "blob_store.h"
#include <chimaera/chimaera.h>
#include <wrp_cte/core/core_client.h>
#include <hermes_shm/util/gpu_api.h>

namespace kvhdf5 {

/**
 * GPU-callable CTE blob store.
 *
 * Satisfies the RawBlobStore concept. All methods are CROSS_FUN
 * and can be called from GPU kernels after CHIMAERA_GPU_INIT.
 *
 * Construction happens on the host (tag creation is host-only).
 * The object is trivially copyable — pass by value into kernels.
 *
 * Delete uses a zero-size sentinel: PutBlob with real_size=0.
 * Exists reads the prefix and checks real_size != 0.
 *
 * Destroy() is host-only (AsyncDelTag requires std::string).
 */
class GpuCteBlobStore {
    wrp_cte::core::TagId tag_id_;
    chi::PoolId pool_id_;

    // Size prefix stored before every blob value (same as CpuCteBlobStore).
    static constexpr size_t kPrefixSize = sizeof(uint64_t);

    // Max hex-encoded key length: 64-byte key -> 128 hex chars + null.
    // Covers ChunkKey (the largest key type at ~80 bytes serialized).
    static constexpr size_t kMaxKeyHexBuf = 256;

public:
    CROSS_FUN GpuCteBlobStore(wrp_cte::core::TagId tag_id, chi::PoolId pool_id)
        : tag_id_(tag_id), pool_id_(pool_id) {}

    CROSS_FUN bool PutBlob(cstd::span<const byte_t> key,
                           cstd::span<const byte_t> value) {
        // 1. Hex-encode key
        char hex_buf[kMaxKeyHexBuf];
        size_t hex_len = KeyToHex(key, hex_buf, kMaxKeyHexBuf);
        hex_buf[hex_len] = '\0';

        // 2. Allocate UVM buffer for [size_prefix][data]
        uint64_t real_size = value.size();
        size_t total = kPrefixSize + real_size;

        hipc::FullPtr<char> buf = CHI_IPC->AllocateBuffer(total);
        if (buf.IsNull()) return false;

        // 3. Write size prefix + data
        cstd::memcpy(buf.ptr_, &real_size, kPrefixSize);
        if (real_size > 0) {
            cstd::memcpy(buf.ptr_ + kPrefixSize, value.data(), real_size);
        }

        // 4. Submit async put
        hipc::ShmPtr<> shm = buf.shm_.template Cast<void>();
        wrp_cte::core::Client client(pool_id_);
        auto future = client.AsyncPutBlob(
            tag_id_, hex_buf,
            0, total,
            shm,
            -1.0f,
            wrp_cte::core::Context(),
            0,
            chi::PoolQuery::Local());

        // 5. Poll until complete
        future.Wait();
        int rc = static_cast<int>(future->GetReturnCode());

        // 6. Free buffer
        CHI_IPC->FreeBuffer(buf);
        return rc == 0;
    }

    CROSS_FUN cstd::expected<cstd::span<byte_t>, BlobStoreError>
    GetBlob(cstd::span<const byte_t> key, cstd::span<byte_t> value_out) {
        // 1. Hex-encode key
        char hex_buf[kMaxKeyHexBuf];
        size_t hex_len = KeyToHex(key, hex_buf, kMaxKeyHexBuf);
        hex_buf[hex_len] = '\0';

        // 2. Allocate receive buffer (prefix + value_out capacity)
        size_t alloc_size = kPrefixSize + value_out.size();
        hipc::FullPtr<char> buf = CHI_IPC->AllocateBuffer(alloc_size);
        if (buf.IsNull()) {
            return cstd::unexpected<BlobStoreError>(BlobStoreError::NotExist);
        }

        // Zero the buffer so we can detect empty reads
        cstd::memset(buf.ptr_, 0, alloc_size);

        // 3. Submit async get
        hipc::ShmPtr<> shm = buf.shm_.template Cast<void>();
        wrp_cte::core::Client client(pool_id_);
        auto future = client.AsyncGetBlob(
            tag_id_, hex_buf,
            0, alloc_size,
            0,
            shm,
            chi::PoolQuery::Local());

        // 4. Poll until complete
        future.Wait();
        int rc = static_cast<int>(future->GetReturnCode());

        if (rc != 0) {
            CHI_IPC->FreeBuffer(buf);
            return cstd::unexpected<BlobStoreError>(BlobStoreError::NotExist);
        }

        // 5. Read size prefix
        uint64_t real_size;
        cstd::memcpy(&real_size, buf.ptr_, kPrefixSize);

        // Sentinel: real_size == 0 means "deleted"
        if (real_size == 0) {
            CHI_IPC->FreeBuffer(buf);
            return cstd::unexpected<BlobStoreError>(BlobStoreError::NotExist);
        }

        if (value_out.size() < real_size) {
            CHI_IPC->FreeBuffer(buf);
            return cstd::unexpected<BlobStoreError>(BlobStoreError::NotEnoughSpace);
        }

        // 6. Copy data into output
        if (real_size > 0) {
            cstd::memcpy(value_out.data(), buf.ptr_ + kPrefixSize, real_size);
        }

        CHI_IPC->FreeBuffer(buf);
        return cstd::span<byte_t>(value_out.data(), real_size);
    }

    CROSS_FUN bool DeleteBlob(cstd::span<const byte_t> key) {
        // Sentinel delete: overwrite blob with real_size=0
        uint64_t zero = 0;

        char hex_buf[kMaxKeyHexBuf];
        size_t hex_len = KeyToHex(key, hex_buf, kMaxKeyHexBuf);
        hex_buf[hex_len] = '\0';

        hipc::FullPtr<char> buf = CHI_IPC->AllocateBuffer(kPrefixSize);
        if (buf.IsNull()) return false;

        cstd::memcpy(buf.ptr_, &zero, kPrefixSize);

        hipc::ShmPtr<> shm = buf.shm_.template Cast<void>();
        wrp_cte::core::Client client(pool_id_);
        auto future = client.AsyncPutBlob(
            tag_id_, hex_buf,
            0, kPrefixSize,
            shm,
            -1.0f,
            wrp_cte::core::Context(),
            0,
            chi::PoolQuery::Local());

        future.Wait();
        int rc = static_cast<int>(future->GetReturnCode());

        CHI_IPC->FreeBuffer(buf);
        return rc == 0;
    }

    CROSS_FUN bool Exists(cstd::span<const byte_t> key) {
        // Read just the size prefix to check existence
        char hex_buf[kMaxKeyHexBuf];
        size_t hex_len = KeyToHex(key, hex_buf, kMaxKeyHexBuf);
        hex_buf[hex_len] = '\0';

        hipc::FullPtr<char> buf = CHI_IPC->AllocateBuffer(kPrefixSize);
        if (buf.IsNull()) return false;

        cstd::memset(buf.ptr_, 0, kPrefixSize);

        hipc::ShmPtr<> shm = buf.shm_.template Cast<void>();
        wrp_cte::core::Client client(pool_id_);
        auto future = client.AsyncGetBlob(
            tag_id_, hex_buf,
            0, kPrefixSize,
            0,
            shm,
            chi::PoolQuery::Local());

        future.Wait();
        int rc = static_cast<int>(future->GetReturnCode());

        if (rc != 0) {
            CHI_IPC->FreeBuffer(buf);
            return false;
        }

        uint64_t real_size;
        cstd::memcpy(&real_size, buf.ptr_, kPrefixSize);

        CHI_IPC->FreeBuffer(buf);
        return real_size != 0;
    }

#if !HSHM_IS_GPU
    /** Host-only: destroy the CTE tag. Call once when done with this store. */
    void Destroy() {
        auto task = WRP_CTE_CLIENT->AsyncDelTag(tag_id_);
        task.Wait();
    }
#endif

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
