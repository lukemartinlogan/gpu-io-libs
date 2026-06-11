#pragma once

#include "../defines.h"
#include "blob_store.h"
#include "../utils/gpu_string.h"
#include "wrp_cte/core/core_client.h"
#include <string>
#include <string_view>

namespace kvhdf5 {

/**
 * CTE-backed blob store implementation (CPU-only).
 * Maps the span-based RawBlobStore API onto CTE's string-named Tag/Blob API.
 *
 * Keys are hex-encoded into CTE blob names. Values are stored with an
 * 8-byte size prefix to handle CTE's no-truncate-on-overwrite behavior
 * and zero-length value restrictions.
 *
 * Requires Chimaera runtime and CTE client to be initialized before use.
 * CPU-only — not usable from GPU kernels.
 */
class CpuCteBlobStore {
    wrp_cte::core::Tag tag_;

public:
    explicit CpuCteBlobStore(const char* tag_name) : CpuCteBlobStore(std::string_view{tag_name}) {}
    explicit CpuCteBlobStore(std::string_view tag_name);
    explicit CpuCteBlobStore(gpu_string_view tag_name);

    bool PutBlob(cstd::span<const byte_t> key, cstd::span<const byte_t> value);

    cstd::expected<cstd::span<byte_t>, BlobStoreError> GetBlob(
        cstd::span<const byte_t> key, cstd::span<byte_t> value_out);

    bool DeleteBlob(cstd::span<const byte_t> key);

    bool Exists(cstd::span<const byte_t> key);

    /**
     * Destroy this store's CTE tag and free its resources (bdev pool, SHM).
     * Blocks until deletion completes. Call once when done with this store.
     */
    void Destroy();

private:
    static std::string KeyToHex(cstd::span<const byte_t> key);
};

static_assert(RawBlobStore<CpuCteBlobStore>);

} // namespace kvhdf5
