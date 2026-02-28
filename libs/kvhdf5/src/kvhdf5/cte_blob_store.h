#pragma once

#include "../defines.h"
#include "blob_store.h"
#include "../utils/gpu_string.h"
#include "wrp_cte/core/core_client.h"
#include <string>
#include <string_view>

namespace kvhdf5 {

/**
 * CTE-backed blob store implementation.
 * Maps the span-based RawBlobStore API onto CTE's string-named Tag/Blob API.
 *
 * Keys are hex-encoded into CTE blob names. Values are stored with an
 * 8-byte size prefix to handle CTE's no-truncate-on-overwrite behavior
 * and zero-length value restrictions.
 *
 * Requires Chimaera runtime and CTE client to be initialized before use.
 * CPU-only — not usable from GPU kernels.
 */
class CteBlobStore {
    wrp_cte::core::Tag tag_;

public:
    explicit CteBlobStore(const char* tag_name) : CteBlobStore(std::string_view{tag_name}) {}
    explicit CteBlobStore(std::string_view tag_name);
    explicit CteBlobStore(gpu_string_view tag_name);

    bool PutBlob(cstd::span<const byte_t> key, cstd::span<const byte_t> value);

    cstd::expected<cstd::span<byte_t>, BlobStoreError> GetBlob(
        cstd::span<const byte_t> key, cstd::span<byte_t> value_out);

    bool DeleteBlob(cstd::span<const byte_t> key);

    bool Exists(cstd::span<const byte_t> key);

private:
    static std::string KeyToHex(cstd::span<const byte_t> key);
};

static_assert(RawBlobStore<CteBlobStore>);

} // namespace kvhdf5
