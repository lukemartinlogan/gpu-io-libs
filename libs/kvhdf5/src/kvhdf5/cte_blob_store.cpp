#include "cte_blob_store.h"
#include <cstring>

namespace kvhdf5 {

CteBlobStore::CteBlobStore(std::string_view tag_name)
    : tag_(std::string{tag_name}) {}

CteBlobStore::CteBlobStore(gpu_string_view tag_name)
    : tag_(std::string(tag_name.data(), tag_name.size())) {}

bool CteBlobStore::PutBlob(cstd::span<const byte_t> key,
                           cstd::span<const byte_t> value) {
    std::string blob_name = KeyToHex(key);

    // Delete first to handle CTE's no-truncate-on-overwrite behavior.
    // GetBlobSize returns 0 for non-existent blobs, so this is safe.
    if (tag_.GetBlobSize(blob_name) > 0) {
        auto task = WRP_CTE_CLIENT->AsyncDelBlob(tag_.GetTagId(), blob_name);
        task.Wait();
    }

    // Build buffer: [uint64_t real_size][data bytes]
    uint64_t real_size = value.size();
    size_t total = sizeof(real_size) + real_size;
    std::vector<char> buffer(total);
    std::memcpy(buffer.data(), &real_size, sizeof(real_size));
    if (real_size > 0) {
        std::memcpy(buffer.data() + sizeof(real_size), value.data(), real_size);
    }

    tag_.PutBlob(blob_name, buffer.data(), buffer.size());
    return true;
}

cstd::expected<cstd::span<byte_t>, BlobStoreError>
CteBlobStore::GetBlob(cstd::span<const byte_t> key,
                      cstd::span<byte_t> value_out) {
    std::string blob_name = KeyToHex(key);

    chi::u64 stored_size = tag_.GetBlobSize(blob_name);
    if (stored_size == 0) {
        return cstd::unexpected(BlobStoreError::NotExist);
    }

    // Read the size prefix
    uint64_t real_size;
    tag_.GetBlob(blob_name, reinterpret_cast<char*>(&real_size),
                 sizeof(real_size), 0);

    if (value_out.size() < real_size) {
        return cstd::unexpected(BlobStoreError::NotEnoughSpace);
    }

    if (real_size > 0) {
        tag_.GetBlob(blob_name, reinterpret_cast<char*>(value_out.data()),
                     real_size, sizeof(real_size));
    }

    return cstd::span<byte_t>(value_out.data(), real_size);
}

bool CteBlobStore::DeleteBlob(cstd::span<const byte_t> key) {
    std::string blob_name = KeyToHex(key);

    if (tag_.GetBlobSize(blob_name) == 0) {
        return false;
    }

    auto task = WRP_CTE_CLIENT->AsyncDelBlob(tag_.GetTagId(), blob_name);
    task.Wait();
    return true;
}

bool CteBlobStore::Exists(cstd::span<const byte_t> key) {
    std::string blob_name = KeyToHex(key);
    return tag_.GetBlobSize(blob_name) > 0;
}

std::string CteBlobStore::KeyToHex(cstd::span<const byte_t> key) {
    if (key.empty()) {
        return "_";
    }

    static constexpr char hex_chars[] = "0123456789abcdef";
    std::string result;
    result.reserve(key.size() * 2);
    for (auto b : key) {
        uint8_t byte = static_cast<uint8_t>(b);
        result += hex_chars[byte >> 4];
        result += hex_chars[byte & 0x0F];
    }
    return result;
}

} // namespace kvhdf5
