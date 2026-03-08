#include "cte_blob_store.h"
#include <cstring>

namespace kvhdf5 {

// Size prefix stored before every blob value.
static constexpr size_t kPrefixSize = sizeof(uint64_t);

// Stack buffer for Put/Get. Avoids heap allocation for most metadata blobs.
// Must exceed BlobStore::DefaultMaxValueSize (1024) + kPrefixSize.
static constexpr size_t kStackBufSize = 2048;

CteBlobStore::CteBlobStore(std::string_view tag_name)
    : tag_(std::string{tag_name}) {}

CteBlobStore::CteBlobStore(gpu_string_view tag_name)
    : tag_(std::string(tag_name.data(), tag_name.size())) {}

bool CteBlobStore::PutBlob(cstd::span<const byte_t> key,
                           cstd::span<const byte_t> value) {
    std::string blob_name = KeyToHex(key);

    // Delete first: CTE's internal overwrite-with-truncation is slower
    // than delete + fresh create.
    if (tag_.GetBlobSize(blob_name) > 0) {
        auto task = WRP_CTE_CLIENT->AsyncDelBlob(tag_.GetTagId(), blob_name);
        task.Wait();
    }

    // Build buffer: [uint64_t real_size][data bytes]
    uint64_t real_size = value.size();
    size_t total = kPrefixSize + real_size;

    // Use stack buffer for small values, heap for large ones
    char stack_buf[kStackBufSize];
    char* buf;
    std::vector<char> heap_buf;
    if (total <= kStackBufSize) {
        buf = stack_buf;
    } else {
        heap_buf.resize(total);
        buf = heap_buf.data();
    }

    std::memcpy(buf, &real_size, kPrefixSize);
    if (real_size > 0) {
        std::memcpy(buf + kPrefixSize, value.data(), real_size);
    }

    tag_.PutBlob(blob_name, buf, total);
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

    if (stored_size < kPrefixSize) {
        // Malformed blob — too small to contain even the size prefix
        return cstd::unexpected(BlobStoreError::NotExist);
    }

    // Read entire blob (prefix + data) in ONE CTE call instead of two.
    char stack_buf[kStackBufSize];
    char* read_buf;
    std::vector<char> heap_buf;

    if (stored_size <= kStackBufSize) {
        read_buf = stack_buf;
    } else {
        heap_buf.resize(stored_size);
        read_buf = heap_buf.data();
    }

    tag_.GetBlob(blob_name, read_buf, stored_size, 0);

    // Extract size prefix and validate
    uint64_t real_size;
    std::memcpy(&real_size, read_buf, kPrefixSize);

    if (real_size > stored_size - kPrefixSize) {
        // Corrupted blob: prefix claims more data than is stored
        return cstd::unexpected(BlobStoreError::NotExist);
    }

    if (value_out.size() < real_size) {
        return cstd::unexpected(BlobStoreError::NotEnoughSpace);
    }

    if (real_size > 0) {
        std::memcpy(value_out.data(), read_buf + kPrefixSize, real_size);
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

void CteBlobStore::Destroy() {
    auto task = WRP_CTE_CLIENT->AsyncDelTag(tag_.GetTagId());
    task.Wait();
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
