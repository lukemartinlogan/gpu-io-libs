#pragma once

#include "../defines.h"
#include "blob_store.h"
#include "context.h"
#include <cuda/std/span>

namespace kvhdf5 {

/**
 * Simple in-memory blob store using IOWarp data structures.
 * Uses linear search - suitable for small numbers of blobs.
 *
 * For Phase 1 (CPU-only) validation. Will be replaced with more
 * efficient structures (hash map) in later phases.
 */
class InMemoryBlobStore {
    struct Entry {
        vector<byte_t> key;
        vector<byte_t> value;

        Entry(AllocatorImpl* alloc) : key(alloc), value(alloc) {}

        Entry(cstd::span<const byte_t> k, cstd::span<const byte_t> v, AllocatorImpl* alloc)
            : key(alloc), value(alloc) {
            for (auto b : k) key.push_back(b);
            for (auto b : v) value.push_back(b);
        }
    };

    vector<Entry> entries_;
    AllocatorImpl* allocator_;

public:
    explicit InMemoryBlobStore(AllocatorImpl* alloc)
        : entries_(alloc), allocator_(alloc) {
        KVHDF5_ASSERT(alloc != nullptr, "InMemoryBlobStore: allocator is null");
    }

    /**
     * Store a blob with the given key and value.
     * Overwrites if key already exists.
     */
    bool PutBlob(cstd::span<const byte_t> key, cstd::span<const byte_t> value) {
        // Check if key exists
        for (size_t i = 0; i < entries_.size(); ++i) {
            if (KeysEqual(entries_[i].key, key)) {
                // Overwrite existing value
                entries_[i].value.clear();
                for (auto b : value) entries_[i].value.push_back(b);
                return true;
            }
        }

        // Add new entry
        Entry new_entry(key, value, allocator_);
        entries_.push_back(new_entry);
        return true;
    }

    /**
     * Retrieve a blob by key into a writable buffer.
     * Returns a span of the actual data (subspan of value_out).
     */
    cstd::expected<cstd::span<byte_t>, BlobStoreError> GetBlob(
        cstd::span<const byte_t> key,
        cstd::span<byte_t> value_out) {

        // Find the entry
        for (size_t i = 0; i < entries_.size(); ++i) {
            if (KeysEqual(entries_[i].key, key)) {
                const auto& stored_value = entries_[i].value;

                // Check buffer size
                if (value_out.size() < stored_value.size()) {
                    return cstd::unexpected(BlobStoreError::NotEnoughSpace);
                }

                // Copy value to output buffer
                for (size_t j = 0; j < stored_value.size(); ++j) {
                    value_out[j] = stored_value[j];
                }

                // Return span of actual data
                return cstd::span<byte_t>(value_out.data(), stored_value.size());
            }
        }

        return cstd::unexpected(BlobStoreError::NotExist);
    }

    /**
     * Delete a blob by key.
     */
    bool DeleteBlob(cstd::span<const byte_t> key) {
        for (size_t i = 0; i < entries_.size(); ++i) {
            if (KeysEqual(entries_[i].key, key)) {
                // Remove entry by swapping with last and popping
                if (i < entries_.size() - 1) {
                    entries_[i] = entries_[entries_.size() - 1];
                }
                entries_.pop_back();
                return true;
            }
        }
        return false;
    }

    /**
     * Check if a blob exists by key.
     */
    bool Exists(cstd::span<const byte_t> key) const {
        for (size_t i = 0; i < entries_.size(); ++i) {
            if (KeysEqual(entries_[i].key, key)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Get the number of stored blobs.
     */
    size_t Size() const {
        return entries_.size();
    }

    /**
     * Clear all stored blobs.
     */
    void Clear() {
        entries_.clear();
    }

private:
    static bool KeysEqual(const vector<byte_t>& stored_key, cstd::span<const byte_t> query_key) {
        if (stored_key.size() != query_key.size()) {
            return false;
        }
        for (size_t i = 0; i < stored_key.size(); ++i) {
            if (stored_key[i] != query_key[i]) {
                return false;
            }
        }
        return true;
    }
};

static_assert(RawBlobStore<InMemoryBlobStore>);

} // namespace kvhdf5
