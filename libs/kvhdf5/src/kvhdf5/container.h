#pragma once

#include "../defines.h"
#include "blob_store.h"
#include "context.h"
#include "types.h"
#include "group.h"
#include "dataset.h"
#include "datatype.h"
#include <cuda/std/atomic>

namespace kvhdf5 {


template<RawBlobStore BlobStoreImpl>
class Container {
    BlobStoreImpl raw_store_;
    BlobStore<BlobStoreImpl> store_;
    Context context_;
    GroupId root_group_;
    cstd::atomic<uint64_t> next_object_id_;

public:
    /**
     * Create a new container.
     * Initializes the root group and sets up ID allocation.
     */
    explicit Container(AllocatorImpl* alloc)
        : raw_store_(alloc)
        , store_(&raw_store_)
        , context_(alloc)
        , next_object_id_(1)  // 0 is reserved for invalid/null
    {
        KVHDF5_ASSERT(alloc != nullptr, "Container: allocator is null");

        // Allocate root group ID
        root_group_ = GroupId(AllocateId());

        // Create empty root group metadata
        GroupMetadata root_metadata{
            root_group_,
            vector<GroupEntry>(alloc),
            vector<Attribute>(alloc)
        };

        // Store root group
        bool success = PutGroup(root_group_, root_metadata);
        KVHDF5_ASSERT(success, "Container: failed to store root group");
    }

    /**
     * Allocate a new unique object ID.
     * Thread-safe via atomic increment.
     */
    CROSS_FUN ObjectId AllocateId() {
        uint64_t id = next_object_id_.fetch_add(1, cstd::memory_order_relaxed);
        return ObjectId(id);
    }

    // ========================================================================
    // Group Operations
    // ========================================================================

    /**
     * Store group metadata by ID.
     * Uses custom serializer for non-POD GroupMetadata.
     */
    CROSS_FUN bool PutGroup(GroupId id, const GroupMetadata& metadata) {
        auto serialize_fn = [](serde::BufferReaderWriter& writer, const GroupMetadata& meta) {
            meta.Serialize(writer);
        };

        return store_.template PutBlob<GroupId, GroupMetadata>(id, metadata, serialize_fn);
    }

    /**
     * Retrieve group metadata by ID.
     * Returns BlobStoreError::NotExist if group doesn't exist.
     */
    CROSS_FUN cstd::expected<GroupMetadata, BlobStoreError> GetGroup(GroupId id) {
        auto deserialize_fn = [this](serde::BufferDeserializer& reader) -> GroupMetadata {
            return GroupMetadata::Deserialize(reader, context_);
        };

        return store_.template GetBlob<GroupId, GroupMetadata>(id, deserialize_fn);
    }

    /**
     * Delete group metadata by ID.
     */
    CROSS_FUN bool DeleteGroup(GroupId id) {
        return store_.DeleteBlob(id);
    }

    /**
     * Check if group exists by ID.
     */
    CROSS_FUN bool GroupExists(GroupId id) {
        return store_.Exists(id);
    }

    // ========================================================================
    // Dataset Operations
    // ========================================================================

    /**
     * Store dataset metadata by ID.
     * Uses custom serializer for non-POD DatasetMetadata.
     */
    CROSS_FUN bool PutDataset(DatasetId id, const DatasetMetadata& metadata) {
        auto serialize_fn = [](serde::BufferReaderWriter& writer, const DatasetMetadata& meta) {
            meta.Serialize(writer);
        };

        return store_.template PutBlob<DatasetId, DatasetMetadata>(id, metadata, serialize_fn);
    }

    /**
     * Retrieve dataset metadata by ID.
     * Returns BlobStoreError::NotExist if dataset doesn't exist.
     */
    CROSS_FUN cstd::expected<DatasetMetadata, BlobStoreError> GetDataset(DatasetId id) {
        auto deserialize_fn = [this](serde::BufferDeserializer& reader) -> DatasetMetadata {
            return DatasetMetadata::Deserialize(reader, context_);
        };

        return store_.template GetBlob<DatasetId, DatasetMetadata>(id, deserialize_fn);
    }

    /**
     * Delete dataset metadata by ID.
     */
    CROSS_FUN bool DeleteDataset(DatasetId id) {
        return store_.DeleteBlob(id);
    }

    /**
     * Check if dataset exists by ID.
     */
    CROSS_FUN bool DatasetExists(DatasetId id) {
        return store_.Exists(id);
    }

    // ========================================================================
    // Datatype Operations
    // ========================================================================

    /**
     * Store complex datatype descriptor by ID.
     * ComplexDatatypeDescriptor is POD, so uses simple serialization.
     */
    CROSS_FUN bool PutDatatype(DatatypeId id, const ComplexDatatypeDescriptor& descriptor) {
        return store_.template PutBlob<DatatypeId, ComplexDatatypeDescriptor>(id, descriptor);
    }

    /**
     * Retrieve complex datatype descriptor by ID.
     * Returns BlobStoreError::NotExist if datatype doesn't exist.
     */
    CROSS_FUN cstd::expected<ComplexDatatypeDescriptor, BlobStoreError> GetDatatype(DatatypeId id) {
        return store_.template GetBlob<DatatypeId, ComplexDatatypeDescriptor>(id);
    }

    /**
     * Delete complex datatype descriptor by ID.
     */
    CROSS_FUN bool DeleteDatatype(DatatypeId id) {
        return store_.DeleteBlob(id);
    }

    /**
     * Check if complex datatype exists by ID.
     */
    CROSS_FUN bool DatatypeExists(DatatypeId id) {
        return store_.Exists(id);
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    /**
     * Get the root group ID.
     * The root group always exists and is created during container initialization.
     */
    CROSS_FUN GroupId RootGroup() const {
        return root_group_;
    }

    /**
     * Get the context (for allocator access).
     */
    CROSS_FUN Context& GetContext() {
        return context_;
    }

    CROSS_FUN const Context& GetContext() const {
        return context_;
    }

    /**
     * Get access to the underlying blob store (for advanced usage).
     */
    CROSS_FUN BlobStore<BlobStoreImpl>& GetBlobStore() {
        return store_;
    }

    CROSS_FUN const BlobStore<BlobStoreImpl>& GetBlobStore() const {
        return store_;
    }
};

} // namespace kvhdf5
