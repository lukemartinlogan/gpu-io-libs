#pragma once

#include "hdf5_dataset.h"  // pulls in hdf5_group.h, container.h, etc.

namespace kvhdf5 {

/// Public handle to a named attribute on a group or dataset.
///
/// Holds a non-owning reference to the container and records which parent
/// object owns the attribute (by ObjectId) and whether that parent is a group
/// or a dataset.  Write/Read round-trip through the parent's metadata so the
/// attribute is always persisted in the blob store.
template<RawBlobStore B>
class AttributeHandle {
    gpu_string<255> name_;
    ObjectId        parent_id_;
    bool            parent_is_group_;
    Ref<Container<B>> container_;

    // --- shared attribute write logic ---

    expected<void> WriteToAttributes(
        vector<Attribute>& attrs,
        const Datatype& type,
        const void* data
    ) {
        size_t data_size = type.GetSize();
        cstd::span<const byte_t> val_span(
            static_cast<const byte_t*>(data), data_size);
        Attribute attr(gpu_string_view(name_), type.ToRef(), val_span);

        for (size_t i = 0; i < attrs.size(); ++i) {
            if (attrs[i].name == name_) {
                attrs[i] = attr;
                return {};
            }
        }
        attrs.push_back(attr);
        return {};
    }

    expected<void> ReadFromAttributes(
        const vector<Attribute>& attrs,
        const Datatype& type,
        void* data
    ) const {
        for (size_t i = 0; i < attrs.size(); ++i) {
            if (attrs[i].name == name_) {
                size_t data_size = type.GetSize();
                auto& val = attrs[i].value;
                auto* dst = static_cast<byte_t*>(data);
                for (size_t j = 0; j < data_size; ++j) {
                    dst[j] = val[j];
                }
                return {};
            }
        }
        return make_error(ErrorCode::InvalidArgument, "attribute not found");
    }

public:
    CROSS_FUN AttributeHandle(
        gpu_string_view name,
        ObjectId parent_id,
        bool parent_is_group,
        Ref<Container<B>> container
    )
        : name_(name)
        , parent_id_(parent_id)
        , parent_is_group_(parent_is_group)
        , container_(container)
    {}

    gpu_string_view GetName() const { return gpu_string_view(name_); }

    expected<void> Write(const Datatype& type, const void* data) {
        if (parent_is_group_) {
            GroupId gid(parent_id_);
            auto meta_result = container_->GetGroup(gid);
            if (!meta_result.has_value()) {
                return make_error(ErrorCode::InvalidArgument, "parent group not found");
            }
            auto meta = meta_result.value();
            auto result = WriteToAttributes(meta.attributes, type, data);
            if (!result.has_value()) return result;
            container_->PutGroup(gid, meta);
            return {};
        } else {
            DatasetId did(parent_id_);
            auto meta_result = container_->GetDataset(did);
            if (!meta_result.has_value()) {
                return make_error(ErrorCode::InvalidArgument, "parent dataset not found");
            }
            auto meta = meta_result.value();
            auto result = WriteToAttributes(meta.attributes, type, data);
            if (!result.has_value()) return result;
            container_->PutDataset(did, meta);
            return {};
        }
    }

    expected<void> Read(const Datatype& type, void* data) const {
        if (parent_is_group_) {
            GroupId gid(parent_id_);
            auto meta_result = container_->GetGroup(gid);
            if (!meta_result.has_value()) {
                return make_error(ErrorCode::InvalidArgument, "parent group not found");
            }
            auto& meta = meta_result.value();
            return ReadFromAttributes(meta.attributes, type, data);
        } else {
            DatasetId did(parent_id_);
            auto meta_result = container_->GetDataset(did);
            if (!meta_result.has_value()) {
                return make_error(ErrorCode::InvalidArgument, "parent dataset not found");
            }
            auto& meta = meta_result.value();
            return ReadFromAttributes(meta.attributes, type, data);
        }
    }
};

// --- Group<B>::OpenAttribute ---
// Defined here because AttributeHandle<B> is now complete.

template<RawBlobStore B>
AttributeHandle<B> Group<B>::OpenAttribute(gpu_string_view name) {
    return AttributeHandle<B>(name, id_.id, true, container_);
}

// --- Dataset<B> attribute methods and OpenAttribute ---
// Defined here because AttributeHandle<B> is now complete.

template<RawBlobStore B>
expected<void> Dataset<B>::SetAttribute(
    gpu_string_view name, const Datatype& type, const void* data
) {
    auto meta_result = container_->GetDataset(id_);
    if (!meta_result.has_value()) {
        return make_error(ErrorCode::InvalidArgument, "dataset not found");
    }
    auto meta = meta_result.value();

    size_t data_size = type.GetSize();
    cstd::span<const byte_t> val_span(
        static_cast<const byte_t*>(data), data_size);
    Attribute attr(name, type.ToRef(), val_span);

    bool found = false;
    for (size_t i = 0; i < meta.attributes.size(); ++i) {
        if (meta.attributes[i].name == name) {
            meta.attributes[i] = attr;
            found = true;
            break;
        }
    }
    if (!found) {
        meta.attributes.push_back(attr);
    }

    container_->PutDataset(id_, meta);
    return {};
}

template<RawBlobStore B>
expected<void> Dataset<B>::GetAttribute(
    gpu_string_view name, const Datatype& type, void* data
) const {
    auto meta_result = container_->GetDataset(id_);
    if (!meta_result.has_value()) {
        return make_error(ErrorCode::InvalidArgument, "dataset not found");
    }
    auto& meta = meta_result.value();

    for (size_t i = 0; i < meta.attributes.size(); ++i) {
        if (meta.attributes[i].name == name) {
            size_t data_size = type.GetSize();
            auto& val = meta.attributes[i].value;
            auto* dst = static_cast<byte_t*>(data);
            for (size_t j = 0; j < data_size; ++j) {
                dst[j] = val[j];
            }
            return {};
        }
    }

    return make_error(ErrorCode::InvalidArgument, "attribute not found");
}

template<RawBlobStore B>
bool Dataset<B>::HasAttribute(gpu_string_view name) const {
    auto meta_result = container_->GetDataset(id_);
    if (!meta_result.has_value()) return false;
    auto& meta = meta_result.value();

    for (size_t i = 0; i < meta.attributes.size(); ++i) {
        if (meta.attributes[i].name == name) return true;
    }
    return false;
}

template<RawBlobStore B>
AttributeHandle<B> Dataset<B>::OpenAttribute(gpu_string_view name) {
    return AttributeHandle<B>(name, id_.id, false, container_);
}

} // namespace kvhdf5
