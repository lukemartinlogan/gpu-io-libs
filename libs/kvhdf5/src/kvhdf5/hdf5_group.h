#pragma once

#include "file.h"
#include "ref.h"
#include "error.h"
#include "hdf5_datatype.h"
#include "dataspace.h"

namespace kvhdf5 {

// Forward declare Dataset<B> - will be defined in hdf5_dataset.h
template<typename B>
class Dataset;

struct DatasetCreateProps {
    cstd::inplace_vector<uint64_t, MAX_DIMS> chunk_dims{};
};

struct GroupInfo {
    size_t num_children;
    size_t num_attributes;
};

template<RawBlobStore B>
class Group {
    GroupId id_;
    Ref<Container<B>> container_;

public:
    CROSS_FUN Group(GroupId id, Ref<Container<B>> container)
        : id_(id), container_(container) {}

    // --- Child groups ---

    expected<Group> CreateGroup(gpu_string_view name) {
        auto meta_result = container_->GetGroup(id_);
        if (!meta_result.has_value()) {
            return make_error(ErrorCode::InvalidArgument, "parent group not found");
        }
        auto meta = meta_result.value();

        // Check no duplicate name
        for (size_t i = 0; i < meta.children.size(); ++i) {
            if (meta.children[i].name == name) {
                return make_error(ErrorCode::InvalidArgument, "child with this name already exists");
            }
        }

        // Allocate child group
        auto child_id = GroupId(container_->AllocateId());
        GroupMetadata child_meta(child_id, container_->GetAllocator());
        container_->PutGroup(child_id, child_meta);

        // Add entry to parent
        meta.children.push_back(GroupEntry::NewGroup(child_id, name));
        container_->PutGroup(id_, meta);

        return Group(child_id, container_);
    }

    expected<Group> OpenGroup(gpu_string_view name) {
        auto meta_result = container_->GetGroup(id_);
        if (!meta_result.has_value()) {
            return make_error(ErrorCode::InvalidArgument, "group not found");
        }
        auto& meta = meta_result.value();

        for (size_t i = 0; i < meta.children.size(); ++i) {
            if (meta.children[i].kind == ChildKind::Group &&
                meta.children[i].name == name) {
                return Group(GroupId(meta.children[i].object_id), container_);
            }
        }

        return make_error(ErrorCode::InvalidArgument, "child group not found");
    }

    // --- Attributes ---

    expected<void> SetAttribute(gpu_string_view name, const Datatype& type, const void* data) {
        auto meta_result = container_->GetGroup(id_);
        if (!meta_result.has_value()) {
            return make_error(ErrorCode::InvalidArgument, "group not found");
        }
        auto meta = meta_result.value();

        // Build attribute value span
        size_t data_size = type.GetSize();
        cstd::span<const byte_t> val_span(
            static_cast<const byte_t*>(data), data_size);
        Attribute attr(name, type.ToRef(), val_span);

        // Replace if exists, else push
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

        container_->PutGroup(id_, meta);
        return {};
    }

    expected<void> GetAttribute(gpu_string_view name, const Datatype& type, void* data) const {
        auto meta_result = container_->GetGroup(id_);
        if (!meta_result.has_value()) {
            return make_error(ErrorCode::InvalidArgument, "group not found");
        }
        auto& meta = meta_result.value();

        for (size_t i = 0; i < meta.attributes.size(); ++i) {
            if (meta.attributes[i].name == name) {
                size_t data_size = type.GetSize();
                auto& val = meta.attributes[i].value;
                // Byte-by-byte copy
                auto* dst = static_cast<byte_t*>(data);
                for (size_t j = 0; j < data_size; ++j) {
                    dst[j] = val[j];
                }
                return {};
            }
        }

        return make_error(ErrorCode::InvalidArgument, "attribute not found");
    }

    bool HasAttribute(gpu_string_view name) const {
        auto meta_result = container_->GetGroup(id_);
        if (!meta_result.has_value()) return false;
        auto& meta = meta_result.value();

        for (size_t i = 0; i < meta.attributes.size(); ++i) {
            if (meta.attributes[i].name == name) {
                return true;
            }
        }
        return false;
    }

    // --- Query ---

    CROSS_FUN GroupId GetId() const { return id_; }

    expected<GroupInfo> GetInfo() const {
        auto meta_result = container_->GetGroup(id_);
        if (!meta_result.has_value()) {
            return make_error(ErrorCode::InvalidArgument, "group not found");
        }
        auto& meta = meta_result.value();
        return GroupInfo{meta.children.size(), meta.attributes.size()};
    }

    // --- Datasets (declarations; defined in hdf5_dataset.h after Dataset<B>) ---
    expected<Dataset<B>> CreateDataset(
        gpu_string_view name, const Datatype& type,
        const Dataspace& space, const DatasetCreateProps& props = {});
    expected<Dataset<B>> OpenDataset(gpu_string_view name);

    // expected<Dataset<B>> CreateDataset(const char* name, const Datatype& type,
    //     const Dataspace& space, const DatasetCreateProps& props = {});
    // expected<Dataset<B>> OpenDataset(const char* name);
};

// Define File<B>::OpenRootGroup now that Group<B> is complete
template<RawBlobStore B>
Group<B> File<B>::OpenRootGroup() {
    return Group<B>(container_.RootGroup(), Ref<Container<B>>(container_));
}

} // namespace kvhdf5
