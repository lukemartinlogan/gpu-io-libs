#pragma once

#include "defines.h"
#include "types.h"
#include "dataset.h"
#include "allocator.h"
#include "context.h"
#include <utils/gpu_string.h>

namespace kvhdf5 {

enum class ChildKind : uint8_t {
    Group,
    Dataset,
};

struct GroupEntry {
    ChildKind kind;
    ObjectId object_id;
    gpu_string<255> name;

    CROSS_FUN static GroupEntry NewGroup(GroupId id, gpu_string_view n) {
        GroupEntry entry;
        entry.kind = ChildKind::Group;
        entry.object_id = id.id;
        entry.name = gpu_string<255>(n);
        return entry;
    }

    CROSS_FUN static GroupEntry NewDataset(DatasetId id, gpu_string_view n) {
        GroupEntry entry;
        entry.kind = ChildKind::Dataset;
        entry.object_id = id.id;
        entry.name = gpu_string<255>(n);
        return entry;
    }

private:
    CROSS_FUN GroupEntry() : kind(ChildKind::Group), object_id(), name() {}

    CROSS_FUN GroupEntry(ChildKind k, ObjectId oid, const gpu_string<255>& n)
        : kind(k), object_id(oid), name(n) {}

public:
    template<serde::Serializer S>
    CROSS_FUN void Serialize(S& s) const {
        serde::Write(s, kind);
        serde::Write(s, object_id);
        name.Serialize(s);
    }

    template<serde::Deserializer D>
    CROSS_FUN static GroupEntry Deserialize(D& d) {
        GroupEntry entry;
        entry.kind = serde::Read<ChildKind>(d);
        entry.object_id = serde::Read<ObjectId>(d);
        entry.name = gpu_string<255>::Deserialize(d);
        return entry;
    }

    CROSS_FUN bool operator==(const GroupEntry& other) const {
        return kind == other.kind &&
               object_id == other.object_id &&
               name == other.name;
    }
};

struct GroupMetadata {
    GroupId id;
    vector<GroupEntry> children;
    vector<Attribute> attributes;

    template<Allocator A>
    CROSS_FUN GroupMetadata(GroupId id_, A& alloc)
        : id(id_),
          children(&alloc),
          attributes(&alloc)
    {}
    
    CROSS_FUN GroupMetadata(GroupId id_, vector<GroupEntry> children_, vector<Attribute> attributes_)
        : id(id_),
          children(cstd::move(children_)),
          attributes(cstd::move(attributes_))
    {}

    template<serde::Serializer S>
    CROSS_FUN void Serialize(S& s) const {
        serde::Write(s, id);

        uint32_t child_count = static_cast<uint32_t>(children.size());
        serde::Write(s, child_count);
        for (uint32_t i = 0; i < child_count; ++i) {
            children[i].Serialize(s);
        }

        uint32_t attr_count = static_cast<uint32_t>(attributes.size());
        serde::Write(s, attr_count);
        for (uint32_t i = 0; i < attr_count; ++i) {
            attributes[i].Serialize(s);
        }
    }

    template<serde::Deserializer D, ProvidesAllocator Ctx>
    CROSS_FUN static GroupMetadata Deserialize(D& d, Ctx& ctx) {
        GroupId id = serde::Read<GroupId>(d);

        uint32_t child_count = serde::Read<uint32_t>(d);
        vector<GroupEntry> children(&ctx.GetAllocator());
        for (uint32_t i = 0; i < child_count; ++i) {
            children.push_back(GroupEntry::Deserialize(d));
        }

        uint32_t attr_count = serde::Read<uint32_t>(d);
        vector<Attribute> attributes(&ctx.GetAllocator());
        for (uint32_t i = 0; i < attr_count; ++i) {
            attributes.push_back(Attribute::Deserialize(d));
        }

        return GroupMetadata{id, children, attributes};
    }

    CROSS_FUN bool operator==(const GroupMetadata& other) const {
        if (id != other.id) return false;
        if (children.size() != other.children.size()) return false;
        for (size_t i = 0; i < children.size(); ++i) {
            if (!(children[i] == other.children[i])) return false;
        }
        if (attributes.size() != other.attributes.size()) return false;
        for (size_t i = 0; i < attributes.size(); ++i) {
            if (!(attributes[i] == other.attributes[i])) return false;
        }
        return true;
    }
};

} // namespace kvhdf5

KVHDF5_AUTO_SERDE(kvhdf5::ChildKind);
