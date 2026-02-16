#pragma once

#include "object_id.h"

namespace kvhdf5 {

struct GroupId {
    ObjectId id;

    CROSS_FUN constexpr GroupId() : id() {}
    CROSS_FUN constexpr explicit GroupId(ObjectId oid) : id(oid) {}
    CROSS_FUN constexpr explicit GroupId(uint64_t raw) : id(raw) {}

    CROSS_FUN constexpr uint64_t Id() const { return id.id; }
    CROSS_FUN constexpr bool IsValid() const { return id.IsValid(); }
    CROSS_FUN constexpr auto operator<=>(const GroupId&) const = default;
    CROSS_FUN constexpr bool operator==(const GroupId&) const = default;
};

} // namespace kvhdf5

KVHDF5_AUTO_SERDE(kvhdf5::GroupId);
