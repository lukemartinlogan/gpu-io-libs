#pragma once

#include "object_id.h"

namespace kvhdf5 {

struct DatatypeId {
    ObjectId id;

    CROSS_FUN constexpr DatatypeId() : id() {}
    CROSS_FUN constexpr explicit DatatypeId(ObjectId oid) : id(oid) {}
    CROSS_FUN constexpr explicit DatatypeId(uint64_t raw) : id(raw) {}

    CROSS_FUN constexpr bool IsValid() const { return id.IsValid(); }
    CROSS_FUN constexpr auto operator<=>(const DatatypeId&) const = default;
    CROSS_FUN constexpr bool operator==(const DatatypeId&) const = default;
};

} // namespace kvhdf5

KVHDF5_AUTO_SERDE(kvhdf5::DatatypeId);
