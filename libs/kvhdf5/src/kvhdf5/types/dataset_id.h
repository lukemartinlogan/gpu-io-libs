#pragma once

#include "object_id.h"

namespace kvhdf5 {

struct DatasetId {
    ObjectId id;

    CROSS_FUN constexpr DatasetId() : id() {}
    CROSS_FUN constexpr explicit DatasetId(ObjectId oid) : id(oid) {}
    CROSS_FUN constexpr explicit DatasetId(uint64_t raw) : id(raw) {}

    CROSS_FUN constexpr bool IsValid() const { return id.IsValid(); }
    CROSS_FUN constexpr auto operator<=>(const DatasetId&) const = default;
    CROSS_FUN constexpr bool operator==(const DatasetId&) const = default;
};

} // namespace kvhdf5

// Opt into serde as POD type
KVHDF5_AUTO_SERDE(kvhdf5::DatasetId);
