#pragma once

#include "../../defines.h"
#include "../../serde.h"

namespace kvhdf5 {

struct ObjectId {
    uint64_t id;  // 0 = null/invalid

    CROSS_FUN constexpr ObjectId() : id(0) {}
    CROSS_FUN constexpr explicit ObjectId(uint64_t i) : id(i) {}

    CROSS_FUN constexpr bool IsValid() const { return id != 0; }
    CROSS_FUN constexpr auto operator<=>(const ObjectId&) const = default;
    CROSS_FUN constexpr bool operator==(const ObjectId&) const = default;
};

} // namespace kvhdf5

KVHDF5_AUTO_SERDE(kvhdf5::ObjectId);
