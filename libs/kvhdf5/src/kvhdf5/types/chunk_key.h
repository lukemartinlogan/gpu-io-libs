#pragma once

#include "dataset_id.h"
#include "../../defines.h"
#include "../../serde.h"
#include <cuda/std/inplace_vector>
#include <cuda/std/span>

namespace kvhdf5 {

struct ChunkKey {
    DatasetId dataset;
    cstd::inplace_vector<uint64_t, MAX_DIMS> coords;

    CROSS_FUN constexpr ChunkKey() : dataset(), coords() {}

    CROSS_FUN ChunkKey(DatasetId ds, cstd::span<const uint64_t> c)
        : dataset(ds), coords(c.begin(), c.end()) {
        KVHDF5_ASSERT(c.size() <= MAX_DIMS, "ChunkKey: span size exceeds MAX_DIMS");
    }

    CROSS_FUN uint8_t ndims() const { return static_cast<uint8_t>(coords.size()); }

    CROSS_FUN constexpr bool operator==(const ChunkKey& other) const {
        return dataset == other.dataset && coords == other.coords;
    }

    CROSS_FUN constexpr bool operator!=(const ChunkKey& other) const {
        return !(*this == other);
    }

    // Lexicographic ordering: first by dataset, then coords
    CROSS_FUN constexpr bool operator<(const ChunkKey& other) const {
        if (dataset != other.dataset) return dataset < other.dataset;
        return coords < other.coords;
    }

    CROSS_FUN constexpr bool operator<=(const ChunkKey& other) const {
        return !(other < *this);
    }

    CROSS_FUN constexpr bool operator>(const ChunkKey& other) const {
        return other < *this;
    }

    CROSS_FUN constexpr bool operator>=(const ChunkKey& other) const {
        return !(*this < other);
    }
};

} // namespace kvhdf5

// inplace_vector is POD-compatible, so we can use auto serde
KVHDF5_AUTO_SERDE(kvhdf5::ChunkKey);
