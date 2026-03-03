#pragma once

#include "dataset_id.h"
#include "../../defines.h"
#include "../../serde.h"
#include <utils/algorithms.h>
#include <cuda/std/array>
#include <cuda/std/span>

namespace kvhdf5 {

struct ChunkKey {
    DatasetId dataset{};
    uint8_t ndims_{0};
    padding<7> _pad;
    cstd::array<uint64_t, MAX_DIMS> coords{};

    CROSS_FUN constexpr ChunkKey() = default;

    CROSS_FUN ChunkKey(DatasetId ds, cstd::span<const uint64_t> c)
        : dataset(ds), ndims_(static_cast<uint8_t>(c.size()))
    {
        KVHDF5_ASSERT(c.size() <= MAX_DIMS, "ChunkKey: span size exceeds MAX_DIMS");
        algorithms::copy(c, cstd::span(coords.data(), c.size()));
    }

    CROSS_FUN uint8_t ndims() const { return ndims_; }

    CROSS_FUN cstd::span<const uint64_t> Coords() const {
        return cstd::span(coords.data(), ndims_);
    }

    CROSS_FUN cstd::span<uint64_t> Coords() {
        return cstd::span(coords.data(), ndims_);
    }

    CROSS_FUN constexpr bool operator==(const ChunkKey& other) const {
        if (dataset != other.dataset || ndims_ != other.ndims_) return false;

        auto a = Coords();
        auto b = other.Coords();

        return cstd::equal(a.begin(), a.end(), b.begin());
    }

    CROSS_FUN constexpr bool operator!=(const ChunkKey& other) const {
        return !(*this == other);
    }

    CROSS_FUN constexpr bool operator<(const ChunkKey& other) const {
        if (dataset != other.dataset) return dataset < other.dataset;
        if (ndims_ != other.ndims_) return ndims_ < other.ndims_;
        
        auto a = Coords();
        auto b = other.Coords();
        
        return cstd::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end());
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

KVHDF5_AUTO_SERDE(kvhdf5::ChunkKey);
