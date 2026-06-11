#pragma once

#include <cstdint>
#include <cstring>
#include <functional>

namespace gs_trad {

enum class Field : uint8_t { U = 0, V = 1 };
enum class Kind  : uint8_t { PingPongA = 0, PingPongB = 1, Snapshot = 2 };

struct BlobKey {
    uint32_t grid_idx;
    uint16_t step;       // 0 for ping-pong slots; otherwise snapshot step number
    Field    field;
    Kind     kind;

    bool operator==(const BlobKey& o) const noexcept {
        return grid_idx == o.grid_idx && step == o.step
            && field == o.field && kind == o.kind;
    }
};
static_assert(sizeof(BlobKey) == 8, "BlobKey must pack to 8 bytes");

struct BlobKeyHash {
    size_t operator()(const BlobKey& k) const noexcept {
        uint64_t bits;
        std::memcpy(&bits, &k, sizeof(bits));
        return std::hash<uint64_t>{}(bits);
    }
};

} // namespace gs_trad
