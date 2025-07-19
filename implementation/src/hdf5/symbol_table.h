#pragma once

#include <array>
#include <vector>

#include "types.h"
#include "../serialization/serialization.h"

enum class SymbolTableEntryCacheType {
    // no data cached, this is guaranteed when link ct > 1
    kNoDataCached = 0,
    // group object header metadata cached in scratch
    kGroupObjectHeaderMetadata = 1,
    // entry is symlink
    // first 4 bytes of scratch are offset into local heap for link value
    // object header addr will be undefined
    kSymbolicLink = 2,
};

struct SymbolTableEntry {
    // offset into group local heap
    // for name of link, null terminated
    offset_t link_name_offset = 0;
    // permanent address of object's metadata,
    // some metadata may be cached in scratch space
    offset_t object_header_addr = 0;
    // cache type, determined from object header
    SymbolTableEntryCacheType cache_ty{};
    // scratch pad space
    std::array<byte_t, 16> scratch_pad_space{};

    void Serialize(Serializer& s) const;

    static SymbolTableEntry Deserialize(Deserializer& de);
};

struct SymbolTableNode {
    std::vector<SymbolTableEntry> entries;

    void Serialize(Serializer& s) const;

    static SymbolTableNode Deserialize(Deserializer& de);
private:
    static constexpr uint8_t kVersionNumber = 0x01;
    static constexpr std::array<uint8_t, 4> kSignature = { 'S', 'N', 'O', 'D' };
};

