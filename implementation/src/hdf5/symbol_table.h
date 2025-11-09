#pragma once

#include <array>
#include <string>
#include <vector>

#include "local_heap.h"
#include "types.h"
#include "../serialization/serialization.h"
#include "gpu_string.h"

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
    cstd::array<byte_t, 16> scratch_pad_space{};

    void Serialize(Serializer& s) const;

    static hdf5::expected<SymbolTableEntry> Deserialize(Deserializer& de);
};

struct SymbolTableNode {
    std::vector<SymbolTableEntry> entries;

    [[nodiscard]] hdf5::expected<cstd::optional<offset_t>> FindEntry(hdf5::string_view name, const LocalHeap& heap, Deserializer& de) const;

    void Serialize(Serializer& s) const;

    static hdf5::expected<SymbolTableNode> Deserialize(Deserializer& de);
private:
    static constexpr uint8_t kVersionNumber = 0x01;
    static constexpr cstd::array<uint8_t, 4> kSignature = { 'S', 'N', 'O', 'D' };
};

