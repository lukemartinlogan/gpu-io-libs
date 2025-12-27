#pragma once

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

    template<serde::Serializer S>
    __device__
    void Serialize(S& s) const {
        serde::Write(s, link_name_offset);
        serde::Write(s, object_header_addr);
        serde::Write(s, static_cast<uint32_t>(cache_ty));
        // 4 bytes to align scratch pad on 16 byte boundary
        serde::Write(s, uint32_t{0});
        serde::Write(s, scratch_pad_space);
    }

    template<serde::Deserializer D>
    __device__
    static hdf5::expected<SymbolTableEntry> Deserialize(D& de) {
        SymbolTableEntry ent{};

        ent.link_name_offset = serde::Read<offset_t>(de);
        ent.object_header_addr = serde::Read<offset_t>(de);
        ent.cache_ty = static_cast<SymbolTableEntryCacheType>(serde::Read<uint32_t>(de));
        // 4 bytes to align scratch pad on 16 byte boundary
        serde::Skip<uint32_t>(de);
        ent.scratch_pad_space = serde::Read<cstd::array<byte_t, 16>>(de);

        constexpr uint8_t kCacheTyAllowedValues = 3;
        if (static_cast<uint8_t>(ent.cache_ty) >= kCacheTyAllowedValues) {
            return hdf5::error(hdf5::HDF5ErrorCode::InvalidDataValue, "Symbol Table Entry had invalid cache type");
        }

        if (ent.cache_ty == SymbolTableEntryCacheType::kSymbolicLink && ent.object_header_addr != kUndefinedOffset) {
            return hdf5::error(hdf5::HDF5ErrorCode::InvalidDataValue, "If symbol table entry cache type is symbolic link, object header addr should be undefined");
        }

        return ent;
    }

    static constexpr uint16_t kSerializedSizeBytes = sizeof(offset_t) * 5;
};

struct SymbolTableNode {
    // TODO: not sure about this size, may need to be increased or max have a fixed size max
    // might need to be bigger? but group_leaf_node_k is ~ 4
    static constexpr size_t kMaxSymbolTableEntries = 16;

    cstd::inplace_vector<SymbolTableEntry, kMaxSymbolTableEntries> entries;

    template<serde::Deserializer D>
    __device__
    [[nodiscard]] hdf5::expected<cstd::optional<offset_t>> FindEntry(hdf5::string_view name, const LocalHeap& heap, D& de) const {
        for (const auto& entry : entries) {
            // TODO(cuda_vector): this likely doesn't need to allocate if only used to check; might be a lifetime nightmare if made generally though
            auto entry_name = heap.ReadString(entry.link_name_offset, de);
            if (!entry_name) {
                return cstd::unexpected(entry_name.error());
            }

            if (*entry_name == name) {
                return entry.object_header_addr;
            }
        }

        return cstd::nullopt;
    }

    template<serde::Serializer S>
    __device__
    void Serialize(S& s) const {
        serde::Write(s, cstd::array<uint8_t, 4>{ 'S', 'N', 'O', 'D' });
        serde::Write(s, static_cast<uint8_t>(0x01));
        serde::Write(s, uint8_t{0});
        serde::Write(s, static_cast<uint16_t>(entries.size()));

        // TODO: does this need to write the extra unused entries?
        for (const SymbolTableEntry& entry : entries) {
            serde::Write(s, entry);
        }
    }

    template<serde::Deserializer D>
    __device__
    static hdf5::expected<SymbolTableNode> Deserialize(D& de) {
        if (serde::Read<cstd::array<uint8_t, 4>>(de) != cstd::array<uint8_t, 4>{ 'S', 'N', 'O', 'D' }) {
            return hdf5::error(hdf5::HDF5ErrorCode::InvalidSignature, "symbol table node signature was invalid");
        }

        if (serde::Read<uint8_t>(de) != static_cast<uint8_t>(0x01)) {
            return hdf5::error(hdf5::HDF5ErrorCode::InvalidVersion, "symbol table node version was invalid");
        }

        // reserved (zero)
        serde::Skip<uint8_t>(de);

        // actual data
        auto num_symbols = serde::Read<uint16_t>(de);

        SymbolTableNode node{};

        if (num_symbols > kMaxSymbolTableEntries) {
            return hdf5::error(
                hdf5::HDF5ErrorCode::CapacityExceeded,
                "Symbol table node has too many entries"
            );
        }

        for (uint16_t i = 0; i < num_symbols; ++i) {
            auto entry_result = serde::Read<SymbolTableEntry>(de);
            if (!entry_result) return cstd::unexpected(entry_result.error());
            node.entries.push_back(*entry_result);
        }

        return node;
    }

    static constexpr uint16_t kMaxSerializedSize = SymbolTableEntry::kSerializedSizeBytes * kMaxSymbolTableEntries + 8;

private:
    static constexpr uint8_t kVersionNumber = 0x01;
    static constexpr cstd::array<uint8_t, 4> kSignature = { 'S', 'N', 'O', 'D' };
};

