#include "symbol_table.h"

#include <stdexcept>

void SymbolTableEntry::Serialize(Serializer& s) const {
    s.Write(link_name_offset);
    s.Write(object_header_addr);
    s.Write<uint32_t>(static_cast<uint32_t>(cache_ty));
    // 4 bytes to align scratch pad on 16 byte boundary
    s.Write<uint32_t>(0);
    s.Write(scratch_pad_space);
}

hdf5::expected<SymbolTableEntry> SymbolTableEntry::Deserialize(Deserializer& de) {
    SymbolTableEntry ent{};

    ent.link_name_offset = de.Read<offset_t>();
    ent.object_header_addr = de.Read<offset_t>();
    ent.cache_ty = static_cast<SymbolTableEntryCacheType>(de.Read<uint32_t>());
    // 4 bytes to align scratch pad on 16 byte boundary
    de.Skip<uint32_t>();
    ent.scratch_pad_space = de.Read<cstd::array<byte_t, 16>>();

    constexpr uint8_t kCacheTyAllowedValues = 3;
    if (static_cast<uint8_t>(ent.cache_ty) >= kCacheTyAllowedValues) {
        return hdf5::error(hdf5::HDF5ErrorCode::InvalidDataValue, "Symbol Table Entry had invalid cache type");
    }

    if (ent.cache_ty == SymbolTableEntryCacheType::kSymbolicLink && ent.object_header_addr != kUndefinedOffset) {
        return hdf5::error(hdf5::HDF5ErrorCode::InvalidDataValue, "If symbol table entry cache type is symbolic link, object header addr should be undefined");
    }

    return ent;
}

hdf5::expected<cstd::optional<offset_t>> SymbolTableNode::FindEntry(hdf5::string_view name, const LocalHeap& heap, Deserializer& de) const {
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

void SymbolTableNode::Serialize(Serializer& s) const {
    s.Write(kSignature);
    s.Write(kVersionNumber);
    s.Write<uint8_t>(0);

    s.Write<uint16_t>(entries.size());

    // TODO: does this need to write the extra unused entries?
    for (const SymbolTableEntry& entry : entries) {
        s.WriteComplex(entry);
    }
}

hdf5::expected<SymbolTableNode> SymbolTableNode::Deserialize(Deserializer& de) {
    if (de.Read<cstd::array<uint8_t, 4>>() != kSignature) {
        return hdf5::error(hdf5::HDF5ErrorCode::InvalidSignature, "symbol table node signature was invalid");
    }

    if (de.Read<uint8_t>() != kVersionNumber) {
        return hdf5::error(hdf5::HDF5ErrorCode::InvalidVersion, "symbol table node version was invalid");
    }

    // reserved (zero)
    de.Skip<uint8_t>();

    // actual data
    auto num_symbols = de.Read<uint16_t>();

    SymbolTableNode node{};

    node.entries.reserve(num_symbols);

    for (uint16_t i = 0; i < num_symbols; ++i) {
        auto entry_result = de.ReadComplex<SymbolTableEntry>();
        if (!entry_result) return cstd::unexpected(entry_result.error());
        node.entries.push_back(*entry_result);
    }

    return node;
}
