#pragma once

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

SymbolTableEntry SymbolTableEntry::Deserialize(Deserializer& de) {
    SymbolTableEntry ent{};

    ent.link_name_offset = de.Read<offset_t>();
    ent.object_header_addr = de.Read<offset_t>();
    ent.cache_ty = static_cast<SymbolTableEntryCacheType>(de.Read<uint32_t>());
    // 4 bytes to align scratch pad on 16 byte boundary
    de.Skip<uint32_t>();
    ent.scratch_pad_space = de.Read<std::array<byte_t, 16>>();

    constexpr uint8_t kCacheTyAllowedValues = 3;
    if (static_cast<uint8_t>(ent.cache_ty) >= kCacheTyAllowedValues) {
        throw std::runtime_error("Symbol Table Entry had invalid cache type");
    }

    if (ent.cache_ty == SymbolTableEntryCacheType::kSymbolicLink && ent.object_header_addr != kUndefinedOffset) {
        throw std::runtime_error("If symbol table entry cache type is symbolic link, object header addr should be undefined");
    }

    return ent;
}
