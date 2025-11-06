#include <stdexcept>
#include <array>

#include "superblock.h"
#include "../util/lookup3.h"

void SuperblockV0::Serialize(Serializer& s) const {
    s.Write(kSuperblockSignature);
    s.Write(kVersionNumber);
    // file free space version num
    s.Write<uint8_t>(0);
    // root group symbol table entry version num
    s.Write<uint8_t>(0);
    // reserved
    s.Write<uint8_t>(0);
    // shared header message format version num
    s.Write<uint8_t>(0);
    s.Write(size_of_offsets);
    s.Write(size_of_lengths);
    // reserved
    s.Write(0);
    s.Write(group_leaf_node_k);
    s.Write(group_internal_node_k);
    // file consistency flags, unused
    s.Write<uint32_t>(0);
    s.Write(base_addr);
    // file free space info addr, always undefined
    s.Write(kUndefinedOffset);
    s.Write(eof_addr);
    s.Write(driver_info_block_addr);
    s.Write(root_group_symbol_table_entry_addr);
}

hdf5::expected<SuperblockV0> SuperblockV0::Deserialize(Deserializer& de) {
    if (de.Read<cstd::array<uint8_t, 8>>() != kSuperblockSignature) {
        return hdf5::error(hdf5::HDF5ErrorCode::InvalidSignature, "Superblock signature was invalid");
    }

    if (de.Read<uint8_t>() != kVersionNumber) {
        return hdf5::error(hdf5::HDF5ErrorCode::InvalidVersion, "Superblock version number was invalid");
    }

    SuperblockV0 sb{};

    // file free space version num
    de.Skip<uint8_t>();
    // root group symbol table entry version num
    de.Skip<uint8_t>();
    // reserved
    de.Skip<uint8_t>();
    // shared header message format version num
    de.Skip<uint8_t>();
    sb.size_of_offsets = de.Read<uint8_t>();
    sb.size_of_lengths = de.Read<uint8_t>();
    // reserved
    de.Skip<uint8_t>();
    sb.group_leaf_node_k = de.Read<uint16_t>();
    sb.group_internal_node_k = de.Read<uint16_t>();
    // file consistency flags, unused
    de.Skip<uint32_t>();
    sb.base_addr = de.Read<offset_t>();
    // file free space info addr, always undefined
    de.Skip<offset_t>();
    sb.eof_addr = de.Read<offset_t>();
    sb.driver_info_block_addr = de.Read<offset_t>();
    sb.root_group_symbol_table_entry_addr = de.Read<SymbolTableEntry>();

    return sb;
}


void SuperblockV2::Serialize(Serializer& s) const {
    const uint32_t checksum = Checksum();

    s.Write(kSuperblockSignature);
    s.Write(kVersionNumber);
    s.Write(size_of_offsets);
    s.Write(size_of_lengths);
    s.Write(file_consistency_flags);
    s.Write(base_addr);
    s.Write(superblock_ext);
    s.Write(eof_addr);
    s.Write(root_group_header_addr);
    s.Write(checksum);
}

hdf5::expected<SuperblockV2> SuperblockV2::Deserialize(Deserializer& de) {
    if (de.Read<cstd::array<uint8_t, 8>>() != kSuperblockSignature) {
        return hdf5::error(hdf5::HDF5ErrorCode::InvalidSignature, "Superblock signature was invalid");
    }

    if (de.Read<uint8_t>() != kVersionNumber) {
        return hdf5::error(hdf5::HDF5ErrorCode::InvalidVersion, "Superblock version number was invalid");
    }

    const SuperblockV2 sb {
        .size_of_offsets = de.Read<uint8_t>(),
        .size_of_lengths = de.Read<uint8_t>(),
        .file_consistency_flags = de.Read<FileConsistencyFlags>(),
        .base_addr = de.Read<offset_t>(),
        .superblock_ext = de.Read<offset_t>(),
        .eof_addr = de.Read<offset_t>(),
        .root_group_header_addr = de.Read<offset_t>(),
    };

    if (sb.size_of_offsets != 8 || sb.size_of_lengths != 8) {
        return hdf5::error(hdf5::HDF5ErrorCode::NotImplemented, "differently sized offset/len not implemented");
    }

    if (de.Read<uint32_t>() != sb.Checksum()) {
        return hdf5::error(hdf5::HDF5ErrorCode::InvalidChecksum, "Superblock checksum didn't match");
    }

    return sb;
}

uint32_t SuperblockV2::Checksum() const { // NOLINT
    struct ChecksumData {
        cstd::array<uint8_t, 8> signature = kSuperblockSignature;
        uint8_t version_number= kVersionNumber;
        SuperblockV2 sb;
    };

    ChecksumData data {
        .sb = *this
    };

    return lookup3::HashLittle(
        std::as_bytes(std::span(&data, 1))
    );
}
