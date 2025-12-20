#pragma once
#include <cstdint>
#include <array>

#include "symbol_table.h"
#include "types.h"
#include "../serialization/serialization.h"

inline constexpr cstd::array<uint8_t, 8> kSuperblockSignature = { 0x89, 'H', 'D', 'F', '\r', '\n', 0x1a, '\n' };

struct FileConsistencyFlags {
    __device__
    [[nodiscard]] bool WriteAccess() const {
        return Get(0);
    }

    __device__
    [[nodiscard]] bool SWMR() const {
        return Get(2);
    }
private:
    __device__
    [[nodiscard]] bool Get(uint8_t bit) const {
        return bitmap & 1u << bit;
    }

    uint8_t bitmap = 0;
};

// TODO: Superblock checked for at 0, 512, 1024, 2048, ..
struct SuperblockV0 {
    // number of bytes used for storing addresses,
    // addresses are relative to base address,
    // superblock address is usually base
    uint8_t size_of_offsets = 8;
    // num of bytes used to store size of object
    uint8_t size_of_lengths = 8;
    // Each leaf node of a group B-tree will have at least this many entries but not more than twice this many
    // if a group has a single leaf node then it may have fewer entries
    // this value must be greater than zero
    uint16_t group_leaf_node_k{};
    // each internal node of a group B-tree will have at least this many entries but not more than twice this many
    // if the group has only one internal node then it might have fewer entries
    // this value must be greater than zero
    uint16_t group_internal_node_k{};
    // file address of first byte of HDF5 data
    // generally 0x00, nonzero for 'driver cases'?
    // must be within the superblock
    offset_t base_addr = 0;
    // address of first bytes past all the HDF5 data,
    // can be used to check for accidental truncation
    offset_t eof_addr = kUndefinedOffset;
    // relative file addr of file driver information block
    offset_t driver_info_block_addr = kUndefinedOffset;
    // symbol table entry of the root group
    // entry point into group graph of file
    SymbolTableEntry root_group_symbol_table_entry_addr;

    template<serde::Serializer S>
    __device__
    void Serialize(S& s) const {
        serde::Write(s, cstd::array<uint8_t, 8>{ 0x89, 'H', 'D', 'F', '\r', '\n', 0x1a, '\n' });
        serde::Write(s, kVersionNumber);
        // file free space version num
        serde::Write(s, static_cast<uint8_t>(0));
        // root group symbol table entry version num
        serde::Write(s, static_cast<uint8_t>(0));
        // reserved
        serde::Write(s, static_cast<uint8_t>(0));
        // shared header message format version num
        serde::Write(s, static_cast<uint8_t>(0));
        serde::Write(s, size_of_offsets);
        serde::Write(s, size_of_lengths);
        // reserved
        serde::Write(s, static_cast<uint8_t>(0));
        serde::Write(s, group_leaf_node_k);
        serde::Write(s, group_internal_node_k);
        // file consistency flags, unused
        serde::Write(s, static_cast<uint32_t>(0));
        serde::Write(s, base_addr);
        // file free space info addr, always undefined
        serde::Write(s, kUndefinedOffset);
        serde::Write(s, eof_addr);
        serde::Write(s, driver_info_block_addr);
        serde::Write(s, root_group_symbol_table_entry_addr);
    }

    template<serde::Deserializer D>
    __device__
    static hdf5::expected<SuperblockV0> Deserialize(D& de) {
        if (serde::Read<cstd::array<uint8_t, 8>>(de) != cstd::array<uint8_t, 8>{ 0x89, 'H', 'D', 'F', '\r', '\n', 0x1a, '\n' }) {
            return hdf5::error(hdf5::HDF5ErrorCode::InvalidSignature, "Superblock signature was invalid");
        }

        if (serde::Read<uint8_t>(de) != kVersionNumber) {
            return hdf5::error(hdf5::HDF5ErrorCode::InvalidVersion, "Superblock version number was invalid");
        }

        SuperblockV0 sb{};

        // file free space version num
        serde::Skip<uint8_t>(de);
        // root group symbol table entry version num
        serde::Skip<uint8_t>(de);
        // reserved
        serde::Skip<uint8_t>(de);
        // shared header message format version num
        serde::Skip<uint8_t>(de);
        sb.size_of_offsets = serde::Read<uint8_t>(de);
        sb.size_of_lengths = serde::Read<uint8_t>(de);
        // reserved
        serde::Skip<uint8_t>(de);
        sb.group_leaf_node_k = serde::Read<uint16_t>(de);
        sb.group_internal_node_k = serde::Read<uint16_t>(de);
        // file consistency flags, unused
        serde::Skip<uint32_t>(de);
        sb.base_addr = serde::Read<offset_t>(de);
        // file free space info addr, always undefined
        serde::Skip<offset_t>(de);
        sb.eof_addr = serde::Read<offset_t>(de);
        sb.driver_info_block_addr = serde::Read<offset_t>(de);

        auto entry_result = serde::Read<SymbolTableEntry>(de);
        if (!entry_result) {
            return cstd::unexpected(entry_result.error());
        }

        sb.root_group_symbol_table_entry_addr = *entry_result;

        return sb;
    }

private:
    static constexpr uint8_t kVersionNumber = 0x00;
};

struct SuperblockV2 {
    // number of bytes used for storing addresses,
    // addresses are relative to base address,
    // superblock address is usually base
    uint8_t size_of_offsets = 8;
    // num of bytes used to store size of object
    uint8_t size_of_lengths = 8;
    // file consistency flags
    FileConsistencyFlags file_consistency_flags{};
    // file address of first byte of HDF5 data
    // generally 0x00, nonzero for 'driver cases'?
    offset_t base_addr = 0;
    // address of extra superblock metadata
    // if undefined -> 0xff...fff
    offset_t superblock_ext = kUndefinedOffset;
    // address of first bytes past all the HDF5 data,
    // can be used to check for accidental truncation
    offset_t eof_addr = kUndefinedOffset;
    // address of data objects, entry point into group graph
    offset_t root_group_header_addr = kUndefinedOffset;

    template<serde::Serializer S>
    __device__
    void Serialize(S& s) const {
        const uint32_t checksum = Checksum();

        serde::Write(s, cstd::array<uint8_t, 8>{ 0x89, 'H', 'D', 'F', '\r', '\n', 0x1a, '\n' });
        serde::Write(s, kVersionNumber);
        serde::Write(s, size_of_offsets);
        serde::Write(s, size_of_lengths);
        serde::Write(s, file_consistency_flags);
        serde::Write(s, base_addr);
        serde::Write(s, superblock_ext);
        serde::Write(s, eof_addr);
        serde::Write(s, root_group_header_addr);
        serde::Write(s, checksum);
    }

    template<serde::Deserializer D>
    __device__
    static hdf5::expected<SuperblockV2> Deserialize(D& de) {
        if (serde::Read<cstd::array<uint8_t, 8>>(de) != cstd::array<uint8_t, 8>{ 0x89, 'H', 'D', 'F', '\r', '\n', 0x1a, '\n' }) {
            return hdf5::error(hdf5::HDF5ErrorCode::InvalidSignature, "Superblock signature was invalid");
        }

        if (serde::Read<uint8_t>(de) != kVersionNumber) {
            return hdf5::error(hdf5::HDF5ErrorCode::InvalidVersion, "Superblock version number was invalid");
        }

        const SuperblockV2 sb {
            .size_of_offsets = serde::Read<uint8_t>(de),
            .size_of_lengths = serde::Read<uint8_t>(de),
            .file_consistency_flags = serde::Read<FileConsistencyFlags>(de),
            .base_addr = serde::Read<offset_t>(de),
            .superblock_ext = serde::Read<offset_t>(de),
            .eof_addr = serde::Read<offset_t>(de),
            .root_group_header_addr = serde::Read<offset_t>(de),
        };

        if (sb.size_of_offsets != 8 || sb.size_of_lengths != 8) {
            return hdf5::error(hdf5::HDF5ErrorCode::NotImplemented, "differently sized offset/len not implemented");
        }

        if (serde::Read<uint32_t>(de) != sb.Checksum()) {
            return hdf5::error(hdf5::HDF5ErrorCode::InvalidChecksum, "Superblock checksum didn't match");
        }

        return sb;
    }

private:
    [[nodiscard]] uint32_t Checksum() const;

    static constexpr uint8_t kVersionNumber = 0x02;
};
