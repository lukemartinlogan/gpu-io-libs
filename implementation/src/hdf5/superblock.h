#pragma once
#include <cstdint>
#include <array>

#include "symbol_table.h"
#include "types.h"
#include "../serialization/serialization.h"

inline constexpr std::array<uint8_t, 8> kSuperblockSignature = { 0x89, 'H', 'D', 'F', '\r', '\n', 0x1a, '\n' };

struct FileConsistencyFlags {
    [[nodiscard]] bool WriteAccess() const {
        return Get(0);
    }

    [[nodiscard]] bool SWMR() const {
        return Get(2);
    }
private:
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

    void Serialize(Serializer& s) const;

    static SuperblockV0 Deserialize(Deserializer& de);
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

    void Serialize(Serializer& s) const;

    static SuperblockV2 Deserialize(Deserializer& de);

private:
    [[nodiscard]] uint32_t Checksum() const;

    static constexpr uint8_t kVersionNumber = 0x02;
};
