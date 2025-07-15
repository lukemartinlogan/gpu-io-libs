#pragma once
#include <cstdint>
#include <array>

#include "types.h"
#include "../serialization/serialization.h"

inline const std::array<uint8_t, 8> kSuperblockSignature = { 0x89, 0x48, 0x44, 0x46, 0x0d, 0x0a, 0x1a, 0x0a };

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
