#include <stdexcept>
#include <array>

#include "superblock.h"
#include "../util/lookup3.h"

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

SuperblockV2 SuperblockV2::Deserialize(Deserializer& de) {
    if (de.Read<std::array<uint8_t, 8>>() != kSuperblockSignature) {
        throw std::runtime_error("Superblock signature was invalid");
    }

    if (de.Read<uint8_t>() != kVersionNumber) {
        throw std::runtime_error("Superblock version number was invalid");
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
        throw std::runtime_error("differently sized offset/len not implemented");
    }

    if (de.Read<uint32_t>() != sb.Checksum()) {
        throw std::runtime_error("Superblock checksum didn't match");
    }

    return sb;
}

uint32_t SuperblockV2::Checksum() const { // NOLINT
    struct ChecksumData {
        std::array<uint8_t, 8> signature = kSuperblockSignature;
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
