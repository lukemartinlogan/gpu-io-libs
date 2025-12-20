#include <stdexcept>
#include <array>

#include "superblock.h"
#include "../util/lookup3.h"


__device__
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
        cstd::as_bytes(cstd::span(&data, 1))
    );
}
