#pragma once

#include <string>
#include <vector>
#include <cstdint>

#include "types.h"
#include "../serialization/serialization.h"

struct LocalHeap {
    len_t free_list_head_offset{};

    // FIXME: don't store this?
    std::vector<byte_t> data_segment;

    std::string ReadString(offset_t offset) const;

    // Read raw data at the given offset
    std::span<const byte_t> ReadData(len_t offset, len_t size) const;

    void Serialize(Serializer& s) const;

    static LocalHeap Deserialize(Deserializer& de);

private:
    offset_t data_segment_address{};

    static constexpr std::array<uint8_t, 4> kSignature = { 'H', 'E', 'A', 'P' };
    static constexpr uint8_t kVersionNumber = 0x00;
};