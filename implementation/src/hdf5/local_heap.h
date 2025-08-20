#pragma once

#include <string>
#include <vector>
#include <cstdint>

#include "types.h"
#include "../serialization/serialization.h"

struct LocalHeap {
    len_t free_list_head_offset{};
    [[nodiscard]] std::string ReadString(offset_t offset, Deserializer& de) const;

    void Serialize(Serializer& s) const;

    static LocalHeap Deserialize(Deserializer& de);

private:
    offset_t data_segment_address{};
    len_t data_segment_size{};

    static constexpr std::array<uint8_t, 4> kSignature = { 'H', 'E', 'A', 'P' };
    static constexpr uint8_t kVersionNumber = 0x00;
};