#pragma once

#include <string>
#include <cstdint>

#include "types.h"
#include "../serialization/serialization.h"

struct FileLink;

struct LocalHeap {
    len_t free_list_head_offset{};

    [[nodiscard]] std::string ReadString(offset_t offset, Deserializer& de) const;

    void Serialize(Serializer& s) const;

    static LocalHeap Deserialize(Deserializer& de);

private:
    struct FreeListBlock {
        len_t next_free_list_offset;
        len_t size;
    };

    struct SuitableFreeSpace {
        std::optional<offset_t> prev_block_offset;
        offset_t this_offset;
        FreeListBlock block;
    };

    std::optional<SuitableFreeSpace> FindFreeSpace(len_t required_size, Deserializer& de) const;

    void ReserveAdditional(FileLink& file, size_t additional_bytes);

private:
    offset_t data_segment_address{};
    len_t data_segment_size{};

    static constexpr offset_t kLastFreeBlock = 1;

    static constexpr std::array<uint8_t, 4> kSignature = { 'H', 'E', 'A', 'P' };
    static constexpr uint8_t kVersionNumber = 0x00;
};