#include <vector>
#include <optional>
#include <stdexcept>

#include "local_heap.h"


std::string ReadNullTerminatedString(Deserializer& de, std::optional<size_t> max_size) {
    std::vector<byte_t> buf;

    while (!max_size || buf.size() < *max_size) {
        auto c = de.Read<byte_t>();

        if (c == static_cast<byte_t>('\0')) {
            break;
        }

        buf.push_back(c);
    }

    return { reinterpret_cast<const char*>(buf.data()), buf.size() };
}

std::string LocalHeap::ReadString(offset_t offset, Deserializer& de) const {
    if (offset >= data_segment_size) {
        throw std::runtime_error("LocalHeap: offset out of bounds");
    }

    auto rem_size = data_segment_size - offset;

    de.SetPosition(data_segment_address + offset);

    return ReadNullTerminatedString(de, rem_size);
}

std::optional<offset_t> LocalHeap::FindFreeSpace(len_t required_size, Deserializer& de) const {
    static_assert(sizeof(FreeListBlock) == 2 * sizeof(len_t), "mismatch between spec");

    if (free_list_head_offset == kUndefinedOffset) {
        return std::nullopt;
    }

    offset_t current_offset = free_list_head_offset;

    while (current_offset != kLastFreeBlock) {
        if (current_offset + sizeof(FreeListBlock) > data_segment_size) {
            throw std::runtime_error("LocalHeap: free list offset out of bounds");
        }

        de.SetPosition(data_segment_address + current_offset);
        auto block = de.ReadRaw<FreeListBlock>();

        if (block.size >= required_size) {
            return current_offset;
        }
    }

    return std::nullopt;
}

void LocalHeap::Serialize(Serializer& s) const {
    s.Write(kSignature);
    s.Write(kVersionNumber);

    // reserved (zero)
    s.Write<std::array<byte_t, 3>>({});

    s.Write<len_t>(data_segment_size);
    s.Write(free_list_head_offset);
    s.Write(data_segment_address);
}

LocalHeap LocalHeap::Deserialize(Deserializer& de) {
    if (de.Read<std::array<uint8_t, 4>>() != kSignature) {
        throw std::runtime_error("Superblock signature was invalid");
    }

    if (de.Read<uint8_t>() != kVersionNumber) {
        throw std::runtime_error("Superblock version number was invalid");
    }
    // reserved (zero)
    de.Skip<uint8_t>();
    de.Skip<uint8_t>();
    de.Skip<uint8_t>();


    LocalHeap heap{};
    heap.data_segment_size = de.Read<len_t>();
    heap.free_list_head_offset = de.Read<len_t>();
    heap.data_segment_address = de.Read<offset_t>();

    return heap;
}