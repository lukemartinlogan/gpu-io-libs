#include <vector>
#include <optional>
#include <stdexcept>

#include "../serialization/buffer.h"
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

std::string LocalHeap::ReadString(offset_t offset) const {
    if (offset >= data_segment.size()) {
        throw std::runtime_error("LocalHeap: offset out of bounds");
    }

    auto rem_size = data_segment.size() - offset;

    BufferDeserializer buf_de(std::span(data_segment.data() + offset, rem_size));

    return ReadNullTerminatedString(buf_de, rem_size);
}

std::span<const byte_t> LocalHeap::ReadData(len_t offset, len_t size) const {
    if (offset + size > data_segment.size()) {
        throw std::runtime_error("LocalHeap: read beyond heap bounds");
    }
    return std::span(data_segment.data() + offset, size);
}

void LocalHeap::Serialize(Serializer& s) const {
    s.Write(kSignature);
    s.Write(kVersionNumber);

    // reserved (zero)
    s.Write<uint8_t>(0);

    s.Write<len_t>(data_segment.size());
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

    auto data_segment_size = de.Read<len_t>();

    LocalHeap heap{};
    heap.free_list_head_offset = de.Read<len_t>();
    heap.data_segment_address = de.Read<offset_t>();

    heap.data_segment.resize(data_segment_size);

    // read local heap
    auto current_pos = de.GetPosition();

    de.SetPosition(heap.data_segment_address);
    de.ReadBuffer(heap.data_segment);

    de.SetPosition(current_pos);

    return heap;
}