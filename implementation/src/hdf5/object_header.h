#pragma once
#include <cstdint>
#include <vector>

#include "types.h"
#include "../serialization/serialization.h"

// TODO: replace this with a variant
struct ObjectHeaderMessage {
    uint16_t type;
    uint16_t size;
    uint8_t flags;
    std::vector<byte_t> message;

    void Serialize(Serializer& s) const;

    static ObjectHeaderMessage Deserialize(Deserializer& de);
};

struct ObjectHeader {
    // total number of messages listed
    // includes continuation messages
    uint16_t message_count{};
    // number of hard links to this object in the current file
    uint64_t object_ref_count{};
    // number of bytes of header message data for this header
    // does not include size of object header continuation blocks
    uint64_t object_header_size{};
    // messages
    std::vector<ObjectHeaderMessage> messages{};

    void Serialize(Serializer& s) const;

    static ObjectHeader Deserialize(Deserializer& de);
private:
    static constexpr uint8_t kVersionNumber = 0x01;
};
