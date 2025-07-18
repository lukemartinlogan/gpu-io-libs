#pragma once
#include <cstdint>
#include <vector>

#include "types.h"
#include "../serialization/serialization.h"

// TODO: replace this with a variant
struct ObjectHeaderMessage {
    uint16_t type;
    uint8_t flags;
    // FIXME: this should not be resized!
    std::vector<byte_t> message;

    uint16_t Size() const {
        return message.size();
    }

    uint16_t InternalSize() const {
        return sizeof(byte_t) * 8 + Size();
    }

    void Serialize(Serializer& s) const;

    static ObjectHeaderMessage Deserialize(Deserializer& de);
};

struct ObjectHeader {
    // total number of messages listed
    // includes continuation messages
    uint16_t message_count{};
    // number of hard links to this object in the current file
    uint32_t object_ref_count{};
    // number of bytes of header message data for this header
    // does not include size of object header continuation blocks
    uint32_t object_header_size{};
    // messages
    std::vector<ObjectHeaderMessage> messages{};

    void Serialize(Serializer& s) const;

    static ObjectHeader Deserialize(Deserializer& de);
private:
    static constexpr uint8_t kVersionNumber = 0x01;
};
