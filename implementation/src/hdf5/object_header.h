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
    uint16_t header_message_count;
    uint64_t object_ref_count;
    std::vector<ObjectHeaderMessage> messages;

    void Serialize(Serializer& s) const;

    static ObjectHeader Deserialize(Deserializer& de);
private:
    static constexpr uint8_t kVersionNumber = 0x01;
};
