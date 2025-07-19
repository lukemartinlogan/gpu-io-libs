#include "datatype.h"

#include <stdexcept>

void DatatypeMessage::Serialize(Serializer& s) const {
    uint8_t high = static_cast<uint8_t>(version);
    uint8_t low = static_cast<uint8_t>(class_v);

    uint8_t class_and_version = (high << 4) | (low & 0x0f);

    s.Write(class_and_version);
    s.Write(bit_field);
    s.Write(size_bytes);

    // TODO: write properties
    // s.WriteBuffer(properties)
}

DatatypeMessage DatatypeMessage::Deserialize(Deserializer& de) {
    uint8_t class_and_version = de.Read<uint8_t>();

    // high
    uint8_t version = ((class_and_version >> 4) & 0x0f);
    // low
    uint8_t class_v = class_and_version & 0x0f;

    if (1 > version || version > 5) {
        throw std::runtime_error("invalid datatype version");
    }

    if (class_v >= 11) {
        throw std::runtime_error("invalid datatype class");
    }

    DatatypeMessage msg{};

    msg.version = static_cast<Version>(version);
    msg.class_v = static_cast<Class>(class_v);

    msg.bit_field = de.Read<std::array<uint8_t, 3>>();
    msg.size_bytes = de.Read<uint32_t>();

    // TODO: read properties
    // msg.properties = /* ... */

    return msg;
}
