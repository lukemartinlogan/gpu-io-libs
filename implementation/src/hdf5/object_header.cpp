#include "object_header.h"

#include <stdexcept>

void ObjectHeaderMessage::Serialize(Serializer& s) const {
    s.Write(type);
    s.Write(size);
    s.Write(flags);

    // FIXME: Serializer::WriteZero<size_t>
    s.Write<uint8_t>(0);
    s.Write<uint8_t>(0);
    s.Write<uint8_t>(0);

    if (size != message.size()) {
        throw std::runtime_error("Mismatch in header message size");
    }

    s.WriteBuffer(message);
}

ObjectHeaderMessage ObjectHeaderMessage::Deserialize(Deserializer& de) {
    ObjectHeaderMessage msg{};

    msg.type = de.Read<uint16_t>();
    msg.size = de.Read<uint16_t>();
    msg.flags = de.Read<uint8_t>();
    de.Skip<3>(); // reserved (0)

    std::vector<byte_t> message(msg.size);

    de.ReadBuffer(msg.message);

    return msg;
}
