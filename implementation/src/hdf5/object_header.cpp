#include "object_header.h"

#include <numeric>
#include <stdexcept>

void ObjectHeaderMessage::Serialize(Serializer& s) const {
    s.Write(type);
    s.Write(size);

    // FIXME: Serializer::WriteZero<size_t>
    s.Write<uint8_t>(0);
    s.Write<uint8_t>(0);
    s.Write<uint8_t>(0);

    s.Write(flags);

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

void ObjectHeader::Serialize(Serializer& s) const {
    s.Write(kVersionNumber);
    s.Write<uint8_t>(0);
    s.Write(message_count);
    s.Write(object_ref_count);
    s.Write(object_header_size);
    // reserved (zero)
    s.Write<uint64_t>(0);

    for (const ObjectHeaderMessage& msg: messages) {
        s.WriteComplex(msg);
    }
}

ObjectHeader ObjectHeader::Deserialize(Deserializer& de) {
    if (de.Read<uint8_t>() != kVersionNumber) {
        throw std::runtime_error("Version number was invalid");
    }
    // reserved (zero)
    de.Skip<uint8_t>();

    ObjectHeader hd{};

    hd.message_count = de.Read<uint16_t>();
    hd.object_ref_count = de.Read<uint64_t>();
    hd.object_header_size = de.Read<uint64_t>();
    // reserved (zero)
    de.Skip<uint64_t>();

    for (uint16_t m = 0; m < hd.message_count; ++m) {
        hd.messages.push_back(de.ReadComplex<ObjectHeaderMessage>());
    }

    uint64_t total_bytes = std::reduce(
        hd.messages.begin(),
        hd.messages.end(),
        static_cast<uint64_t>(0),
        [](uint64_t acc, const ObjectHeaderMessage& msg) {
            return acc + msg.InternalSize();
        }
    );

    if (total_bytes != hd.object_header_size) {
        throw std::runtime_error("Failed to read correct number of header bytes");
    }

    return hd;
}
