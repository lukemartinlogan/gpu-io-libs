#include <numeric>
#include <stdexcept>

#include "object_header.h"

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

void ObjectHeaderMessage::Serialize(Serializer& s) const {
    s.Write(type);
    s.Write(InternalSize());

    // FIXME: Serializer::WriteZero<size_t>
    s.Write<uint8_t>(0);
    s.Write<uint8_t>(0);
    s.Write<uint8_t>(0);

    s.Write(flags);

    switch (type) {
        case Type::kDatatype: {
            s.Write(std::get<DatatypeMessage>(message));
            break;
        }
        case Type::kObjectHeaderContinuation: {
            s.Write(std::get<ObjectHeaderContinuationMessage>(message));
            break;
        }
        case Type::kSymbolTable: {
            s.Write(std::get<SymbolTableMessage>(message));
            break;
        }
        default: {
            throw std::logic_error("not implemented");
        }
    }
}

ObjectHeaderMessage ObjectHeaderMessage::Deserialize(Deserializer& de) {
    ObjectHeaderMessage msg{};

    uint16_t type = de.Read<uint16_t>();

    constexpr uint16_t kMessageTypeCt = 0x18;
    if (type >= kMessageTypeCt) {
        throw std::runtime_error("Not a valid message type");
    }

    msg.type = static_cast<Type>(type);

    uint16_t size = de.Read<uint16_t>();
    msg.flags = de.Read<uint8_t>();
    de.Skip<3>(); // reserved (0)

    switch (msg.type) {
        case Type::kDatatype: {
            msg.message = de.ReadComplex<DatatypeMessage>();
        }
        case Type::kObjectHeaderContinuation: {
            msg.message = de.ReadComplex<ObjectHeaderContinuationMessage>();
            break;
        }
        case Type::kSymbolTable: {
            msg.message = de.ReadComplex<SymbolTableMessage>();
            break;
        }
        default: {
            throw std::logic_error("not implemented");
        }
    }

    if (std::visit([](const auto& m) { return m.InternalSize(); }, msg.message) != size) {
        throw std::runtime_error("message size was incorrect");
    }

    return msg;
}

void ObjectHeader::Serialize(Serializer& s) const {
    s.Write(kVersionNumber);
    s.Write<uint8_t>(0);
    s.Write(message_count);
    s.Write(object_ref_count);
    s.Write(object_header_size);
    // reserved (zero)
    s.Write<uint32_t>(0);

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
    hd.object_ref_count = de.Read<uint32_t>();
    hd.object_header_size = de.Read<uint32_t>();
    // reserved (zero)
    de.Skip<uint32_t>();

    for (uint16_t m = 0; m < hd.message_count; ++m) {
        hd.messages.push_back(de.ReadComplex<ObjectHeaderMessage>());
    }

    uint64_t total_bytes = std::accumulate(
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
