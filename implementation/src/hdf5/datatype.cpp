#include "datatype.h"

#include <stdexcept>

void FixedPoint::Serialize(Serializer& s) const {
    // first four bits are used
    s.Write(static_cast<uint8_t>(bitset_.to_ulong()));
    // reserved (zero)
    s.Write<uint16_t>(0);

    s.Write(size);
    s.Write(bit_offset);
    s.Write(bit_precision);
}

FixedPoint FixedPoint::Deserialize(Deserializer& de) {
    FixedPoint fp{};

    // first four bits are used
    fp.bitset_ = de.Read<uint8_t>() & 0x0f;
    // reserved (zero)
    de.Skip<2>();

    fp.size = de.Read<uint32_t>();
    fp.bit_offset = de.Read<uint16_t>();
    fp.bit_precision = de.Read<uint16_t>();

    return fp;
}

void DatatypeMessage::Serialize(Serializer& s) const {
    uint8_t high = static_cast<uint8_t>(version);
    uint8_t low = static_cast<uint8_t>(class_v);

    uint8_t class_and_version = (high << 4) | (low & 0x0f);

    s.Write(class_and_version);

    std::visit([&s](const auto& data) { return data.Serialize(s); }, data);
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

    switch (msg.class_v) {
        case Class::kFixedPoint: {
            msg.data = de.ReadComplex<FixedPoint>();
            break;
        }
        default: {
            throw std::logic_error("not implemented");
        }
    }

    // TODO: read properties
    // msg.properties = /* ... */

    return msg;
}
