#include "datatype.h"

#include <stdexcept>

void FixedPoint::Serialize(Serializer& s) const {
    // first four bits are used
    s.Write(static_cast<uint8_t>(bitset_.to_ulong()) & 0x0f);
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

void FloatingPoint::Serialize(Serializer& s) const {
    s.Write(static_cast<uint8_t>(bitset_.to_ulong()) & 0x7f);
    s.Write(sign_location);
    // reserved (zero)
    s.Write<uint8_t>(0);

    s.Write(size);

    s.Write(bit_offset);
    s.Write(bit_precision);
    s.Write(exponent_location);
    s.Write(exponent_size);
    s.Write(mantissa_location);
    s.Write(mantissa_size);
    s.Write(exponent_bias);
}

FloatingPoint FloatingPoint::Deserialize(Deserializer& de) {
    FloatingPoint fp{};

    fp.bitset_ = de.Read<uint8_t>() & 0x7f;
    fp.sign_location = de.Read<uint8_t>();
    // reserved (zero)
    de.Skip<1>();

    fp.size = de.Read<uint32_t>();

    fp.bit_offset = de.Read<uint16_t>();
    fp.bit_precision = de.Read<uint16_t>();
    fp.exponent_location = de.Read<uint8_t>();
    fp.exponent_size = de.Read<uint8_t>();
    fp.mantissa_location = de.Read<uint8_t>();
    fp.mantissa_size = de.Read<uint8_t>();
    fp.exponent_bias = de.Read<uint32_t>();

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
        case Class::kFloatingPoint: {
            msg.data = de.ReadComplex<FloatingPoint>();
            break;
        }
        case Class::kCompound: {
            msg.data = de.ReadComplex<CompoundDatatype>();
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


void WritePaddedString(std::string_view name, Serializer&s) {
    size_t name_size = name.size();

    // write string
    s.WriteBuffer(std::span(
        reinterpret_cast<const byte_t*>(name.data()),
        name_size
    ));

    // pad to 8 bytes
    size_t padding = (name_size / 8 + 1) * 8 - name_size;
    static constexpr std::array<byte_t, 8> nul_bytes{};

    s.WriteBuffer(std::span(nul_bytes.data(), padding));
}

void CompoundMember::Serialize(Serializer& s) const {
    // includes null terminator
    WritePaddedString(name, s);

    s.Write(byte_offset);

    auto dimensionality = static_cast<uint8_t>(dimension_sizes.size());
    s.Write(dimensionality);

    // reserved (zero)
    s.Write<uint8_t>(0);
    s.Write<uint8_t>(0);
    s.Write<uint8_t>(0);
    // dimension permutation (unused)
    s.Write<uint32_t>(0);
    // reserved (zero)
    s.Write<uint32_t>(0);

    for (uint8_t i = 0; i < 4; ++i) {
        if (i < dimensionality) {
            s.Write<uint32_t>(dimension_sizes.at(i));
        } else {
            s.Write<uint32_t>(0);
        }
    }

    s.WriteComplex(message);
}

std::string ReadPaddedString(Deserializer& de) {
    std::string name;
    size_t read = 0;

    for (;;) {
        // 8 byte blocks
        std::array<byte_t, 8> buf{};

        if (!de.ReadBuffer(buf)) {
            throw std::runtime_error("failed to read string block");
        }

        read += buf.size();

        auto nul_pos = std::find(buf.begin(), buf.end(), '\0');

        name.append(
            reinterpret_cast<const char*>(buf.data()),
            std::distance(buf.begin(), nul_pos)
        );

        if (nul_pos != buf.end()) {
            break;
        }
    }

    return name;
}

CompoundMember CompoundMember::Deserialize(Deserializer& de) {
    CompoundMember mem{};

    mem.name = ReadPaddedString(de);
    mem.byte_offset = de.Read<uint32_t>();

    uint8_t dimensionality = de.Read<uint8_t>();
    // reserved (zero)
    de.Skip<3>();
    // dimension permutation (unused)
    de.Skip<uint32_t>();
    // reserved (zero)
    de.Skip<uint32_t>();

    for (uint8_t i = 0; i < 4; ++i) {
        auto size = de.Read<uint32_t>();

        if (size < dimensionality) {
            mem.dimension_sizes.push_back(size);
        }
    }

    mem.message = de.ReadComplex<DatatypeMessage>();

    return mem;
}
