#include "datatype.h"

#include <stdexcept>

void FixedPoint::Serialize(Serializer& s) const {
    // first four bits are used
    s.Write(static_cast<uint8_t>(bitset_.to_ulong() & 0x0f));
    // reserved (zero)
    s.Write<uint16_t>(0);

    s.Write(size);
    s.Write(bit_offset);
    s.Write(bit_precision);
}

hdf5::expected<FixedPoint> FixedPoint::Deserialize(Deserializer& de) {
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
    s.Write(static_cast<uint8_t>(bitset_.to_ulong() & 0x7f));
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

hdf5::expected<FloatingPoint> FloatingPoint::Deserialize(Deserializer& de) {
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

// TODO: have this method presented in a different way?
FloatingPoint::FloatingPoint(
    uint32_t size,
    uint8_t sign_location,
    uint16_t bit_offset,
    uint16_t bit_precision,
    uint8_t exponent_location,
    uint8_t exponent_size,
    uint8_t mantissa_location,
    uint8_t mantissa_size,
    uint32_t exponent_bias,

    ByteOrder byte_order,
    MantissaNormalization norm,
    bool low_padding,
    bool high_padding,
    bool internal_padding
) {
    this->size = size;
    this->sign_location = sign_location;
    this->bit_offset = bit_offset;
    this->bit_precision = bit_precision;
    this->exponent_size = exponent_size;
    this->mantissa_location = mantissa_location;
    this->mantissa_size = mantissa_size;
    this->exponent_bias = exponent_bias;
    this->exponent_location = exponent_location;

    this->bitset_.set(1, low_padding);
    this->bitset_.set(2, high_padding);
    this->bitset_.set(3, internal_padding);

    switch (byte_order) {
        case ByteOrder::kLittleEndian: {
            this->bitset_.set(0, false);
            this->bitset_.set(6, false);
            break;
        }
        case ByteOrder::kBigEndian: {
            this->bitset_.set(0, true);
            this->bitset_.set(6, false);
            break;
        }
        case ByteOrder::kVAXEndian: {
            this->bitset_.set(0, true);
            this->bitset_.set(6, true);
            break;
        }
    }

    switch (norm) {
        case MantissaNormalization::kNone: {
            this->bitset_.set(5, false);
            this->bitset_.set(4, false);
            break;
        }
        case MantissaNormalization::kMSBSet: {
            this->bitset_.set(5, false);
            this->bitset_.set(4, true);
            break;
        }
        case MantissaNormalization::kMSBImpliedSet: {
            this->bitset_.set(5, true);
            this->bitset_.set(4, false);
            break;
        }
    }
}

const FloatingPoint FloatingPoint::f32_t = FloatingPoint(
    4, 31, 0,
    32,23, 8,
    0, 23, 127,
    ByteOrder::kLittleEndian,
    MantissaNormalization::kMSBImpliedSet,
    false, false, false
);

const DatatypeMessage DatatypeMessage::f32_t = {
    .version = Version::kEarlyCompound,
    .class_v = Class::kFloatingPoint,
    .data = FloatingPoint::f32_t,
};

void DatatypeMessage::Serialize(Serializer& s) const {
    auto high = static_cast<uint8_t>(version);
    auto low = static_cast<uint8_t>(class_v);

    uint8_t class_and_version = (high << 4) | (low & 0x0f);

    s.Write(class_and_version);

    cstd::visit([&s](const auto& data) { return data.Serialize(s); }, data);
}

hdf5::expected<DatatypeMessage> DatatypeMessage::Deserialize(Deserializer& de) {
    auto class_and_version = de.Read<uint8_t>();

    // high
    uint8_t version = ((class_and_version >> 4) & 0x0f);
    // low
    uint8_t class_v = class_and_version & 0x0f;

    if (1 > version || version > 5) {
        return hdf5::error(hdf5::HDF5ErrorCode::InvalidVersion, "invalid datatype version");
    }

    if (class_v >= 11) {
        return hdf5::error(hdf5::HDF5ErrorCode::InvalidClass, "invalid datatype class");
    }

    DatatypeMessage msg{};

    msg.version = static_cast<Version>(version);
    msg.class_v = static_cast<Class>(class_v);

    switch (msg.class_v) {
        case Class::kFixedPoint: {
            auto result = de.ReadComplex<FixedPoint>();
            if (!result) return cstd::unexpected(result.error());
            msg.data = *result;
            break;
        }
        case Class::kFloatingPoint: {
            auto result = de.ReadComplex<FloatingPoint>();
            if (!result) return cstd::unexpected(result.error());
            msg.data = *result;
            break;
        }
        case Class::kCompound: {
            auto result = de.ReadComplex<CompoundDatatype>();
            if (!result) return cstd::unexpected(result.error());
            msg.data = *result;
            break;
        }
        case Class::kVariableLength: {
            auto result = de.ReadComplex<VariableLength>();
            if (!result) return cstd::unexpected(result.error());
            msg.data = *result;
            break;
        }
        default: {
            return hdf5::error(hdf5::HDF5ErrorCode::NotImplemented, "datatype message type not implemented");
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
    static constexpr cstd::array<byte_t, 8> nul_bytes{};

    s.WriteBuffer(std::span(nul_bytes.data(), padding));
}

VariableLength::VariableLength(const VariableLength& other)
    : type(other.type),
      padding(other.padding),
      charset(other.charset),
      size(other.size)
{
    if (other.parent_type) {
        parent_type = std::make_unique<DatatypeMessage>(*other.parent_type);
    } else {
        parent_type = nullptr;
    }
}

VariableLength& VariableLength::operator=(const VariableLength& other) {
    if (this == &other) {
        return *this;
    }

    type = other.type;
    padding = other.padding;
    charset = other.charset;
    size = other.size;

    if (other.parent_type) {
        parent_type = std::make_unique<DatatypeMessage>(*other.parent_type);
    } else {
        parent_type = nullptr;
    }

    return *this;
}

void VariableLength::Serialize(Serializer& s) const {
    uint8_t bitset_1 = (static_cast<uint8_t>(padding) << 4) | static_cast<uint8_t>(type);
    s.Write(bitset_1);

    // Write charset in lower 4 bits, upper 4 bits zero
    s.Write<uint8_t>(static_cast<uint8_t>(charset) & 0b1111);

    // Reserved byte (zero)
    s.Write<uint8_t>(0);

    // Write size
    s.Write(size);

    // If not a string, write the parent_type
    if (type != Type::kString) {
        s.WriteComplex(*parent_type);
    }
}

hdf5::expected<VariableLength> VariableLength::Deserialize(Deserializer& de) {
    auto bitset_1 = de.Read<uint8_t>();

    VariableLength vl{};

    // type
    auto type = bitset_1 & 0b1111;

    if (type >= 2) {
        return hdf5::error(hdf5::HDF5ErrorCode::InvalidDataValue, "type wasn't valid");
    }

    vl.type = static_cast<Type>(type);

    // padding
    auto padding_ty = (bitset_1 >> 4) & 0b1111;

    if (padding_ty >= 3) {
        return hdf5::error(hdf5::HDF5ErrorCode::InvalidDataValue, "padding type wasn't valid");
    }

    vl.padding = static_cast<PaddingType>(padding_ty);

    // charset
    auto charset = de.Read<uint8_t>() & 0b1111;

    if (charset >= 2) {
        return hdf5::error(hdf5::HDF5ErrorCode::InvalidDataValue, "charset wasn't valid");
    }

    vl.charset = static_cast<Charset>(charset);

    // reserved (zero)
    de.Skip<uint8_t>();

    vl.size = de.Read<uint32_t>();

    if (vl.type != Type::kString) {
        auto datatype_result = de.ReadComplex<DatatypeMessage>();
        if (!datatype_result) return cstd::unexpected(datatype_result.error());
        vl.parent_type = std::make_unique<DatatypeMessage>(*datatype_result);
    }

    return vl;
}

CompoundMember::CompoundMember(const CompoundMember& other)
        : name(other.name),
          byte_offset(other.byte_offset),
          dimension_sizes(other.dimension_sizes),
          message(std::make_unique<DatatypeMessage>(*other.message))
{ }

CompoundMember& CompoundMember::operator=(const CompoundMember& other) {
    if (this == &other) {
        return *this;
    }

    name = other.name;
    byte_offset = other.byte_offset;
    dimension_sizes = other.dimension_sizes;
    message = std::make_unique<DatatypeMessage>(*other.message);
    return *this;
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

    s.WriteComplex(*message);
}

hdf5::expected<std::string> ReadPaddedString(Deserializer& de) {
    std::string name;

    for (;;) {
        // 8 byte blocks
        cstd::array<byte_t, 8> buf{};

        if (!de.ReadBuffer(buf)) {
            return hdf5::error(hdf5::HDF5ErrorCode::BufferTooSmall, "failed to read string block");
        }

        auto nul_pos = std::ranges::find(buf, static_cast<byte_t>('\0'));

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

hdf5::expected<CompoundMember> CompoundMember::Deserialize(Deserializer& de) {
    CompoundMember mem{};

    auto name_result = ReadPaddedString(de);
    if (!name_result) return cstd::unexpected(name_result.error());
    mem.name = *name_result;

    mem.byte_offset = de.Read<uint32_t>();

    auto dimensionality = de.Read<uint8_t>();
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

    auto msg_result = de.ReadComplex<DatatypeMessage>();
    if (!msg_result) return cstd::unexpected(msg_result.error());

    mem.message = std::make_unique<DatatypeMessage>(*msg_result);

    return mem;
}

void CompoundDatatype::Serialize(Serializer& s) const {
    auto num_members = static_cast<uint16_t>(members.size());

    s.Write(num_members);
    // reserved (zero)
    s.Write<uint8_t>(0);

    s.Write(size);

    for (const CompoundMember& mem : members) {
        s.Write(mem);
    }
}

hdf5::expected<CompoundDatatype> CompoundDatatype::Deserialize(Deserializer& de) {
    auto num_members = de.Read<uint16_t>();
    // reserved (zero)
    de.Skip<uint8_t>();

    CompoundDatatype comp{};
    comp.members.reserve(num_members);

    comp.size = de.Read<uint32_t>();

    for (uint16_t i = 0; i < num_members; ++i) {
        auto member_result = de.ReadComplex<CompoundMember>();
        if (!member_result) return cstd::unexpected(member_result.error());
        comp.members.push_back(*member_result);
    }

    return comp;
}
