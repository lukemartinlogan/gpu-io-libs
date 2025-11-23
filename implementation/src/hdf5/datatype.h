#pragma once

#include <bitset>
#include <memory>
#include <stdexcept>
#include <variant>
#include <vector>

#include "types.h"
#include "../serialization/serialization.h"
#include "../util/string.h"

struct FixedPoint {
    // bit offset of first significant bit of fixed point value in datatype
    // bit offset specifies num of bits "to right of" value (set to LowPadding() value)
    uint16_t bit_offset{};
    // num of bits of precision of fixed point value in datatype
    // this, combined with datatype's element size and bit offset
    // specifies the num of bits "to left of" value (which are set to HighPadding () value)
    uint8_t bit_precision{};
    // size in bytes
    uint32_t size{};

    [[nodiscard]] bool BigEndian() const {
        return bitset_.test(0);
    }

    [[nodiscard]] bool LowPadding() const {
        return bitset_.test(1);
    }

    [[nodiscard]] bool HighPadding() const {
        return bitset_.test(2);
    }

    // is signed in two's complement?
    [[nodiscard]] bool Signed() const {
        return bitset_.test(3);
    }

    template<serde::Serializer S>
    void Serialize(S& s) const {
        // first four bits are used
        serde::Write(s, static_cast<uint8_t>(bitset_.to_ulong() & 0x0f));
        // reserved (zero)
        serde::Write(s, static_cast<uint16_t>(0));

        serde::Write(s, size);
        serde::Write(s, bit_offset);
        serde::Write(s, bit_precision);
    }

    template<serde::Deserializer D>
    static hdf5::expected<FixedPoint> Deserialize(D& de) {
        FixedPoint fp{};

        // first four bits are used
        fp.bitset_ = serde::Read<D, uint8_t>(de) & 0x0f;
        // reserved (zero)
        serde::Skip<D, uint16_t>(de);

        fp.size = serde::Read<D, uint32_t>(de);
        fp.bit_offset = serde::Read<D, uint16_t>(de);
        fp.bit_precision = serde::Read<D, uint16_t>(de);

        return fp;
    }

private:
    cstd::bitset<4> bitset_{};
};

// TODO: don't use a bitset internally for the enums?
struct FloatingPoint {
    // TODO: comments
    uint32_t size{};
    uint8_t sign_location{};
    uint16_t bit_offset{};
    uint16_t bit_precision{};
    uint8_t exponent_location{};
    uint8_t exponent_size{};
    uint8_t mantissa_location{};
    uint8_t mantissa_size{};
    uint32_t exponent_bias{};

    enum class ByteOrder : uint8_t { kLittleEndian, kBigEndian, kVAXEndian };

    [[nodiscard]] hdf5::expected<ByteOrder> GetByteOrder() const {
        const bool _0 = bitset_.test(0);
        const bool _6 = bitset_.test(6);

        if (!_0 && !_6) {
            return ByteOrder::kLittleEndian;
        }
        if (_0 && !_6) {
            return ByteOrder::kBigEndian;
        }
        if (_0 && _6) { // NOLINT
            return ByteOrder::kVAXEndian;
        }

        return hdf5::error(hdf5::HDF5ErrorCode::InvalidDataValue, "Invalid byte order");
    }

    [[nodiscard]] bool LowPadding() const {
        return bitset_.test(1);
    }

    [[nodiscard]] bool HighPadding() const {
        return bitset_.test(2);
    }

    [[nodiscard]] bool InternalPadding() const {
        return bitset_.test(3);
    }

    enum class MantissaNormalization : uint8_t { kNone, kMSBSet, kMSBImpliedSet };

    [[nodiscard]] hdf5::expected<MantissaNormalization> GetMantissaNormalization() const {
        // get bits 4 & 5
        switch ((bitset_.to_ulong() >> 4) & 0b11) {
            case 0: return MantissaNormalization::kNone;
            case 1: return MantissaNormalization::kMSBSet;
            case 2: return MantissaNormalization::kMSBImpliedSet;
            default: return hdf5::error(hdf5::HDF5ErrorCode::InvalidDataValue, "Invalid mantissa normalization");
        }
    }

    template<serde::Serializer S>
    void Serialize(S& s) const {
        serde::Write(s, static_cast<uint8_t>(bitset_.to_ulong() & 0x7f));
        serde::Write(s, sign_location);
        // reserved (zero)
        serde::Write(s, static_cast<uint8_t>(0));

        serde::Write(s, size);

        serde::Write(s, bit_offset);
        serde::Write(s, bit_precision);
        serde::Write(s, exponent_location);
        serde::Write(s, exponent_size);
        serde::Write(s, mantissa_location);
        serde::Write(s, mantissa_size);
        serde::Write(s, exponent_bias);
    }

    template<serde::Deserializer D>
    static hdf5::expected<FloatingPoint> Deserialize(D& de) {
        FloatingPoint fp{};

        fp.bitset_ = serde::Read<D, uint8_t>(de) & 0x7f;
        fp.sign_location = serde::Read<D, uint8_t>(de);
        // reserved (zero)
        serde::Skip<D, uint8_t>(de);

        fp.size = serde::Read<D, uint32_t>(de);

        fp.bit_offset = serde::Read<D, uint16_t>(de);
        fp.bit_precision = serde::Read<D, uint16_t>(de);
        fp.exponent_location = serde::Read<D, uint8_t>(de);
        fp.exponent_size = serde::Read<D, uint8_t>(de);
        fp.mantissa_location = serde::Read<D, uint8_t>(de);
        fp.mantissa_size = serde::Read<D, uint8_t>(de);
        fp.exponent_bias = serde::Read<D, uint32_t>(de);

        return fp;
    }

    static const FloatingPoint f32_t;

    FloatingPoint() = default;

private:
    FloatingPoint(
        uint32_t size, uint8_t sign_location, uint16_t bit_offset,
        uint16_t bit_precision, uint8_t exponent_location, uint8_t exponent_size,
        uint8_t mantissa_location, uint8_t mantissa_size, uint32_t exponent_bias,
        ByteOrder byte_order, MantissaNormalization norm,
        bool low_padding, bool high_padding, bool internal_padding
    );

private:
    cstd::bitset<7> bitset_{};
};

struct DatatypeMessage;

struct VariableLength {
    enum class Type : uint8_t { kSequence = 0, kString = 1 } type{};
    enum class PaddingType : uint8_t { kNullTerminate = 0, kNullPadded = 1, kSpacePad = 2 } padding{};
    enum class Charset : uint8_t { kASCII = 0, kUTF8 = 1 } charset{};

    uint32_t size{};
    // nullable
    std::unique_ptr<DatatypeMessage> parent_type{};

    VariableLength() = default;

    VariableLength(const VariableLength& other);
    VariableLength& operator=(const VariableLength& other);

    VariableLength(VariableLength&& other) noexcept = default;
    VariableLength& operator=(VariableLength&& other) noexcept = default;

    template<serde::Serializer S>
    void Serialize(S& s) const {
        uint8_t bitset_1 = (static_cast<uint8_t>(padding) << 4) | static_cast<uint8_t>(type);
        serde::Write(s, bitset_1);

        // Write charset in lower 4 bits, upper 4 bits zero
        serde::Write(s, static_cast<uint8_t>(static_cast<uint8_t>(charset) & 0b1111));

        // Reserved byte (zero)
        serde::Write(s, static_cast<uint8_t>(0));

        // Write size
        serde::Write(s, size);

        // If not a string, write the parent_type
        if (type != Type::kString) {
            serde::Write(s, *parent_type);
        }
    }

    template<serde::Deserializer D>
    static hdf5::expected<VariableLength> Deserialize(D& de) {
        auto bitset_1 = serde::Read<D, uint8_t>(de);

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
        auto charset = serde::Read<D, uint8_t>(de) & 0b1111;

        if (charset >= 2) {
            return hdf5::error(hdf5::HDF5ErrorCode::InvalidDataValue, "charset wasn't valid");
        }

        vl.charset = static_cast<Charset>(charset);

        // reserved (zero)
        serde::Skip<D, uint8_t>(de);

        vl.size = serde::Read<D, uint32_t>(de);

        if (vl.type != Type::kString) {
            auto datatype_result = serde::Read<D, DatatypeMessage>(de);
            if (!datatype_result) return cstd::unexpected(datatype_result.error());
            vl.parent_type = std::make_unique<DatatypeMessage>(*datatype_result);
        }

        return vl;
    }
};

struct CompoundMember {
    // null terminated to multiple of 8 bytes
    hdf5::string name;
    // byte offset within datatype
    uint32_t byte_offset{};
    // according to spec, only up to four dimensions are allowed
    cstd::inplace_vector<uint32_t, 4> dimension_sizes{};
    // TODO: finding a better way to introduce indirection here would be nice
    std::unique_ptr<DatatypeMessage> message;

    CompoundMember() = default;

    CompoundMember(const CompoundMember& other);

    CompoundMember& operator=(const CompoundMember& other);

    CompoundMember(CompoundMember&& other) noexcept = default;

    CompoundMember& operator=(CompoundMember&& other) noexcept = default;

    template<serde::Serializer S>
    void Serialize(S& s) const;

    template<serde::Deserializer D>
    static hdf5::expected<CompoundMember> Deserialize(D& de);
};

struct CompoundDatatype {
    static constexpr size_t kMaxCompoundMembers = 16;

    cstd::inplace_vector<CompoundMember, kMaxCompoundMembers> members;
    uint32_t size{};

    template<serde::Serializer S>
    void Serialize(S& s) const;

    template<serde::Deserializer D>
    static hdf5::expected<CompoundDatatype> Deserialize(D& de);
};

// TODO: make meaningful data accessible
struct DatatypeMessage {
    // TODO: strongly typed variant
    enum class Version : uint8_t {
        // used by early library versions for compound datatypes with explicit array fields
        kEarlyCompound = 1,
        // array
        kArray = 2,
        // VAX byte ordered type
        kVAX = 3,
        kRevisedReference = 4,
        // complex number
        kComplexNumber = 5,
    } version;

    // TODO: eventually store this field in the variant
    enum class Class : uint8_t {
        kFixedPoint = 0,
        kFloatingPoint = 1,
        kTime = 2,
        kString = 3,
        kBitField = 4,
        kOpaque = 5,
        kCompound = 6,
        kReference = 7,
        kEnumerated = 8,
        kVariableLength = 9,
        kArray = 10,
        kComplex = 11,
    } class_v;

    cstd::variant<
        FixedPoint,
        FloatingPoint,
        CompoundDatatype,
        VariableLength
    > data{};

    [[nodiscard]] uint16_t Size() const {
        return cstd::visit([](const auto& elem) { return elem.size; }, data);
    }

    template<serde::Serializer S>
    void Serialize(S& s) const;

    template<serde::Deserializer D>
    static hdf5::expected<DatatypeMessage> Deserialize(D& de);

public:
    static const DatatypeMessage f32_t;

public:
    static constexpr uint16_t kType = 0x03;
};

template<serde::Serializer S>
void CompoundMember::Serialize(S& s) const {
    // includes null terminator
    WritePaddedString(name, s);

    serde::Write(s, byte_offset);

    auto dimensionality = static_cast<uint8_t>(dimension_sizes.size());
    serde::Write(s, dimensionality);

    // reserved (zero)
    serde::Write(s, static_cast<uint8_t>(0));
    serde::Write(s, static_cast<uint8_t>(0));
    serde::Write(s, static_cast<uint8_t>(0));
    // dimension permutation (unused)
    serde::Write(s, static_cast<uint32_t>(0));
    // reserved (zero)
    serde::Write(s, static_cast<uint32_t>(0));

    for (uint8_t i = 0; i < 4; ++i) {
        if (i < dimensionality) {
            serde::Write(s, dimension_sizes.at(i));
        } else {
            serde::Write(s, static_cast<uint32_t>(0));
        }
    }

    serde::Write(s, *message);
}

template<serde::Deserializer D>
hdf5::expected<CompoundMember> CompoundMember::Deserialize(D& de) {
    CompoundMember mem{};

    auto name_result = ReadPaddedString(de);
    if (!name_result) return cstd::unexpected(name_result.error());
    mem.name = *name_result;

    mem.byte_offset = serde::Read<D, uint32_t>(de);

    auto dimensionality = serde::Read<D, uint8_t>(de);
    // reserved (zero)
    serde::Skip<D, uint8_t>(de);
    serde::Skip<D, uint8_t>(de);
    serde::Skip<D, uint8_t>(de);
    // dimension permutation (unused)
    serde::Skip<D, uint32_t>(de);
    // reserved (zero)
    serde::Skip<D, uint32_t>(de);

    for (uint8_t i = 0; i < 4; ++i) {
        auto size = serde::Read<D, uint32_t>(de);

        if (size < dimensionality) {
            mem.dimension_sizes.push_back(size);
        }
    }

    auto msg_result = serde::Read<D, DatatypeMessage>(de);
    if (!msg_result) return cstd::unexpected(msg_result.error());

    mem.message = std::make_unique<DatatypeMessage>(*msg_result);

    return mem;
}

template<serde::Serializer S>
void CompoundDatatype::Serialize(S& s) const {
    auto num_members = static_cast<uint16_t>(members.size());

    serde::Write(s, num_members);
    // reserved (zero)
    serde::Write(s, static_cast<uint8_t>(0));

    serde::Write(s, size);

    for (const CompoundMember& mem : members) {
        serde::Write(s, mem);
    }
}

template<serde::Deserializer D>
hdf5::expected<CompoundDatatype> CompoundDatatype::Deserialize(D& de) {
    auto num_members = serde::Read<D, uint16_t>(de);
    // reserved (zero)
    serde::Skip<D, uint8_t>(de);

    CompoundDatatype comp{};

    if (num_members > kMaxCompoundMembers) {
        return hdf5::error(
            hdf5::HDF5ErrorCode::CapacityExceeded,
            "Compound datatype has too many members"
        );
    }

    comp.size = serde::Read<D, uint32_t>(de);

    for (uint16_t i = 0; i < num_members; ++i) {
        auto member_result = serde::Read<D, CompoundMember>(de);
        if (!member_result) return cstd::unexpected(member_result.error());
        comp.members.push_back(*member_result);
    }

    return comp;
}

template<serde::Serializer S>
void DatatypeMessage::Serialize(S& s) const {
    auto high = static_cast<uint8_t>(version);
    auto low = static_cast<uint8_t>(class_v);

    uint8_t class_and_version = (high << 4) | (low & 0x0f);

    serde::Write(s, class_and_version);

    cstd::visit([&s](const auto& data) { return data.Serialize(s); }, data);
}

template<serde::Deserializer D>
hdf5::expected<DatatypeMessage> DatatypeMessage::Deserialize(D& de) {
    auto class_and_version = serde::Read<D, uint8_t>(de);

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
            auto result = serde::Read<D, FixedPoint>(de);
            if (!result) return cstd::unexpected(result.error());
            msg.data = *result;
            break;
        }
        case Class::kFloatingPoint: {
            auto result = serde::Read<D, FloatingPoint>(de);
            if (!result) return cstd::unexpected(result.error());
            msg.data = *result;
            break;
        }
        case Class::kCompound: {
            auto result = serde::Read<D, CompoundDatatype>(de);
            if (!result) return cstd::unexpected(result.error());
            msg.data = *result;
            break;
        }
        case Class::kVariableLength: {
            auto result = serde::Read<D, VariableLength>(de);
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