#pragma once

#include <bitset>
#include <memory>
#include <stdexcept>
#include <variant>
#include <vector>

#include "types.h"
#include "../serialization/serialization.h"

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

    void Serialize(VirtualSerializer& s) const;

    static hdf5::expected<FixedPoint> Deserialize(VirtualDeserializer& de);
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

    void Serialize(VirtualSerializer& s) const;

    static hdf5::expected<FloatingPoint> Deserialize(VirtualDeserializer& de);

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

    void Serialize(VirtualSerializer& s) const;
    static hdf5::expected<VariableLength> Deserialize(VirtualDeserializer& de);
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

    void Serialize(VirtualSerializer& s) const;

    static hdf5::expected<CompoundMember> Deserialize(VirtualDeserializer& de);
};

struct CompoundDatatype {
    static constexpr size_t kMaxCompoundMembers = 16;

    cstd::inplace_vector<CompoundMember, kMaxCompoundMembers> members;
    uint32_t size{};

    void Serialize(VirtualSerializer& s) const;

    static hdf5::expected<CompoundDatatype> Deserialize(VirtualDeserializer& de);
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

    void Serialize(VirtualSerializer& s) const;

    static hdf5::expected<DatatypeMessage> Deserialize(VirtualDeserializer& de);

public:
    static const DatatypeMessage f32_t;

public:
    static constexpr uint16_t kType = 0x03;
};