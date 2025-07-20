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

    bool BigEndian() const {
        return bitset_.test(0);
    }

    bool LowPadding() const {
        return bitset_.test(1);
    }

    bool HighPadding() const {
        return bitset_.test(2);
    }

    // is signed in two's complement?
    bool Signed() const {
        return bitset_.test(3);
    }

    void Serialize(Serializer& s) const;

    static FixedPoint Deserialize(Deserializer& de);
private:
    std::bitset<4> bitset_{};
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

    ByteOrder ByteOrder() const {
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

        throw std::logic_error("Invalid byte order");
    }

    bool LowPadding() const {
        return bitset_.test(1);
    }

    bool HighPadding() const {
        return bitset_.test(2);
    }

    bool InternalPadding() const {
        return bitset_.test(3);
    }

    enum class MantissaNormalization : uint8_t { kNone, kMSBSet, kMSBImpliedSet };

    MantissaNormalization MantissaNormalization() const {
        // get bits 4 & 5
        switch ((bitset_.to_ulong() >> 4) & 0b11) {
            case 0: return MantissaNormalization::kNone;
            case 1: return MantissaNormalization::kMSBSet;
            case 2: return MantissaNormalization::kMSBImpliedSet;
            default: throw std::logic_error("invalid mantissa normalization");
        }
    }

    void Serialize(Serializer& s) const;

    static FloatingPoint Deserialize(Deserializer& de);

private:
    std::bitset<7> bitset_{};
};

struct DatatypeMessage;

struct CompoundMember {
    // null terminated to multiple of 8 bytes
    std::string name;
    // byte offset within datatype
    uint32_t byte_offset{};
    // TODO: smallvec? max size for this is 4, which would make it smaller than sizeof(std::vector)
    std::vector<uint32_t> dimension_sizes;
    // TODO: finding a better way to introduce indirection here would be nice
    std::unique_ptr<DatatypeMessage> message;

    CompoundMember() = default;

    CompoundMember(const CompoundMember& other);

    CompoundMember& operator=(const CompoundMember& other);

    CompoundMember(CompoundMember&& other) noexcept = default;

    CompoundMember& operator=(CompoundMember&& other) noexcept = default;

    void Serialize(Serializer& s) const;

    static CompoundMember Deserialize(Deserializer& de);
};

struct CompoundDatatype {
    std::vector<CompoundMember> members;
    uint32_t size{};

    void Serialize(Serializer& s) const;

    static CompoundDatatype Deserialize(Deserializer& de);
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

    std::variant<
        FixedPoint,
        FloatingPoint,
        CompoundDatatype
    > data{};

    uint16_t InternalSize() const { // NOLINT
        // TODO: correctly calculate this size
        return 0;
    }

    void Serialize(Serializer& s) const;

    static DatatypeMessage Deserialize(Deserializer& de);

private:
    static constexpr uint16_t kType = 0x03;
};