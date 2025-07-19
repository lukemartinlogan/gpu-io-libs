#pragma once

#include <bitset>
#include <variant>

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
        FixedPoint
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