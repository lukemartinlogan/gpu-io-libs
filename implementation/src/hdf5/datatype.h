#pragma once

#include <vector>

#include "types.h"
#include "../serialization/serialization.h"

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

    std::array<uint8_t, 3> bit_field{};
    uint32_t size_bytes;
    // datatype class specific
    std::vector<byte_t> properties;

    uint16_t InternalSize() const { // NOLINT
        // TODO: correctly calculate this size
        return 0;
    }

    void Serialize(Serializer& s) const;

    static DatatypeMessage Deserialize(Deserializer& de);

private:
    static constexpr uint16_t kType = 0x03;
};