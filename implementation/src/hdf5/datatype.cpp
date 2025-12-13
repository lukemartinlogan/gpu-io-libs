#include "datatype.h"

#include <stdexcept>


// TODO: have this method presented in a different way?
__device__ __host__
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

__device__ __host__
VariableLength::VariableLength(const VariableLength& other)
    : type(other.type),
      padding(other.padding),
      charset(other.charset),
      size(other.size)
{
    // TODO(recursive-datatypes)
    // if (other.parent_type) {
    //     parent_type = std::make_unique<DatatypeMessage>(*other.parent_type);
    // } else {
    //     parent_type = nullptr;
    // }
}

__device__ __host__
VariableLength& VariableLength::operator=(const VariableLength& other) {
    if (this == &other) {
        return *this;
    }

    type = other.type;
    padding = other.padding;
    charset = other.charset;
    size = other.size;

    // TODO(recursive-datatypes)
    // if (other.parent_type) {
    //     parent_type = std::make_unique<DatatypeMessage>(*other.parent_type);
    // } else {
    //     parent_type = nullptr;
    // }

    return *this;
}

__device__ __host__
CompoundMember::CompoundMember(const CompoundMember& other)
        : name(other.name),
          byte_offset(other.byte_offset),
          dimension_sizes(other.dimension_sizes),
          // message(std::make_unique<DatatypeMessage>(*other.message)) /* TODO(recursive-datatypes) */
{ }

__device__ __host__
CompoundMember& CompoundMember::operator=(const CompoundMember& other) {
    if (this == &other) {
        return *this;
    }

    name = other.name;
    byte_offset = other.byte_offset;
    dimension_sizes = other.dimension_sizes;
    // TODO(recursive-datatypes)
    // message = std::make_unique<DatatypeMessage>(*other.message);
    return *this;
}
