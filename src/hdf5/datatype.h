#pragma once

#include "types.h"
#include "gpu_allocator.h"
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

    __device__
    [[nodiscard]] bool BigEndian() const {
        return bitset_.test(0);
    }

    __device__
    [[nodiscard]] bool LowPadding() const {
        return bitset_.test(1);
    }

    __device__
    [[nodiscard]] bool HighPadding() const {
        return bitset_.test(2);
    }

    // is signed in two's complement?
    __device__
    [[nodiscard]] bool Signed() const {
        return bitset_.test(3);
    }

    template<serde::Serializer S>
    __device__
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
    __device__
    static hdf5::expected<FixedPoint> Deserialize(D& de) {
        FixedPoint fp{};

        // first four bits are used
        fp.bitset_ = serde::Read<uint8_t>(de) & 0x0f;
        // reserved (zero)
        serde::Skip<uint16_t>(de);

        fp.size = serde::Read<uint32_t>(de);
        fp.bit_offset = serde::Read<uint16_t>(de);
        fp.bit_precision = serde::Read<uint16_t>(de);

        return fp;
    }

    __device__ static FixedPoint i32_t();

    FixedPoint() = default;

private:
    FixedPoint(
        uint32_t size,
        uint16_t bit_offset,
        uint16_t bit_precision,
        bool big_endian,
        bool low_padding,
        bool high_padding,
        bool is_signed
    );

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

    __device__
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

    __device__
    [[nodiscard]] bool LowPadding() const {
        return bitset_.test(1);
    }

    __device__
    [[nodiscard]] bool HighPadding() const {
        return bitset_.test(2);
    }

    __device__
    [[nodiscard]] bool InternalPadding() const {
        return bitset_.test(3);
    }

    enum class MantissaNormalization : uint8_t { kNone, kMSBSet, kMSBImpliedSet };

    __device__
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
    __device__
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
    __device__
    static hdf5::expected<FloatingPoint> Deserialize(D& de) {
        FloatingPoint fp{};

        fp.bitset_ = serde::Read<uint8_t>(de) & 0x7f;
        fp.sign_location = serde::Read<uint8_t>(de);
        // reserved (zero)
        serde::Skip<uint8_t>(de);

        fp.size = serde::Read<uint32_t>(de);

        fp.bit_offset = serde::Read<uint16_t>(de);
        fp.bit_precision = serde::Read<uint16_t>(de);
        fp.exponent_location = serde::Read<uint8_t>(de);
        fp.exponent_size = serde::Read<uint8_t>(de);
        fp.mantissa_location = serde::Read<uint8_t>(de);
        fp.mantissa_size = serde::Read<uint8_t>(de);
        fp.exponent_bias = serde::Read<uint32_t>(de);

        return fp;
    }

    __device__ static FloatingPoint f32_t();

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
    // TODO(recursive-datatypes): currently, there's no great way to have recursive types on the GPU; another method for resolving datatypes might need to be implemented
    // std::unique_ptr<DatatypeMessage> parent_type{};

    __device__
    VariableLength() = default;

    __device__
    VariableLength(const VariableLength& other);
    __device__
    VariableLength& operator=(const VariableLength& other);

    __device__
    VariableLength(VariableLength&& other) noexcept = default;
    __device__
    VariableLength& operator=(VariableLength&& other) noexcept = default;

    template<serde::Serializer S>
    __device__
    void Serialize(S& s) const {
        uint8_t bitset_1 = (static_cast<uint8_t>(padding) << 4) | static_cast<uint8_t>(type);
        serde::Write(s, bitset_1);

        // Write charset in lower 4 bits, upper 4 bits zero
        serde::Write(s, static_cast<uint8_t>(static_cast<uint8_t>(charset) & 0b1111));

        // Reserved byte (zero)
        serde::Write(s, static_cast<uint8_t>(0));

        // Write size
        serde::Write(s, size);

        // TODO(recursive-datatypes)
        // If not a string, write the parent_type
        // if (type != Type::kString) {
        //     serde::Write(s, *parent_type);
        // }
    }

    template<serde::Deserializer D>
    __device__
    static hdf5::expected<VariableLength> Deserialize(D& de) {
        auto bitset_1 = serde::Read<uint8_t>(de);

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
        auto charset = serde::Read<uint8_t>(de) & 0b1111;

        if (charset >= 2) {
            return hdf5::error(hdf5::HDF5ErrorCode::InvalidDataValue, "charset wasn't valid");
        }

        vl.charset = static_cast<Charset>(charset);

        // reserved (zero)
        serde::Skip<uint8_t>(de);

        vl.size = serde::Read<uint32_t>(de);

        // TODO(recursive-datatypes)
        if (vl.type != Type::kString) {
            // auto datatype_result = serde::Read<DatatypeMessage>(de);
            // if (!datatype_result) return cstd::unexpected(datatype_result.error());
            // vl.parent_type = std::make_unique<DatatypeMessage>(*datatype_result);

            return hdf5::error(
                hdf5::HDF5ErrorCode::FeatureNotSupported,
                "Recursive VariableLength types not yet supported on GPU"
            );
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
    // TODO(recursive-datatypes): currently, there's no great way to have recursive types on the GPU; another method for resolving datatypes might need to be implemented
    // TODO: finding a better way to introduce indirection here would be nice
    // std::unique_ptr<DatatypeMessage> message;

    __device__
    CompoundMember() = default;

    __device__
    CompoundMember(const CompoundMember& other);

    __device__
    CompoundMember& operator=(const CompoundMember& other);

    __device__
    CompoundMember(CompoundMember&& other) noexcept = default;

    __device__
    CompoundMember& operator=(CompoundMember&& other) noexcept = default;

    template<serde::Serializer S>
    __device__
    void Serialize(S& s) const;

    template<serde::Deserializer D>
    __device__
    static hdf5::expected<CompoundMember> Deserialize(D& de);
};

struct CompoundDatatype {
    static constexpr size_t kMaxCompoundMembers = 16;

    hdf5::vector<CompoundMember> members;
    uint32_t size{};

    hdf5::padding<4> _pad{};

    __device__ __host__
    explicit CompoundDatatype(hdf5::HdfAllocator* alloc)
        : members(alloc), size(0) {}

    __device__ __host__
    CompoundDatatype() : members(nullptr), size(0) {}

    template<serde::Serializer S>
    __device__
    void Serialize(S& s) const;

    template<serde::Deserializer D> requires iowarp::ProvidesAllocator<D>
    __device__
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

    hdf5::padding<6> _pad{};

    cstd::variant<
        FixedPoint,
        FloatingPoint,
        CompoundDatatype,
        VariableLength
    > data{};

    __device__
    [[nodiscard]] uint16_t Size() const {
        return cstd::visit([](const auto& elem) { return elem.size; }, data);
    }

    template<serde::Serializer S>
    __device__
    void Serialize(S& s) const;

    template<serde::Deserializer D>
    __device__
    static hdf5::expected<DatatypeMessage> Deserialize(D& de);

public:
    __device__ static DatatypeMessage i32_t();
    __device__ static DatatypeMessage f32_t();

public:
    static constexpr uint16_t kType = 0x03;
};

template<serde::Serializer S>
__device__
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

    // TODO(recursive-datatypes)
    // serde::Write(s, *message);
}

template<serde::Deserializer D>
__device__
hdf5::expected<CompoundMember> CompoundMember::Deserialize(D& de) {
    CompoundMember mem{};

    auto name_result = ReadPaddedString(de);
    if (!name_result) return cstd::unexpected(name_result.error());
    mem.name = *name_result;

    mem.byte_offset = serde::Read<uint32_t>(de);

    auto dimensionality = serde::Read<uint8_t>(de);
    // reserved (zero)
    serde::Skip<uint8_t>(de);
    serde::Skip<uint8_t>(de);
    serde::Skip<uint8_t>(de);
    // dimension permutation (unused)
    serde::Skip<uint32_t>(de);
    // reserved (zero)
    serde::Skip<uint32_t>(de);

    for (uint8_t i = 0; i < 4; ++i) {
        auto size = serde::Read<uint32_t>(de);

        if (size < dimensionality) {
            mem.dimension_sizes.push_back(size);
        }
    }

    auto msg_result = serde::Read<DatatypeMessage>(de);
    if (!msg_result) return cstd::unexpected(msg_result.error());

    // TODO(CUDA): Re-enable when recursive types are supported
    // mem.message = std::make_unique<DatatypeMessage>(*msg_result);
    // return mem;

    return hdf5::error(
        hdf5::HDF5ErrorCode::FeatureNotSupported,
        "Compound datatypes with nested members not yet supported on GPU"
    );
}

template<serde::Serializer S>
__device__
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

template<serde::Deserializer D> requires iowarp::ProvidesAllocator<D>
__device__
hdf5::expected<CompoundDatatype> CompoundDatatype::Deserialize(D& de) {
    auto num_members = serde::Read<uint16_t>(de);
    // reserved (zero)
    serde::Skip<uint8_t>(de);

    CompoundDatatype comp(de.GetAllocator());

    if (num_members > kMaxCompoundMembers) {
        return hdf5::error(
            hdf5::HDF5ErrorCode::CapacityExceeded,
            "Compound datatype has too many members"
        );
    }

    comp.size = serde::Read<uint32_t>(de);

    for (uint16_t i = 0; i < num_members; ++i) {
        auto member_result = serde::Read<CompoundMember>(de);
        if (!member_result) return cstd::unexpected(member_result.error());
        comp.members.push_back(*member_result);
    }

    return comp;
}

template<serde::Serializer S>
__device__
void DatatypeMessage::Serialize(S& s) const {
    auto high = static_cast<uint8_t>(version);
    auto low = static_cast<uint8_t>(class_v);

    uint8_t class_and_version = (high << 4) | (low & 0x0f);

    serde::Write(s, class_and_version);

    cstd::visit([&s](const auto& data) { return data.Serialize(s); }, data);
}

template<serde::Deserializer D>
__device__
hdf5::expected<DatatypeMessage> DatatypeMessage::Deserialize(D& de) {
    auto class_and_version = serde::Read<uint8_t>(de);

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
            auto result = serde::Read<FixedPoint>(de);
            if (!result) return cstd::unexpected(result.error());
            msg.data = *result;
            break;
        }
        case Class::kFloatingPoint: {
            auto result = serde::Read<FloatingPoint>(de);
            if (!result) return cstd::unexpected(result.error());
            msg.data = *result;
            break;
        }
        case Class::kCompound: {
            auto result = serde::Read<CompoundDatatype>(de);
            if (!result) return cstd::unexpected(result.error());
            msg.data = *result;
            break;
        }
        case Class::kVariableLength: {
            auto result = serde::Read<VariableLength>(de);
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