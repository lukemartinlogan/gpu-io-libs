#pragma once
#include <cstdint>
#include <variant>
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

struct ObjectHeaderContinuationMessage {
    // where header continuation block is located
    offset_t offset = kUndefinedOffset;
    // size of header continuation block
    len_t length{};

    uint16_t InternalSize() const { // NOLINT
        return sizeof(ObjectHeaderContinuationMessage);
    }

    void Serialize(Serializer& s) const {
        s.WriteRaw(*this);
    }

    static ObjectHeaderContinuationMessage Deserialize(Deserializer& de) {
        return de.ReadRaw<ObjectHeaderContinuationMessage>();
    }

private:
    static constexpr uint16_t kType = 0x10;
};

struct SymbolTableMessage {
    // address of v1 b-tree containing symbol table entries
    offset_t b_tree_addr = kUndefinedOffset;
    // address of local heap containing link names
    // for symbol table entries
    offset_t local_heap_addr = kUndefinedOffset;

    uint16_t InternalSize() const { // NOLINT
        return sizeof(SymbolTableMessage);
    }

    void Serialize(Serializer& s) const {
        s.WriteRaw(*this);
    }

    static SymbolTableMessage Deserialize(Deserializer& de) {
        return de.ReadRaw<SymbolTableMessage>();
    }

private:
    static constexpr uint16_t kType = 0x11;
};

struct ObjectHeaderMessage {
    // TODO: this can be stored in the variant
    enum class Type : uint16_t {
        // ignore message, variable length
        kNil = 0x0000,
        // exactly 1 req for datasets
        // variable len based on num of dimensions
        kDataspace = 0x0001,
        // ?current state of links
        kLinkInfo = 0x0002,
        // exactly 1 req for datasets
        // datatype for each elem of dataset
        kDatatype = 0x0003,
        // uninit value
        kFillValueOld = 0x0004,
        // uninit value, same datatype as dataset
        kFillValue = 0x0005,
        // info for link in group object header
        kLink = 0x0006,
        // indicated data for object stored out of file
        kExternalDataFiles = 0x0007,
        // how elems of multi dimensions array are stored
        kDataLayout = 0x0008,
        // for testing, should never appear
        kBogus = 0x0009,
        // info for constants defining group behavior
        kGroupInfo = 0x000a,
        //
        kFilterPipeline = 0x000b,
        //
        kAttribute = 0x000c,
        //
        kObjectComment = 0x000d,
        //
        kObjectModificationTimeOld = 0x000e,
        //
        kSharedMessageTable = 0x000f,
        // location containing more header messages for current data object
        // can be used if header blocks are too large or likely to change over time
        kObjectHeaderContinuation = 0x0010,
        //
        kSymbolTable = 0x0011,
        //
        kObjectModificationTime = 0x0012,
        //
        kBTreeKValues = 0x0013,
        //
        kDriverInfo = 0x0014,
        //
        kAttributeInfo = 0x0015,
        //
        kObjectRefCount = 0x0016,
        //
        kFileSpaceInfo = 0x0017,
    } type;

    // if this gets too large, put it on the heap
    std::variant<
        DatatypeMessage,
        ObjectHeaderContinuationMessage,
        SymbolTableMessage
    > message{};
    uint8_t flags;

    [[nodiscard]] uint16_t InternalSize() const {
        uint16_t size = std::visit(
            [](const auto& m) { return m.InternalSize(); },
            message
        );

        return size + 8 * sizeof(byte_t);
    }

    void Serialize(Serializer& s) const;

    static ObjectHeaderMessage Deserialize(Deserializer& de);
};

struct ObjectHeader {
    // total number of messages listed
    // includes continuation messages
    uint16_t message_count{};
    // number of hard links to this object in the current file
    uint32_t object_ref_count{};
    // number of bytes of header message data for this header
    // does not include size of object header continuation blocks
    uint32_t object_header_size{};
    // messages
    std::vector<ObjectHeaderMessage> messages{};

    void Serialize(Serializer& s) const;

    static ObjectHeader Deserialize(Deserializer& de);
private:
    static constexpr uint8_t kVersionNumber = 0x01;
};
