#pragma once
#include <chrono>
#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

#include "types.h"
#include "datatype.h"
#include "../serialization/buffer.h"
#include "../serialization/serialization.h"

struct NilMessage {
    uint16_t size{};

    uint16_t InternalSize() const { // NOLINT
        return size;
    }

    void Serialize(Serializer& s) const {
        // TODO: this can be optimized
        for (uint16_t i = 0; i < size; ++i) {
            s.Write<uint8_t>(0);
        }
    }

private:
    static constexpr uint16_t kType = 0x00;
};

struct DimensionInfo {
    len_t size;
    len_t max_size;
    len_t permutation_index;
};

struct DataspaceMessage {
    std::vector<DimensionInfo> dimensions;

    [[nodiscard]] bool IsMaxDimensionsPresent() const {
        return bitset_.test(0);
    }

    [[nodiscard]] bool PermutationIndicesPresent() const {
        return bitset_.test(1);
    }

    uint16_t InternalSize() const { // NOLINT
        uint16_t dimension_info_size = sizeof(DimensionInfo) * dimensions.size();
        uint16_t header_size = 8 * sizeof(byte_t);

        return dimension_info_size * header_size;
    }

    size_t TotalElements() const;
    size_t MaxElements() const;

    void Serialize(Serializer& s) const;

    static DataspaceMessage Deserialize(Deserializer& de);

private:
    std::bitset<2> bitset_;

    static constexpr uint8_t kVersionNumber = 0x01;
    static constexpr uint16_t kType = 0x01;
};

struct FillValueMessage {
    enum class SpaceAllocTime {
        kNotUsed = 0,
        kEarly = 1,
        kLate = 2,
        kIncremental = 3,
    } space_alloc_time;

    enum class ValWriteTime {
        kOnAlloc = 0,
        kNever = 1,
        kIfExplicit = 2,
    } write_time;

    std::optional<std::vector<byte_t>> fill_value;

    uint16_t InternalSize() const { // NOLINT
        return 0; // FIXME
    }

    void Serialize(Serializer& s) const;

    static FillValueMessage Deserialize(Deserializer& de);
private:
    static constexpr uint8_t kVersionNumber = 0x02;
    static constexpr uint16_t kType = 0x05;
};

struct CompactStorageProperty {
    std::vector<byte_t> raw_data;

    void Serialize(Serializer& s) const;

    static CompactStorageProperty Deserialize(Deserializer& de);
};

struct ContiguousStorageProperty {
    offset_t address{};
    len_t size{};

    void Serialize(Serializer& s) const;

    static ContiguousStorageProperty Deserialize(Deserializer& de);
};

struct ChunkedStorageProperty {
    offset_t b_tree_addr = kUndefinedOffset;
    // units of array elements, not bytes
    std::vector<uint32_t> dimension_sizes;
    uint32_t elem_size_bytes;

    void Serialize(Serializer& s) const;

    static ChunkedStorageProperty Deserialize(Deserializer& de);
};

struct DataLayoutMessage {
    std::variant<
        CompactStorageProperty,
        ContiguousStorageProperty,
        ChunkedStorageProperty
    > properties;

    uint16_t InternalSize() const { // NOLINT
        return 0; // FIXME
    }

    void Serialize(Serializer& s) const;

    static DataLayoutMessage Deserialize(Deserializer& de);
private:
    static constexpr uint8_t kVersionNumber = 0x03;
    static constexpr uint16_t kType = 0x05;

    static constexpr uint8_t kCompact = 0, kContiguous = 1, kChunked = 2;
};

struct AttributeMessage {
    std::string name;
    DatatypeMessage datatype;
    DataspaceMessage dataspace;

    // TODO: is there a better way to create this
    std::vector<byte_t> data;

    uint16_t InternalSize() const { // NOLINT
        return 0;
    }

    template<typename T>
    T ReadDataAs() {
        BufferDeserializer buf_de(data);

        T out = buf_de.Read<T>();

        if (!buf_de.IsExhausted()) {
            throw std::runtime_error("Invalid type was read from data");
        }

        return out;
    }

    void Serialize(Serializer& s) const;

    static AttributeMessage Deserialize(Deserializer& de);

private:
    static constexpr uint8_t kVersionNumber = 0x01;
    static constexpr uint16_t kType = 0x0c;
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

struct ObjectModificationTimeMessage {
    std::chrono::system_clock::time_point modification_time;

    uint16_t InternalSize() const { // NOLINT
        return 8 * sizeof(byte_t);
    }

    void Serialize(Serializer& s) const;

    static ObjectModificationTimeMessage Deserialize(Deserializer& de);
private:
    static constexpr uint16_t kType = 0x12;
    static constexpr uint8_t kVersionNumber = 0x01;
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
    } type{};

    // if this gets too large, put it on the heap
    std::variant<
        NilMessage,
        DataspaceMessage,
        DatatypeMessage,
        FillValueMessage,
        DataLayoutMessage,
        AttributeMessage,
        ObjectHeaderContinuationMessage,
        SymbolTableMessage,
        ObjectModificationTimeMessage
    > message{};
    uint8_t flags{};

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
