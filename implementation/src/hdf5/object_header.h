#pragma once
#include <chrono>
#include <cstdint>
#include <vector>

#include "types.h"
#include "datatype.h"
#include "../serialization/buffer.h"
#include "../serialization/serialization.h"

struct NilMessage {
    uint16_t size{};

    void Serialize(Serializer& s) const {
        // TODO: this can be optimized
        for (uint16_t i = 0; i < size; ++i) {
            s.Write<uint8_t>(0);
        }
    }

    static constexpr uint16_t kType = 0x00;
};

struct DimensionInfo {
    len_t size;
    len_t max_size;
    len_t permutation_index;
};

struct DataspaceMessage {
    hdf5::dim_vector<DimensionInfo> dimensions;

    DataspaceMessage(const hdf5::dim_vector<DimensionInfo>&, bool max_dim_present, bool perm_indices_present);

    [[nodiscard]] bool IsMaxDimensionsPresent() const {
        return bitset_.test(0);
    }

    [[nodiscard]] bool PermutationIndicesPresent() const {
        return bitset_.test(1);
    }

    [[nodiscard]] size_t TotalElements() const;
    [[nodiscard]] size_t MaxElements() const;

    void Serialize(Serializer& s) const;

    static hdf5::expected<DataspaceMessage> Deserialize(Deserializer& de);

    DataspaceMessage() = default;

private:
    cstd::bitset<2> bitset_;

    static constexpr uint8_t kVersionNumber = 0x01;
public:
    static constexpr uint16_t kType = 0x01;
};

struct LinkInfoMessage {
    cstd::optional<uint64_t> max_creation_index;
    offset_t fractal_heap_addr = kUndefinedOffset;
    offset_t index_names_btree_addr = kUndefinedOffset;
    cstd::optional<offset_t> creation_order_btree_addr;

    void Serialize(Serializer& s) const;

    static hdf5::expected<LinkInfoMessage> Deserialize(Deserializer& de);

private:
    static constexpr uint8_t kVersionNumber = 0x00;
public:
    static constexpr uint16_t kType = 0x02;
};

struct FillValueOldMessage {
    std::vector<byte_t> fill_value;

    void Serialize(Serializer& s) const {
        s.Write<uint32_t>(fill_value.size());

        if (!fill_value.empty()) {
            s.WriteBuffer(fill_value);
        }
    }

    static hdf5::expected<FillValueOldMessage> Deserialize(Deserializer& de) {
        FillValueOldMessage msg{};

        auto size = de.Read<uint32_t>();

        if (size > 0) {
            msg.fill_value.resize(size);
            de.ReadBuffer(msg.fill_value);
        } else {
            msg.fill_value.clear();
        }

        return msg;
    }

    static constexpr uint16_t kType = 0x04;
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

    cstd::optional<std::vector<byte_t>> fill_value;

    void Serialize(Serializer& s) const;

    static hdf5::expected<FillValueMessage> Deserialize(Deserializer& de);

private:
    static constexpr uint8_t kVersionNumber = 0x02;
public:
    static constexpr uint16_t kType = 0x05;
};

struct LinkMessage {
    void Serialize(Serializer& s) const { // NOLINT
        UNREACHABLE("LinkMessage::Serialize not implemented");
    }

    static hdf5::expected<LinkMessage> Deserialize(Deserializer& de) {
        return hdf5::error(hdf5::HDF5ErrorCode::NotImplemented, "LinkMessage not implemented");
    }

    static constexpr uint16_t kType = 0x06;
};

struct ExternalDataFilesMessage {
    struct ExternalFileSlot {
        // the byte offset within the local name heap for the name of the file
        len_t name_offset;
        // the byte offset within the file for the start of the data
        len_t file_offset;
        // total number of bytes reserved in the specified file for raw data storage
        len_t data_size;

        void Serialize(Serializer& s) const {
            s.Write(name_offset);
            s.Write(file_offset);
            s.Write(data_size);
        }

        static ExternalFileSlot Deserialize(Deserializer& de) {
            return {
                .name_offset = de.Read<len_t>(),
                .file_offset = de.Read<len_t>(),
                .data_size = de.Read<len_t>()
            };
        }
    };

    offset_t heap_address;
    std::vector<ExternalFileSlot> slots;

    void Serialize(Serializer& s) const;

    static hdf5::expected<ExternalDataFilesMessage> Deserialize(Deserializer& de);

private:
    static constexpr uint8_t kVersionNumber = 0x01;
public:
    static constexpr uint16_t kType = 0x07;
};

struct BogusMessage {
    static constexpr uint32_t kBogusValue = 0xdeadbeef;

    void Serialize(Serializer& s) const { // NOLINT
        s.Write<uint32_t>(kBogusValue);
    }

    static hdf5::expected<BogusMessage> Deserialize(Deserializer& de) {
        if (de.Read<uint32_t>() != kBogusValue) {
            return hdf5::error(hdf5::HDF5ErrorCode::InvalidDataValue, "BogusMessage: value is not 0xdeadbeef");
        }

        return BogusMessage{};
    }

    static constexpr uint16_t kType = 0x09;
};

struct GroupInfoMessage {
    // maximum number of links to store "compactly"
    cstd::optional<uint16_t> max_compact;
    // minimum number of links to store "densely"
    cstd::optional<uint16_t> min_dense;
    // estimated number of entries in the group
    cstd::optional<uint16_t> est_num_entries;
    // estimated length of entry name
    cstd::optional<uint16_t> est_entries_name_len;

    [[nodiscard]] uint16_t GetEstimatedNumberOfEntries() const {
        return est_num_entries.value_or(4);
    }

    [[nodiscard]] uint16_t GetEstimatedEntryNameLength() const {
        return est_entries_name_len.value_or(8);
    }

    void Serialize(Serializer& s) const;

    static hdf5::expected<GroupInfoMessage> Deserialize(Deserializer& de);

private:
    static constexpr uint8_t kVersionNumber = 0x00;
public:
    static constexpr uint16_t kType = 0x0a;
};

struct FilterPipelineMessage {
    void Serialize(Serializer& _s) const { // NOLINT
        UNREACHABLE("FilterPipelineMessage::Serialize not implemented");
    }

    static hdf5::expected<FilterPipelineMessage> Deserialize(Deserializer& _de) {
        return hdf5::error(hdf5::HDF5ErrorCode::NotImplemented, "FilterPipelineMessage not implemented");
    }

    static constexpr uint16_t kType = 0x0b;
};

struct CompactStorageProperty {
    std::vector<byte_t> raw_data;

    void Serialize(Serializer& s) const;

    static hdf5::expected<CompactStorageProperty> Deserialize(Deserializer& de);
};

struct ContiguousStorageProperty {
    offset_t address{};
    len_t size{};

    void Serialize(Serializer& s) const;

    static hdf5::expected<ContiguousStorageProperty> Deserialize(Deserializer& de);
};

struct ChunkedStorageProperty {
    offset_t b_tree_addr = kUndefinedOffset;
    // units of array elements, not bytes
    hdf5::dim_vector<uint32_t> dimension_sizes;
    uint32_t elem_size_bytes;

    void Serialize(Serializer& s) const;

    static hdf5::expected<ChunkedStorageProperty> Deserialize(Deserializer& de);
};

struct DataLayoutMessage {
    cstd::variant<
        CompactStorageProperty,
        ContiguousStorageProperty,
        ChunkedStorageProperty
    > properties;

    void Serialize(Serializer& s) const;

    static hdf5::expected<DataLayoutMessage> Deserialize(Deserializer& de);
private:
    static constexpr uint8_t kVersionNumber = 0x03;
    static constexpr uint8_t kCompact = 0, kContiguous = 1, kChunked = 2;
public:
    static constexpr uint16_t kType = 0x08;
};

struct AttributeMessage {
    std::string name;
    DatatypeMessage datatype;
    DataspaceMessage dataspace;

    // TODO: is there a better way to create this
    std::vector<byte_t> data;

    template<typename T>
    hdf5::expected<T> ReadDataAs() {
        BufferDeserializer buf_de(data);

        T out = buf_de.Read<T>();

        if (!buf_de.IsExhausted()) {
            return hdf5::error(hdf5::HDF5ErrorCode::IncorrectByteCount, "Invalid type was read from data");
        }

        return out;
    }

    void Serialize(Serializer& s) const;

    static hdf5::expected<AttributeMessage> Deserialize(Deserializer& de);

private:
    static constexpr uint8_t kVersionNumber = 0x01;
public:
    static constexpr uint16_t kType = 0x0c;
};

struct ObjectCommentMessage {
    std::string comment;

    void Serialize(Serializer& s) const;

    static hdf5::expected<ObjectCommentMessage> Deserialize(Deserializer& de);

    static constexpr uint16_t kType = 0x0d;
};

struct ObjectModificationTimeOldMessage {
    void Serialize(Serializer& _s) const { // NOLINT
        UNREACHABLE("old object modification time message is deprecated");
    }

    static hdf5::expected<ObjectModificationTimeOldMessage> Deserialize(Deserializer& _de) {
        return hdf5::error(hdf5::HDF5ErrorCode::DeprecatedFeature, "old object modification time message is deprecated");
    }

    static constexpr uint16_t kType = 0x0e;
};

struct SharedMessageTableMessage {
    offset_t table_address = kUndefinedOffset;
    uint8_t num_indices{};

    void Serialize(Serializer& s) const {
        s.Write(kVersionNumber);
        s.Write(table_address);
        s.Write(num_indices);
    }

    static hdf5::expected<SharedMessageTableMessage> Deserialize(Deserializer& de) {
        if (de.Read<uint8_t>() != kVersionNumber) {
            return hdf5::error(hdf5::HDF5ErrorCode::InvalidVersion, "SharedMessageTableMessage: unsupported version");
        }

        return SharedMessageTableMessage{
            .table_address = de.Read<offset_t>(),
            .num_indices = de.Read<uint8_t>()
        };
    }

private:
    static constexpr uint8_t kVersionNumber = 0x00;
public:
    static constexpr uint16_t kType = 0x0f;
};

struct ObjectHeaderContinuationMessage {
    // where header continuation block is located
    offset_t offset = kUndefinedOffset;
    // size of header continuation block
    len_t length{};

    void Serialize(Serializer& s) const {
        s.WriteRaw(*this);
    }

    static hdf5::expected<ObjectHeaderContinuationMessage> Deserialize(Deserializer& de) {
        return de.ReadRaw<ObjectHeaderContinuationMessage>();
    }

    static constexpr uint16_t kType = 0x10;
};

struct SymbolTableMessage {
    // address of v1 b-tree containing symbol table entries
    offset_t b_tree_addr = kUndefinedOffset;
    // address of local heap containing link names
    // for symbol table entries
    offset_t local_heap_addr = kUndefinedOffset;

    void Serialize(Serializer& s) const {
        s.WriteRaw(*this);
    }

    static hdf5::expected<SymbolTableMessage> Deserialize(Deserializer& de) {
        return de.ReadRaw<SymbolTableMessage>();
    }

    static constexpr uint16_t kType = 0x11;
};

struct ObjectModificationTimeMessage {
    cstd::chrono::system_clock::time_point modification_time;

    void Serialize(Serializer& s) const;

    static hdf5::expected<ObjectModificationTimeMessage> Deserialize(Deserializer& de);
private:
    static constexpr uint8_t kVersionNumber = 0x01;
public:
    static constexpr uint16_t kType = 0x12;
};

struct BTreeKValuesMessage {
    // node k value for each internal node
    uint16_t indexed_storage_internal_k{};
    // node k value for each internal node in group b-tree
    uint16_t group_internal_k{};
    // node k value for each leaf node in group b-tree
    uint16_t group_leaf_k{};

    void Serialize(Serializer& s) const {
        s.Write(kVersionNumber);
        s.Write(indexed_storage_internal_k);
        s.Write(group_internal_k);
        s.Write(group_leaf_k);
    }

    static hdf5::expected<BTreeKValuesMessage> Deserialize(Deserializer& de) {
        if (de.Read<uint8_t>() != kVersionNumber) {
            return hdf5::error(hdf5::HDF5ErrorCode::InvalidVersion, "BTreeKValuesMessage: unsupported version");
        }

        return BTreeKValuesMessage{
            .indexed_storage_internal_k = de.Read<uint16_t>(),
            .group_internal_k = de.Read<uint16_t>(),
            .group_leaf_k = de.Read<uint16_t>()
        };
    }

private:
    static constexpr uint8_t kVersionNumber = 0x00;
public:
    static constexpr uint16_t kType = 0x13;
};

struct DriverInfoMessage {
    // 8 ascii bytes
    std::string driver_id{};
    std::vector<byte_t> driver_info;

    void Serialize(Serializer& s) const;

    static hdf5::expected<DriverInfoMessage> Deserialize(Deserializer& de);

private:
    static constexpr size_t kDriverIdSize = 8;
    static constexpr uint8_t kVersionNumber = 0x00;
public:
    static constexpr uint16_t kType = 0x14;
};

struct AttributeInfoMessage {
    // maximum creation order index value for attributes on object
    cstd::optional<uint16_t> max_creation_index;
    // address of fractal heap for dense attributes
    offset_t fractal_heap_addr = kUndefinedOffset;
    // address of v2 b-tree for names of densely stored attributes
    offset_t name_btree_addr = kUndefinedOffset;
    // addr of v2 b-tree to index creation order of desnsely stored attributes
    cstd::optional<offset_t> creation_order_btree_addr;

    void Serialize(Serializer& s) const;

    static hdf5::expected<AttributeInfoMessage> Deserialize(Deserializer& de);

private:
    static constexpr uint8_t kVersionNumber = 0x00;
public:
    static constexpr uint16_t kType = 0x15;
};

struct ObjectReferenceCountMessage {
    uint32_t reference_count{};

    void Serialize(Serializer& s) const {
        s.Write(kVersionNumber);
        s.Write(reference_count);
    }

    static hdf5::expected<ObjectReferenceCountMessage> Deserialize(Deserializer& de) {
        if (de.Read<uint8_t>() != kVersionNumber) {
            return hdf5::error(hdf5::HDF5ErrorCode::InvalidVersion, "ObjectReferenceCountMessage: invalid version");
        }

        return ObjectReferenceCountMessage{
            .reference_count = de.Read<uint32_t>()
        };
    }

private:
    static constexpr uint8_t kVersionNumber = 0x00;
public:
    static constexpr uint16_t kType = 0x16;
};

// FIXME: refactor this message
struct FileSpaceInfoMessage {
    // strategy used to manage file space
    enum class Strategy : uint8_t {
        // free space managers, aggregators, virtual file drivers
        kFSMAggregators = 0,
        // free space managers, embedded page aggregation, virtual file drivers
        kPage = 1,
        // aggregators, virtual file drivers
        kAggregators = 2,
        // virtual vile drivers
        kNone = 3
    } strategy{};

    // smallest free-space section size that free-space manager will track
    len_t free_space_threshold{};
    // file space page size, used when paged aggregation is enabled
    uint32_t file_space_page_size{};
    // smallest free-space section size at end of a page that free space manager will track
    // used when pageed aggregation is enabled
    uint16_t page_end_metadata_threshold{};
    // the end of allocated [space] of free-space manager header and section info
    // for self-referential free-space managers when persisting free-space
    offset_t eoa{};

    // 6 small-sized free-space managers
    cstd::optional<cstd::array<offset_t, 6>> small_managers;
    // 6 large-sized free-space managers
    cstd::optional<cstd::array<offset_t, 6>> large_managers;

    [[nodiscard]] bool PersistingFreeSpace() const {
        ASSERT((small_managers.has_value() == large_managers.has_value()), "FileSpaceInfoMessage: small and large managers must be both present or both absent");

        return small_managers.has_value();
    }

    void Serialize(Serializer& s) const;

    static hdf5::expected<FileSpaceInfoMessage> Deserialize(Deserializer& de);

private:
    static constexpr uint8_t kVersionNumber = 0x01;
public:
    static constexpr uint16_t kType = 0x17;
};

using HeaderMessageVariant = cstd::variant<
    // ignore message, variable length
    NilMessage, // 0x00
    // exactly 1 req for datasets
    // variable len based on num of dimensions
    DataspaceMessage, // 0x01
    // ?current state of links
    LinkInfoMessage, // 0x02
    // exactly 1 req for datasets
    // datatype for each elem of dataset
    DatatypeMessage, // 0x03
    // uninit value, same datatype as dataset
    FillValueOldMessage, // 0x04
    // uninit value, same datatype as dataset
    FillValueMessage, // 0x05
    // info for link in group object header, TODO
    LinkMessage, // 0x06
    // indicated data for object stored out of file
    ExternalDataFilesMessage, // 0x07
    // how elems of multi dimensions array are stored
    DataLayoutMessage, // 0x08
    // for testing, should never appear
    BogusMessage, // 0x09
    // info for constants defining group behavior
    GroupInfoMessage, // 0x0a
    // filter pipeline, TODO
    FilterPipelineMessage, // 0x0b
    AttributeMessage, // 0x0c
    // short description about object
    ObjectCommentMessage, // 0x0d
    // old object modification time, deprecated
    ObjectModificationTimeOldMessage, // 0x0e
    // shared object header message indices
    SharedMessageTableMessage, // 0x0f
    ObjectHeaderContinuationMessage, // 0x10
    SymbolTableMessage, // 0x11
    ObjectModificationTimeMessage, // 0x12
    // contains k values for b-trees, only found in superblock extension
    BTreeKValuesMessage, // 0x13
    // contains driver id and info
    DriverInfoMessage, // 0x14
    // infromation about attributes on an object
    AttributeInfoMessage, // 0x15
    // number of hard links to this object in the current file
    // only present in v2+ of object headers; if not present, refct is assumed 1
    ObjectReferenceCountMessage, // 0x16
    // file space info, used to manage free space in file
    FileSpaceInfoMessage // 0x17
>;

struct ObjectHeaderMessage {
    // if this gets too large, put it on the heap
    HeaderMessageVariant message{};

    uint16_t size{};

    [[nodiscard]] uint16_t MessageType() const;

    [[nodiscard]] bool DataConstant() const {
        return flags_.test(0);
    }

    [[nodiscard]] bool MessageShared() const {
        auto isShared = flags_.test(1);

        if (isShared && MessageType() != SharedMessageTableMessage::kType) {
            throw std::runtime_error("Only SharedMessageTableMessage can have the shared flag set");
        }

        return isShared;
    }

    [[nodiscard]] bool ShouldNotBeShared() const {
        return flags_.test(2);
    }

    [[nodiscard]] bool AssertUnderstandMessageForWrite() const {
        return flags_.test(3);
    }

    [[nodiscard]] bool ShouldNotifyIfNotUnderstoodAndObjectModified() const {
        return flags_.test(4);
    }

    [[nodiscard]] bool NotUnderstoodAndObjectModified() const {
        return flags_.test(5);
    }

    [[nodiscard]] bool Shareable() const {
        return flags_.test(6);
    }

    [[nodiscard]] bool AssertUnderstandMessage() const {
        return flags_.test(7);
    }

    void NotifyNotUnderstoodAndModified() {
        if (ShouldNotifyIfNotUnderstoodAndObjectModified()) {
            flags_.set(5);
        }
    }

    void Serialize(Serializer& s) const;

    static hdf5::expected<ObjectHeaderMessage> Deserialize(Deserializer& de);

private:
    cstd::bitset<8> flags_{};
};

struct ObjectHeader {
    // number of hard links to this object in the current file
    uint32_t object_ref_count{};
    // number of bytes of header message data for this header
    // does not include size of object header continuation blocks
    uint32_t object_header_size{};
    // messages
    std::vector<ObjectHeaderMessage> messages{};

    void Serialize(Serializer& s) const;

    // FIXME: ignore unknown messages
    static hdf5::expected<ObjectHeader> Deserialize(Deserializer& de);
private:
    friend class Object;

    static constexpr uint8_t kVersionNumber = 0x01;
};
