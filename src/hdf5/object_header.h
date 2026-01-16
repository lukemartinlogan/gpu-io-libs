#pragma once

#include "types.h"
#include "datatype.h"
#include "gpu_allocator.h"
#include "../serialization/buffer.h"
#include "../serialization/serialization.h"
#include "../util/align.h"

struct NilMessage {
    uint16_t size{};

    template<serde::Serializer S>
    __device__
    void Serialize(S& s) const {
        // TODO: this can be optimized
        for (uint16_t i = 0; i < size; ++i) {
            serde::Write(s, static_cast<uint8_t>(0));
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

    __device__
    DataspaceMessage(const hdf5::dim_vector<DimensionInfo>& dims, bool max_dim_present, bool perm_indices_present);

    __device__
    [[nodiscard]] bool IsMaxDimensionsPresent() const {
        return bitset_.test(0);
    }

    __device__
    [[nodiscard]] bool PermutationIndicesPresent() const {
        return bitset_.test(1);
    }

    __device__
    [[nodiscard]] size_t TotalElements() const;
    __device__
    [[nodiscard]] size_t MaxElements() const;

    template<serde::Serializer S>
    __device__
    void Serialize(S& s) const {
        serde::Write(s, static_cast<uint8_t>(0x01));

        auto dimensionality = static_cast<uint8_t>(dimensions.size());
        serde::Write(s, dimensionality);

        serde::Write(s, static_cast<uint8_t>(bitset_.to_ulong() & 0b11));

        // reserved
        serde::Write(s, static_cast<uint32_t>(0));
        serde::Write(s, static_cast<uint8_t>(0));

        for (const DimensionInfo& d : dimensions) {
            serde::Write(s, d.size);
        }

        for (const DimensionInfo& d : dimensions) {
            serde::Write(s, d.max_size);
        }

        if (PermutationIndicesPresent()) {
            for (const DimensionInfo& d : dimensions) {
                serde::Write(s, d.permutation_index);
            }
        }
    }

    template<serde::Deserializer D>
    __device__
    static hdf5::expected<DataspaceMessage> Deserialize(D& de) {
        if (serde::Read<uint8_t>(de) != static_cast<uint8_t>(0x01)) {
            return hdf5::error(hdf5::HDF5ErrorCode::InvalidVersion, "Version number was invalid");
        }

        DataspaceMessage msg{};

        auto dimensionality = serde::Read<uint8_t>(de);
        msg.dimensions.resize(dimensionality);

        msg.bitset_ = serde::Read<uint8_t>(de) & 0b11;

        // reserved
        serde::Skip(de, 5);

        for (DimensionInfo& dimension : msg.dimensions) {
            dimension.size = serde::Read<len_t>(de);
        }

        for (DimensionInfo& dimension : msg.dimensions) {
            dimension.max_size = serde::Read<len_t>(de);
        }

        if (msg.PermutationIndicesPresent()) {
            for (DimensionInfo& dimension : msg.dimensions) {
                dimension.permutation_index = serde::Read<len_t>(de);
            }
        }

        return msg;
    }

    __device__
    DataspaceMessage() = default;

private:
    cstd::bitset<2> bitset_{};

    static constexpr uint8_t kVersionNumber = 0x01;
public:
    static constexpr uint16_t kType = 0x01;
};

struct LinkInfoMessage {
    cstd::optional<uint64_t> max_creation_index;
    offset_t fractal_heap_addr = kUndefinedOffset;
    offset_t index_names_btree_addr = kUndefinedOffset;
    cstd::optional<offset_t> creation_order_btree_addr;

    template<serde::Serializer S>
    __device__
    void Serialize(S& s) const {
        serde::Write(s, static_cast<uint8_t>(0x00));

        cstd::bitset<2> flags;
        flags.set(0, max_creation_index.has_value());
        flags.set(1, creation_order_btree_addr.has_value());

        serde::Write(s, static_cast<uint8_t>(flags.to_ulong()));

        if (max_creation_index.has_value()) {
            serde::Write(s, *max_creation_index);
        }

        serde::Write(s, fractal_heap_addr);
        serde::Write(s, index_names_btree_addr);

        if (creation_order_btree_addr.has_value()) {
            serde::Write(s, *creation_order_btree_addr);
        }
    }

    template<serde::Deserializer D>
    __device__
    static hdf5::expected<LinkInfoMessage> Deserialize(D& de) {
        if (serde::Read<uint8_t>(de) != static_cast<uint8_t>(0x00)) {
            return hdf5::error(hdf5::HDF5ErrorCode::InvalidVersion, "Version number was invalid");
        }

        LinkInfoMessage msg{};

        auto flags = serde::Read<uint8_t>(de);
        cstd::bitset<2> flag_bits(flags);

        if (flag_bits.test(0)) {
            msg.max_creation_index = serde::Read<uint64_t>(de);
        } else {
            msg.max_creation_index = cstd::nullopt;
        }

        msg.fractal_heap_addr = serde::Read<offset_t>(de);
        msg.index_names_btree_addr = serde::Read<offset_t>(de);

        if (flag_bits.test(1)) {
            msg.creation_order_btree_addr = serde::Read<offset_t>(de);
        } else {
            msg.creation_order_btree_addr = cstd::nullopt;
        }

        return msg;
    }

private:
    static constexpr uint8_t kVersionNumber = 0x00;
public:
    static constexpr uint16_t kType = 0x02;
};

struct FillValueOldMessage {
    static constexpr size_t kMaxFillValueSize = 256;

    hdf5::vector<byte_t> fill_value;

    __device__ __host__
    explicit FillValueOldMessage(hdf5::HdfAllocator* alloc)
        : fill_value(alloc) {}

    __device__ __host__
    FillValueOldMessage() : fill_value(nullptr) {}

    template<serde::Serializer S>
    __device__
    void Serialize(S& s) const {
        serde::Write(s, static_cast<uint32_t>(fill_value.size()));

        if (!fill_value.empty()) {
            s.WriteBuffer(cstd::span<const byte_t>(fill_value.data(), fill_value.size()));
        }
    }

    template<serde::Deserializer D> requires iowarp::ProvidesAllocator<D>
    __device__
    static hdf5::expected<FillValueOldMessage> Deserialize(D& de) {
        FillValueOldMessage msg(de.GetAllocator());

        auto size = serde::Read<uint32_t>(de);

        if (size > 0) {
            msg.fill_value.resize(size);
            de.ReadBuffer(cstd::span<byte_t>(msg.fill_value.data(), msg.fill_value.size()));
        }

        return msg;
    }

    static constexpr uint16_t kType = 0x04;
};

struct FillValueMessage {
    static constexpr size_t kMaxFillValueSizeBytes = 256;

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

    hdf5::vector<byte_t> fill_value;
    bool has_fill_value_ = false;

    hdf5::padding<7> _pad{};

    __device__ __host__
    explicit FillValueMessage(hdf5::HdfAllocator* alloc)
        : space_alloc_time(SpaceAllocTime::kNotUsed), write_time(ValWriteTime::kOnAlloc),
          fill_value(alloc), has_fill_value_(false) {}

    __device__ __host__
    FillValueMessage()
        : space_alloc_time(SpaceAllocTime::kNotUsed), write_time(ValWriteTime::kOnAlloc),
          fill_value(nullptr), has_fill_value_(false) {}

    template<serde::Serializer S>
    __device__
    void Serialize(S& s) const {
        serde::Write(s, static_cast<uint8_t>(0x02));

        serde::Write(s, static_cast<uint8_t>(space_alloc_time));
        serde::Write(s, static_cast<uint8_t>(write_time));

        if (has_fill_value_) {
            serde::Write(s, static_cast<uint8_t>(1));
            serde::Write(s, static_cast<uint32_t>(fill_value.size()));
            s.WriteBuffer(cstd::span<const byte_t>(fill_value.data(), fill_value.size()));
        } else {
            serde::Write(s, static_cast<uint8_t>(0));
        }
    }

    template<serde::Deserializer D> requires iowarp::ProvidesAllocator<D>
    __device__
    static hdf5::expected<FillValueMessage> Deserialize(D& de) {
        if (serde::Read<uint8_t>(de) != static_cast<uint8_t>(0x02)) {
            return hdf5::error(hdf5::HDF5ErrorCode::InvalidVersion, "Version number was invalid");
        }

        FillValueMessage msg(de.GetAllocator());

        // space allocation time
        auto space_alloc = serde::Read<uint8_t>(de);
        if (space_alloc >= 4) {
            return hdf5::error(hdf5::HDF5ErrorCode::InvalidDataValue, "space alloc time was invalid");
        }

        msg.space_alloc_time = static_cast<SpaceAllocTime>(space_alloc);

        // fv write time
        auto write_time_val = serde::Read<uint8_t>(de);
        if (write_time_val >= 3) {
            return hdf5::error(hdf5::HDF5ErrorCode::InvalidDataValue, "fill value write time was invalid");
        }

        msg.write_time = static_cast<ValWriteTime>(write_time_val);

        auto defined = serde::Read<uint8_t>(de);
        if (defined == 0) {
            msg.has_fill_value_ = false;
        } else if (defined == 1) {
            auto size = serde::Read<uint32_t>(de);

            msg.fill_value.resize(size);
            de.ReadBuffer(cstd::span<byte_t>(msg.fill_value.data(), msg.fill_value.size()));
            msg.has_fill_value_ = true;
        } else {
            return hdf5::error(hdf5::HDF5ErrorCode::InvalidDataValue, "invalid fill value defined state");
        }

        return msg;
    }

private:
    static constexpr uint8_t kVersionNumber = 0x02;
public:
    static constexpr uint16_t kType = 0x05;
};

struct LinkMessage {
    template<serde::Serializer S>
    __device__
    void Serialize(S& s) const { // NOLINT
        UNREACHABLE("LinkMessage::Serialize not implemented");
    }

    template<serde::Deserializer D>
    __device__
    static hdf5::expected<LinkMessage> Deserialize(D& de) {
        return hdf5::error(hdf5::HDF5ErrorCode::NotImplemented, "LinkMessage not implemented");
    }

    static constexpr uint16_t kType = 0x06;
};

struct ExternalDataFilesMessage {
    static constexpr size_t kMaxExternalFileSlots = 16;

    struct ExternalFileSlot {
        // the byte offset within the local name heap for the name of the file
        len_t name_offset;
        // the byte offset within the file for the start of the data
        len_t file_offset;
        // total number of bytes reserved in the specified file for raw data storage
        len_t data_size;

        template<serde::Serializer S>
        __device__
        void Serialize(S& s) const {
            serde::Write(s, name_offset);
            serde::Write(s, file_offset);
            serde::Write(s, data_size);
        }

        template<serde::Deserializer D>
        __device__
        static ExternalFileSlot Deserialize(D& de) {
            return {
                .name_offset = serde::Read<len_t>(de),
                .file_offset = serde::Read<len_t>(de),
                .data_size = serde::Read<len_t>(de)
            };
        }
    };

    offset_t heap_address = kUndefinedOffset;
    cstd::inplace_vector<ExternalFileSlot, kMaxExternalFileSlots> slots;

    template<serde::Serializer S>
    __device__
    void Serialize(S& s) const {
        serde::Write(s, static_cast<uint8_t>(0x01));

        // reserved (zero)
        serde::Write(s, static_cast<uint8_t>(0));
        serde::Write(s, static_cast<uint8_t>(0));
        serde::Write(s, static_cast<uint8_t>(0));

        // allocated slots
        serde::Write(s, static_cast<uint16_t>(slots.size()));
        // used slots
        serde::Write(s, static_cast<uint16_t>(slots.size()));

        serde::Write(s, heap_address);

        for (const ExternalFileSlot& slot: slots) {
            serde::Write(s, slot);
        }
    }

    template<serde::Deserializer D>
    __device__
    static hdf5::expected<ExternalDataFilesMessage> Deserialize(D& de) {
        if (serde::Read<uint8_t>(de) != 1) {
            return hdf5::error(hdf5::HDF5ErrorCode::InvalidVersion, "ExternalDataFilesMessage: unsupported version");
        }

        serde::Skip(de, 3);

        ExternalDataFilesMessage msg{};

        auto allocated_slots = serde::Read<uint16_t>(de);
        auto used_slots = serde::Read<uint16_t>(de);

        if (allocated_slots != used_slots) {
            // "The current library simply uses the number of Used Slots for this message"
            return hdf5::error(hdf5::HDF5ErrorCode::InvalidDataValue, "ExternalDataFilesMessage: allocated slots does not match used slots");
        }

        if (used_slots > kMaxExternalFileSlots) {
            return hdf5::error(hdf5::HDF5ErrorCode::CapacityExceeded, "ExternalDataFilesMessage: too many slots");
        }

        msg.heap_address = serde::Read<offset_t>(de);

        msg.slots.reserve(used_slots);

        for (uint16_t i = 0; i < used_slots; ++i) {
            auto slot = serde::Read<ExternalFileSlot>(de);
            msg.slots.push_back(slot);
        }

        return msg;
    }

private:
    __device__ __host__
    ExternalDataFilesMessage() = default;

    static constexpr uint8_t kVersionNumber = 0x01;
public:
    static constexpr uint16_t kType = 0x07;
};

struct BogusMessage {
    static constexpr uint32_t kBogusValue = 0xdeadbeef;

    template<serde::Serializer S>
    __device__
    void Serialize(S& s) const { // NOLINT
        serde::Write(s, static_cast<uint32_t>(0xdeadbeef));
    }

    template<serde::Deserializer D>
    __device__
    static hdf5::expected<BogusMessage> Deserialize(D& de) {
        if (serde::Read<uint32_t>(de) != static_cast<uint32_t>(0xdeadbeef)) {
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

    __device__
    [[nodiscard]] uint16_t GetEstimatedNumberOfEntries() const {
        return est_num_entries.value_or(4);
    }

    __device__
    [[nodiscard]] uint16_t GetEstimatedEntryNameLength() const {
        return est_entries_name_len.value_or(8);
    }

    template<serde::Serializer S>
    __device__
    void Serialize(S& s) const {
        serde::Write(s, static_cast<uint8_t>(0x00));

        cstd::bitset<2> flags;
        flags.set(0, max_compact.has_value());

        ASSERT((max_compact.has_value() == min_dense.has_value()), "max_compact and min_dense must both be present or absent");

        flags.set(1, est_num_entries.has_value());

        ASSERT((est_num_entries.has_value() == est_entries_name_len.has_value()), "est_num_entries and est_entries_name_len must both be present or absent");

        serde::Write(s, static_cast<uint8_t>(flags.to_ulong()));

        if (max_compact.has_value() && min_dense.has_value()) {
            serde::Write(s, *max_compact);
            serde::Write(s, *min_dense);
        }
        if (est_num_entries.has_value() && est_entries_name_len.has_value()) {
            serde::Write(s, *est_num_entries);
            serde::Write(s, *est_entries_name_len);
        }
    }

    template<serde::Deserializer D>
    __device__
    static hdf5::expected<GroupInfoMessage> Deserialize(D& de) {
        if (serde::Read<uint8_t>(de) != 0) {
            return hdf5::error(hdf5::HDF5ErrorCode::InvalidVersion, "Invalid version number for GroupInfoMessage");
        }

        GroupInfoMessage msg;

        auto flags = serde::Read<uint8_t>(de);
        cstd::bitset<2> flags_bits(flags);

        if (flags_bits.test(0)) {
            msg.max_compact = serde::Read<uint16_t>(de);
            msg.min_dense = serde::Read<uint16_t>(de);
        }
        if (flags_bits.test(1)) {
            msg.est_num_entries = serde::Read<uint16_t>(de);
            msg.est_entries_name_len = serde::Read<uint16_t>(de);
        }

        return msg;
    }

private:
    static constexpr uint8_t kVersionNumber = 0x00;
public:
    static constexpr uint16_t kType = 0x0a;
};

struct FilterPipelineMessage {
    template<serde::Serializer S>
    __device__
    void Serialize(S& _s) const { // NOLINT
        UNREACHABLE("FilterPipelineMessage::Serialize not implemented");
    }

    template<serde::Deserializer D>
    __device__
    static hdf5::expected<FilterPipelineMessage> Deserialize(D& _de) {
        return hdf5::error(hdf5::HDF5ErrorCode::NotImplemented, "FilterPipelineMessage not implemented");
    }

    static constexpr uint16_t kType = 0x0b;
};

struct CompactStorageProperty {
    hdf5::vector<byte_t> raw_data;

    __device__ __host__
    explicit CompactStorageProperty(hdf5::HdfAllocator* alloc)
        : raw_data(alloc) {}

    template<serde::Serializer S>
    __device__
    void Serialize(S& s) const {
        serde::Write(s, static_cast<uint16_t>(raw_data.size()));
        s.WriteBuffer(cstd::span<const byte_t>(raw_data.data(), raw_data.size()));
    }

    template<serde::Deserializer D> requires iowarp::ProvidesAllocator<D>
    __device__
    static hdf5::expected<CompactStorageProperty> Deserialize(D& de) {
        auto size = serde::Read<uint16_t>(de);

        CompactStorageProperty msg(de.GetAllocator());
        msg.raw_data.resize(size);

        de.ReadBuffer(cstd::span<byte_t>(msg.raw_data.data(), msg.raw_data.size()));

        return msg;
    }
};

struct ContiguousStorageProperty {
    offset_t address{};
    len_t size{};

    template<serde::Serializer S>
    __device__
    void Serialize(S& s) const {
        serde::Write(s, address);
        serde::Write(s, size);
    }

    template<serde::Deserializer D>
    __device__
    static hdf5::expected<ContiguousStorageProperty> Deserialize(D& de) {
        return ContiguousStorageProperty {
            .address = serde::Read<offset_t>(de),
            .size = serde::Read<len_t>(de),
        };
    }
};

struct ChunkedStorageProperty {
    offset_t b_tree_addr = kUndefinedOffset;
    // units of array elements, not bytes
    hdf5::dim_vector<uint32_t> dimension_sizes;
    uint32_t elem_size_bytes;

    template<serde::Serializer S>
    __device__
    void Serialize(S& s) const {
        serde::Write(s, static_cast<uint8_t>(dimension_sizes.size() + 1));

        serde::Write(s, b_tree_addr);

        for (uint32_t dim_size : dimension_sizes) {
            serde::Write(s, dim_size);
        }

        serde::Write(s, elem_size_bytes);
    }

    template<serde::Deserializer D>
    __device__
    static hdf5::expected<ChunkedStorageProperty> Deserialize(D& de) {
        auto dimensionality = serde::Read<uint8_t>(de) - 1;

        ChunkedStorageProperty prop{};

        prop.b_tree_addr = serde::Read<offset_t>(de);

        prop.dimension_sizes.reserve(dimensionality); // NOLINT(*-static-accessed-through-instance)
        for (uint8_t i = 0; i < dimensionality; ++i) {
            prop.dimension_sizes.push_back(serde::Read<uint32_t>(de));
        }

        prop.elem_size_bytes = serde::Read<uint32_t>(de);

        return prop;
    }
};

struct DataLayoutMessage {
    cstd::variant<
        CompactStorageProperty,
        ContiguousStorageProperty,
        ChunkedStorageProperty
    > properties;

    template<serde::Serializer S>
    __device__
    void Serialize(S& s) const {
        serde::Write(s, static_cast<uint8_t>(0x03));

        if (cstd::holds_alternative<CompactStorageProperty>(properties)) {
            serde::Write(s, static_cast<uint8_t>(kCompact));
            serde::Write(s, cstd::get<CompactStorageProperty>(properties));
        } else if (cstd::holds_alternative<ContiguousStorageProperty>(properties)) {
            serde::Write(s, static_cast<uint8_t>(kContiguous));
            serde::Write(s, cstd::get<ContiguousStorageProperty>(properties));
        } else if (cstd::holds_alternative<ChunkedStorageProperty>(properties)) {
            serde::Write(s, static_cast<uint8_t>(kChunked));
            serde::Write(s, cstd::get<ChunkedStorageProperty>(properties));
        } else {
            UNREACHABLE("invalid data layout class");
        }
    }

    template<serde::Deserializer D>
    __device__
    static hdf5::expected<DataLayoutMessage> Deserialize(D& de) {
        if (serde::Read<uint8_t>(de) != static_cast<uint8_t>(0x03)) {
            return hdf5::error(hdf5::HDF5ErrorCode::InvalidVersion, "Version number was invalid");
        }

        auto layout_class = serde::Read<uint8_t>(de);

        DataLayoutMessage msg{};

        if (layout_class == kCompact) {
            auto result = serde::Read<CompactStorageProperty>(de);
            if (!result) return cstd::unexpected(result.error());
            msg.properties = *result;
        } else if (layout_class == kContiguous) {
            auto result = serde::Read<ContiguousStorageProperty>(de);
            if (!result) return cstd::unexpected(result.error());
            msg.properties = *result;
        } else if (layout_class == kChunked) {
            auto result = serde::Read<ChunkedStorageProperty>(de);
            if (!result) return cstd::unexpected(result.error());
            msg.properties = *result;
        } else {
            return hdf5::error(hdf5::HDF5ErrorCode::InvalidClass, "invalid data layout class");
        }

        return msg;
    }
private:
    static constexpr uint8_t kVersionNumber = 0x03;
    static constexpr uint8_t kCompact = 0, kContiguous = 1, kChunked = 2;
public:
    static constexpr uint16_t kType = 0x08;
};

struct AttributeMessage {
    // TODO(kernel-hang): shrunk to avoid GPU stack overflow (prev: 1024)
    static constexpr size_t kMaxAttributeDataSize = 32;

    hdf5::string name;
    DatatypeMessage datatype;
    DataspaceMessage dataspace;

    // TODO: is there a better way to create this
    cstd::inplace_vector<byte_t, kMaxAttributeDataSize> data;

    template<typename T>
    __device__
    hdf5::expected<T> ReadDataAs() {
        BufferDeserializer buf_de(data);

        T out = serde::Read<T>(buf_de);

        if (!buf_de.IsExhausted()) {
            return hdf5::error(hdf5::HDF5ErrorCode::IncorrectByteCount, "Invalid type was read from data");
        }

        return out;
    }

    template<serde::Serializer S> requires serde::Seekable<S>
    __device__
    void Serialize(S& s) const {
        serde::Write(s, static_cast<uint8_t>(0x01));
        // reserved (zero)
        serde::Write(s, static_cast<uint8_t>(0));

        // FIXME: these values aren't correctly calculated in the slightest
        serde::Write(s, static_cast<uint16_t>(name.size() + 1));
        serde::Write(s, static_cast<uint16_t>(0 /* TODO: datatype size */));
        serde::Write(s, static_cast<uint16_t>(0 /* TODO: dataspace size */));

        // write name
        WriteEightBytePaddedFields(s, cstd::span(
            reinterpret_cast<const byte_t*>(name.data()),
            name.size() + 1
        ));

        // write datatype
        offset_t datatype_write_start = s.GetPosition();
        serde::Write(s, datatype);
        WriteRemainingToPadToEightBytes(s, s.GetPosition() - datatype_write_start);

        // write dataspace
        offset_t dataspace_write_start = s.GetPosition();
        serde::Write(s, dataspace);
        WriteRemainingToPadToEightBytes(s, s.GetPosition() - dataspace_write_start);

        // write data
        s.WriteBuffer(data);
    }

    template<serde::Deserializer D> requires serde::ProvidesAllocator<D>
    __device__
    static hdf5::expected<AttributeMessage> Deserialize(D& de) {
        if (serde::Read<uint8_t>(de) != static_cast<uint8_t>(0x01)) {
            return hdf5::error(hdf5::HDF5ErrorCode::InvalidVersion, "Version number was invalid");
        }
        // reserved (zero)
        serde::Skip<uint8_t>(de);

        auto name_size = serde::Read<uint16_t>(de);
        auto datatype_size = serde::Read<uint16_t>(de);
        auto dataspace_size = serde::Read<uint16_t>(de);

        // TODO: windows defines max as a macro :(
        auto max_buf_size = (name_size > datatype_size ? name_size : datatype_size) > dataspace_size ? (name_size > datatype_size ? name_size : datatype_size) : dataspace_size;

        // all are generally small; this should be enough
        constexpr size_t kMaxAttributeBufferSize = kMaxAttributeDataSize * 2;

        if (max_buf_size > kMaxAttributeBufferSize) {
            return hdf5::error(
                hdf5::HDF5ErrorCode::CapacityExceeded,
                "Attribute component size exceeds maximum buffer size"
            );
        }

        cstd::array<byte_t, kMaxAttributeBufferSize> buf{};

        AttributeMessage msg{};

        // read name
        ReadEightBytePaddedData(de, cstd::span(buf.data(), name_size));

        if (buf.at(name_size - 1) != static_cast<byte_t>('\0')) {
            return hdf5::error(hdf5::HDF5ErrorCode::StringNotNullTerminated, "string read was not null-terminated");
        }

        auto name_result = hdf5::string::from_chars(reinterpret_cast<const char*>(buf.data()), name_size - 1);
        if (!name_result) return cstd::unexpected(name_result.error());
        msg.name = *name_result;

        // read datatype
        cstd::span datatype_buf(buf.data(), datatype_size);
        ReadEightBytePaddedData(de, datatype_buf);

        auto datatype_result = serde::Read<DatatypeMessage>(BufferDeserializer(datatype_buf));
        if (!datatype_result) return cstd::unexpected(datatype_result.error());
        msg.datatype = *datatype_result;

        // read dataspace
        cstd::span dataspace_buf(buf.data(), dataspace_size);
        ReadEightBytePaddedData(de, dataspace_buf);

        auto dataspace_result = serde::Read<DataspaceMessage>(BufferDeserializer(dataspace_buf));
        if (!dataspace_result) return cstd::unexpected(dataspace_result.error());
        msg.dataspace = *dataspace_result;


        size_t data_size = msg.datatype.Size() * msg.dataspace.MaxElements();
        msg.data.resize(data_size);

        de.ReadBuffer(msg.data);

        return msg;
    }
private:
    template<serde::Serializer S>
    __device__
    static void WriteRemainingToPadToEightBytes(S& s, offset_t written) {
        size_t leftover = (8 - written % 8) % 8;

        static cstd::array<const byte_t, 8> extra_padding{};
        s.WriteBuffer(cstd::span(extra_padding.data(), leftover));
    }

    template<serde::Serializer S>
    __device__
    static void WriteEightBytePaddedFields(S& s, cstd::span<const byte_t> buf) {
        s.WriteBuffer(buf);

        WriteRemainingToPadToEightBytes(s, buf.size());
    }

    template<serde::Deserializer D>
    __device__
    static void ReadEightBytePaddedData(D& de, cstd::span<byte_t> buf) {
        de.ReadBuffer(buf);

        size_t leftover = (8 - buf.size() % 8) % 8;

        static cstd::array<byte_t, 8> leftover_buf;
        de.ReadBuffer(cstd::span(leftover_buf.data(), leftover));
    }

private:
    static constexpr uint8_t kVersionNumber = 0x01;
public:
    static constexpr uint16_t kType = 0x0c;
};

struct ObjectCommentMessage {
    hdf5::string comment;

    template<serde::Serializer S>
    __device__
    void Serialize(S& s) const {
        s.WriteBuffer(cstd::span(
            reinterpret_cast<const byte_t*>(comment.c_str()),
            comment.size() + 1 // null terminator
        ));
    }

    template<serde::Deserializer D> requires serde::ProvidesAllocator<D>
    __device__
    static hdf5::expected<ObjectCommentMessage> Deserialize(D& de) {
        auto comment = ReadNullTerminatedString(de);

        if (!comment)
            return cstd::unexpected(comment.error());

        return ObjectCommentMessage{
            .comment = cstd::move(*comment)
        };
    }

    static constexpr uint16_t kType = 0x0d;
};

struct ObjectModificationTimeOldMessage {
    template<serde::Serializer S>
    __device__
    void Serialize(S& _s) const { // NOLINT
        UNREACHABLE("old object modification time message is deprecated");
    }

    template<serde::Deserializer D>
    __device__
    static hdf5::expected<ObjectModificationTimeOldMessage> Deserialize(D& _de) {
        return hdf5::error(hdf5::HDF5ErrorCode::DeprecatedFeature, "old object modification time message is deprecated");
    }

    static constexpr uint16_t kType = 0x0e;
};

struct SharedMessageTableMessage {
    offset_t table_address = kUndefinedOffset;
    uint8_t num_indices{};

    template<serde::Serializer S>
    __device__
    void Serialize(S& s) const {
        serde::Write(s, static_cast<uint8_t>(0x00));
        serde::Write(s, table_address);
        serde::Write(s, num_indices);
    }

    template<serde::Deserializer D>
    __device__
    static hdf5::expected<SharedMessageTableMessage> Deserialize(D& de) {
        if (serde::Read<uint8_t>(de) != static_cast<uint8_t>(0x00)) {
            return hdf5::error(hdf5::HDF5ErrorCode::InvalidVersion, "SharedMessageTableMessage: unsupported version");
        }

        return SharedMessageTableMessage{
            .table_address = serde::Read<offset_t>(de),
            .num_indices = serde::Read<uint8_t>(de)
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

    static constexpr uint16_t kType = 0x10;
};

static_assert(serde::TriviallySerializable<ObjectHeaderContinuationMessage>);

struct SymbolTableMessage {
    // address of v1 b-tree containing symbol table entries
    offset_t b_tree_addr = kUndefinedOffset;
    // address of local heap containing link names
    // for symbol table entries
    offset_t local_heap_addr = kUndefinedOffset;

    static constexpr uint16_t kType = 0x11;
};

static_assert(serde::TriviallySerializable<SymbolTableMessage>);

struct ObjectModificationTimeMessage {
    cstd::chrono::system_clock::time_point modification_time;

    template<serde::Serializer S>
    __device__
    void Serialize(S& s) const {
        serde::Write(s, static_cast<uint8_t>(0x01));

        serde::Write(s, static_cast<uint8_t>(0));
        serde::Write(s, static_cast<uint8_t>(0));
        serde::Write(s, static_cast<uint8_t>(0));

        auto seconds_since_epoch = cstd::chrono::duration_cast<cstd::chrono::seconds>(
            modification_time.time_since_epoch()
        ).count();

        serde::Write(s, static_cast<uint32_t>(seconds_since_epoch));
    }

    template<serde::Deserializer D>
    __device__
    static hdf5::expected<ObjectModificationTimeMessage> Deserialize(D& de) {
        if (serde::Read<uint8_t>(de) != static_cast<uint8_t>(0x01)) {
            return hdf5::error(hdf5::HDF5ErrorCode::InvalidVersion, "Version number was invalid");
        }

        // reserved (zero)
        serde::Skip(de, 3);

        auto seconds_since_epoch = serde::Read<uint32_t>(de);

        return ObjectModificationTimeMessage{
            .modification_time = cstd::chrono::system_clock::time_point{ cstd::chrono::seconds{seconds_since_epoch} }
        };
    }
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

    template<serde::Serializer S>
    __device__
    void Serialize(S& s) const {
        serde::Write(s, static_cast<uint8_t>(0x00));
        serde::Write(s, indexed_storage_internal_k);
        serde::Write(s, group_internal_k);
        serde::Write(s, group_leaf_k);
    }

    template<serde::Deserializer D>
    __device__
    static hdf5::expected<BTreeKValuesMessage> Deserialize(D& de) {
        if (serde::Read<uint8_t>(de) != static_cast<uint8_t>(0x00)) {
            return hdf5::error(hdf5::HDF5ErrorCode::InvalidVersion, "BTreeKValuesMessage: unsupported version");
        }

        return BTreeKValuesMessage{
            .indexed_storage_internal_k = serde::Read<uint16_t>(de),
            .group_internal_k = serde::Read<uint16_t>(de),
            .group_leaf_k = serde::Read<uint16_t>(de)
        };
    }

private:
    static constexpr uint8_t kVersionNumber = 0x00;
public:
    static constexpr uint16_t kType = 0x13;
};

struct DriverInfoMessage {
    static constexpr size_t kMaxDriverInfoSize = 512;

    // 8 ascii bytes
    hdf5::gpu_string<8> driver_id{};
    cstd::inplace_vector<byte_t, kMaxDriverInfoSize> driver_info;

    template<serde::Serializer S>
    __device__
    void Serialize(S& s) const {
        serde::Write(s, static_cast<uint8_t>(0x00));

        // check that driver_id has len kDriverIdSize and is all ascii
        ASSERT(
            driver_id.size() == kDriverIdSize && cstd::all_of(driver_id.begin(), driver_id.end(), [](char c) { return c >= 0 && c <= 127; }),
            "DriverInfoMessage: driver_id must be exactly eight ASCII characters"
        );

        s.WriteBuffer(cstd::span(
            reinterpret_cast<const byte_t*>(driver_id.data()),
            driver_id.size()
        ));

        serde::Write(s, static_cast<uint16_t>(driver_info.size()));

        s.WriteBuffer(driver_info);
    }

    template<serde::Deserializer D>
    __device__
    static hdf5::expected<DriverInfoMessage> Deserialize(D& de) {
        if (serde::Read<uint8_t>(de) != static_cast<uint8_t>(0x00)) {
            return hdf5::error(hdf5::HDF5ErrorCode::InvalidVersion, "DriverInfoMessage: unsupported version");
        }

        // read 8 bytes, then make a string out of it
        DriverInfoMessage msg{};

        // read id
        cstd::array<char, kDriverIdSize> id{};
        de.ReadBuffer(cstd::span(reinterpret_cast<byte_t*>(id.data()), id.size()));

        auto driver_id_result = hdf5::gpu_string<8>::from_chars(id.data(), id.size());
        if (!driver_id_result) return cstd::unexpected(driver_id_result.error());
        msg.driver_id = *driver_id_result;


        auto driver_info_size = serde::Read<uint16_t>(de);
        msg.driver_info.resize(driver_info_size);
        de.ReadBuffer(msg.driver_info);

        return msg;
    }

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

    template<serde::Serializer S>
    __device__
    void Serialize(S& s) const {
        serde::Write(s, static_cast<uint8_t>(0x00));

        cstd::bitset<2> flags;
        flags.set(0, max_creation_index.has_value());
        flags.set(1, creation_order_btree_addr.has_value());

        serde::Write(s, static_cast<uint8_t>(flags.to_ulong()));

        if (max_creation_index.has_value()) {
            serde::Write(s, *max_creation_index);
        }

        serde::Write(s, fractal_heap_addr);
        serde::Write(s, name_btree_addr);

        if (creation_order_btree_addr.has_value()) {
            serde::Write(s, *creation_order_btree_addr);
        }
    }

    template<serde::Deserializer D>
    __device__
    static hdf5::expected<AttributeInfoMessage> Deserialize(D& de) {
        if (serde::Read<uint8_t>(de) != static_cast<uint8_t>(0x00)) {
            return hdf5::error(hdf5::HDF5ErrorCode::InvalidVersion, "AttributeInfoMessage: unsupported version");
        }

        AttributeInfoMessage msg{};

        auto flags = serde::Read<uint8_t>(de);
        cstd::bitset<8> flag_bits(flags);

        if (flag_bits.test(0)) {
            msg.max_creation_index = serde::Read<uint16_t>(de);
        } else {
            msg.max_creation_index = cstd::nullopt;
        }

        msg.fractal_heap_addr = serde::Read<offset_t>(de);
        msg.name_btree_addr = serde::Read<offset_t>(de);

        if (flag_bits.test(1)) {
            msg.creation_order_btree_addr = serde::Read<offset_t>(de);
        } else {
            msg.creation_order_btree_addr = cstd::nullopt;
        }

        return msg;
    }

private:
    static constexpr uint8_t kVersionNumber = 0x00;
public:
    static constexpr uint16_t kType = 0x15;
};

struct ObjectReferenceCountMessage {
    uint32_t reference_count{};

    template<serde::Serializer S>
    __device__
    void Serialize(S& s) const {
        serde::Write(s, static_cast<uint8_t>(0x00));
        serde::Write(s, reference_count);
    }

    template<serde::Deserializer D>
    __device__
    static hdf5::expected<ObjectReferenceCountMessage> Deserialize(D& de) {
        if (serde::Read<uint8_t>(de) != static_cast<uint8_t>(0x00)) {
            return hdf5::error(hdf5::HDF5ErrorCode::InvalidVersion, "ObjectReferenceCountMessage: invalid version");
        }

        return ObjectReferenceCountMessage{
            .reference_count = serde::Read<uint32_t>(de)
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

    __device__
    [[nodiscard]] bool PersistingFreeSpace() const {
        ASSERT((small_managers.has_value() == large_managers.has_value()), "FileSpaceInfoMessage: small and large managers must be both present or both absent");

        return small_managers.has_value();
    }

    template<serde::Serializer S>
    __device__
    void Serialize(S& s) const {
        serde::Write(s, static_cast<uint8_t>(0x01));
        serde::Write(s, static_cast<uint8_t>(strategy));

        auto persisting_free_space = PersistingFreeSpace();

        serde::Write(s, static_cast<uint8_t>(persisting_free_space));
        serde::Write(s, free_space_threshold);
        serde::Write(s, file_space_page_size);
        serde::Write(s, page_end_metadata_threshold);
        serde::Write(s, eoa);

        if (persisting_free_space) {
            serde::Write(s, *small_managers);
            serde::Write(s, *large_managers);
        }
    }

    template<serde::Deserializer D>
    __device__
    static hdf5::expected<FileSpaceInfoMessage> Deserialize(D& de) {
        if (serde::Read<uint8_t>(de) != static_cast<uint8_t>(0x01)) {
            return hdf5::error(hdf5::HDF5ErrorCode::InvalidVersion, "FileSpaceInfoMessage: invalid version");
        }

        FileSpaceInfoMessage msg{};

        auto strategy = serde::Read<uint8_t>(de);

        constexpr uint16_t kStrategyCt = 4;
        if (strategy >= kStrategyCt) {
            return hdf5::error(hdf5::HDF5ErrorCode::InvalidDataValue, "FileSpaceInfoMessage: invalid strategy");
        }

        msg.strategy = static_cast<Strategy>(strategy);

        auto persisting_free_space = serde::Read<uint8_t>(de) != 0;

        msg.free_space_threshold = serde::Read<len_t>(de);
        msg.file_space_page_size = serde::Read<uint32_t>(de);
        msg.page_end_metadata_threshold = serde::Read<uint16_t>(de);
        msg.eoa = serde::Read<offset_t>(de);

        if (persisting_free_space) {
            msg.small_managers = serde::Read<cstd::array<offset_t, 6>>(de);
            msg.large_managers = serde::Read<cstd::array<offset_t, 6>>(de);
        }

        return msg;
    }

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

    __device__
    [[nodiscard]] uint16_t MessageType() const;

    __device__
    [[nodiscard]] bool DataConstant() const {
        return flags_.test(0);
    }

    __device__
    [[nodiscard]] hdf5::expected<bool> MessageShared() const {
        auto isShared = flags_.test(1);

        if (isShared && MessageType() != SharedMessageTableMessage::kType) {
            return hdf5::error(hdf5::HDF5ErrorCode::InvalidDataValue, "Only SharedMessageTableMessage can have the shared flag set");
        }

        return isShared;
    }

    __device__
    [[nodiscard]] bool ShouldNotBeShared() const {
        return flags_.test(2);
    }

    __device__
    [[nodiscard]] bool AssertUnderstandMessageForWrite() const {
        return flags_.test(3);
    }

    __device__
    [[nodiscard]] bool ShouldNotifyIfNotUnderstoodAndObjectModified() const {
        return flags_.test(4);
    }

    __device__
    [[nodiscard]] bool NotUnderstoodAndObjectModified() const {
        return flags_.test(5);
    }

    __device__
    [[nodiscard]] bool Shareable() const {
        return flags_.test(6);
    }

    __device__
    [[nodiscard]] bool AssertUnderstandMessage() const {
        return flags_.test(7);
    }

    __device__
    void NotifyNotUnderstoodAndModified() {
        if (ShouldNotifyIfNotUnderstoodAndObjectModified()) {
            flags_.set(5);
        }
    }

    template<serde::Serializer S>
    __device__
    void Serialize(S& s) const {
        serde::Write(s, MessageType());
        serde::Write(s, size);

        serde::Write(s, static_cast<uint8_t>(flags_.to_ulong()));

        // FIXME: Serializer::WriteZero<size_t>
        serde::Write(s, static_cast<uint8_t>(0));
        serde::Write(s, static_cast<uint8_t>(0));
        serde::Write(s, static_cast<uint8_t>(0));

        cstd::visit([&s](const auto& msg) { serde::Write(s, msg); }, message);
    }

    // TODO: this method probably shouldn't be public
    template<typename T, serde::Deserializer D>
    __device__
    static hdf5::expected<HeaderMessageVariant> DeserializeMessageType(D&& de) {
        using Ret = decltype(serde::Read<T>(de));

        if constexpr (std::is_same_v<Ret, T>) {
            return HeaderMessageVariant(serde::Read<T>(de));
        } else if constexpr (std::is_same_v<Ret, hdf5::expected<T>>) {
            auto result = serde::Read<T>(de);

            if (!result) {
                return cstd::unexpected(result.error());
            }

            return HeaderMessageVariant(*result);
        } else {
            UNREACHABLE("DeserializeMessageType: unsupported return type from Deserialize");
            return {};
        }
    }

    template<serde::Deserializer D>
    __device__
    static hdf5::expected<ObjectHeaderMessage> Deserialize(D& de) {
        ObjectHeaderMessage msg{};

        auto type = serde::Read<uint16_t>(de);

        constexpr uint16_t kMessageTypeCt = 0x18;
        if (type >= kMessageTypeCt) {
            return hdf5::error(hdf5::HDF5ErrorCode::InvalidType, "Not a valid message type");
        }

        msg.size = serde::Read<uint16_t>(de);
        msg.flags_ = serde::Read<uint8_t>(de);
        serde::Skip(de, 3);

        auto start = de.GetPosition();

        auto message_result = [&]() -> hdf5::expected<HeaderMessageVariant> {
            switch (type) {
                case NilMessage::kType: {
                    // FIXME: this can be optimized
                    for (uint16_t i = 0; i < msg.size; ++i) {
                        serde::Skip<uint8_t>(de);
                    }
                    return HeaderMessageVariant(NilMessage { msg.size });
                }
                case DataspaceMessage::kType:
                    return DeserializeMessageType<DataspaceMessage>(de);
                case LinkInfoMessage::kType:
                    return DeserializeMessageType<LinkInfoMessage>(de);
                case DatatypeMessage::kType:
                    return DeserializeMessageType<DatatypeMessage>(de);
                case FillValueOldMessage::kType:
                    return DeserializeMessageType<FillValueOldMessage>(de);
                case FillValueMessage::kType:
                    return DeserializeMessageType<FillValueMessage>(de);
                case LinkMessage::kType:
                    return DeserializeMessageType<LinkMessage>(de);
                case ExternalDataFilesMessage::kType:
                    return DeserializeMessageType<ExternalDataFilesMessage>(de);
                case DataLayoutMessage::kType:
                    return DeserializeMessageType<DataLayoutMessage>(de);
                case BogusMessage::kType:
                    return DeserializeMessageType<BogusMessage>(de);
                case GroupInfoMessage::kType:
                    return DeserializeMessageType<GroupInfoMessage>(de);
                case FilterPipelineMessage::kType:
                    return DeserializeMessageType<FilterPipelineMessage>(de);
                case AttributeMessage::kType:
                    return DeserializeMessageType<AttributeMessage>(de);
                case ObjectCommentMessage::kType:
                    return DeserializeMessageType<ObjectCommentMessage>(de);
                case ObjectModificationTimeOldMessage::kType:
                    return DeserializeMessageType<ObjectModificationTimeOldMessage>(de);
                case SharedMessageTableMessage::kType:
                    return DeserializeMessageType<SharedMessageTableMessage>(de);
                case ObjectHeaderContinuationMessage::kType:
                    return DeserializeMessageType<ObjectHeaderContinuationMessage>(de);
                case SymbolTableMessage::kType:
                    return DeserializeMessageType<SymbolTableMessage>(de);
                case ObjectModificationTimeMessage::kType:
                    return DeserializeMessageType<ObjectModificationTimeMessage>(de);
                case BTreeKValuesMessage::kType:
                    return DeserializeMessageType<BTreeKValuesMessage>(de);
                case DriverInfoMessage::kType:
                    return DeserializeMessageType<DriverInfoMessage>(de);
                case AttributeInfoMessage::kType:
                    return DeserializeMessageType<AttributeInfoMessage>(de);
                case ObjectReferenceCountMessage::kType:
                    return DeserializeMessageType<ObjectReferenceCountMessage>(de);
                case FileSpaceInfoMessage::kType:
                    return DeserializeMessageType<FileSpaceInfoMessage>(de);
                default:
                    return hdf5::error(hdf5::HDF5ErrorCode::InvalidType, "invalid object header message type");
            }
        }();

        if (!message_result) {
            return cstd::unexpected(message_result.error());
        }

        msg.message = *message_result;

        auto difference = de.GetPosition() - start;

        if (difference > msg.size) {
            return hdf5::error(hdf5::HDF5ErrorCode::IncorrectByteCount, "read an incorrect number of bytes");
        }

        if (msg.size > difference) {
            auto padding_ct = msg.size - difference;

            if (padding_ct >= 8) {
                return hdf5::error(hdf5::HDF5ErrorCode::IncorrectByteCount, "shouldn't be more than 8 bytes to pad to 8 bytes");
            }

            cstd::array<byte_t, 8> padding{};

            de.ReadBuffer(cstd::span(padding.data(), padding_ct));
        }

        return msg;
    }

private:
    cstd::bitset<8> flags_{};
};

struct ObjectHeader {
    // TODO(kernel-hang): shrunk to avoid GPU stack overflow (prev: 48)
    static constexpr size_t kMaxObjectHeaderMessages = 8;

    // number of hard links to this object in the current file
    uint32_t object_ref_count{};
    // number of bytes of header message data for this header
    // does not include size of object header continuation blocks
    uint32_t object_header_size{};
    // messages
    cstd::inplace_vector<ObjectHeaderMessage, kMaxObjectHeaderMessages> messages{};

    template<serde::Serializer S>
    __device__
    void Serialize(S& s) const {
        serde::Write(s, static_cast<uint8_t>(0x01));
        serde::Write(s, static_cast<uint8_t>(0));
        serde::Write(s, static_cast<uint16_t>(messages.size()));
        serde::Write(s, object_ref_count);
        serde::Write(s, object_header_size);
        // reserved (zero)
        serde::Write(s, static_cast<uint32_t>(0));

        // FIXME: handle object continuation
        for (const ObjectHeaderMessage& msg: messages) {
            serde::Write(s, msg);
        }
    }

    template<serde::Deserializer D>
    __device__
    static hdf5::expected<void> ParseObjectHeaderMessages(ObjectHeader& hd, D& de, uint32_t size_limit, uint16_t total_message_ct);

    // FIXME: ignore unknown messages
    template<serde::Deserializer D> requires serde::ProvidesAllocator<D>
    __device__
    static hdf5::expected<ObjectHeader> Deserialize(D& de) {
        if (serde::Read<uint8_t>(de) != static_cast<uint8_t>(0x01)) {
            return hdf5::error(hdf5::HDF5ErrorCode::InvalidVersion, "Version number was invalid");
        }
        // reserved (zero)
        serde::Skip<uint8_t>(de);

        auto message_count = serde::Read<uint16_t>(de);

        ObjectHeader hd{};

        hd.object_ref_count = serde::Read<uint32_t>(de);
        hd.object_header_size = serde::Read<uint32_t>(de);
        // reserved (zero)
        serde::Skip<uint32_t>(de);

        auto result = ParseObjectHeaderMessages(hd, de, hd.object_header_size, message_count);
        if (!result) return cstd::unexpected(result.error());

        return hd;
    }
private:
    friend class Object;

    static constexpr uint8_t kVersionNumber = 0x01;
    static constexpr size_t kMaxContinuationDepth = 16;
    // TODO(kernel-hang): shrunk to avoid GPU stack overflow (prev: 1024 * 8)
    static constexpr size_t kMaxHeaderMessageSerializedSizeBytes = 512;
};

// methods for use in object.h/object.cpp
inline constexpr uint32_t kPrefixSize = 8;

template<serde::Serializer S>
__device__
static void WriteHeader(S& s, uint16_t type, uint16_t size, uint8_t flags) {
    serde::Write(s, type);
    serde::Write(s, size);

    serde::Write(s, flags);
    serde::Write<cstd::array<byte_t, 3>>(s, {});
}

__device__
static len_t EmptyHeaderMessagesSize(len_t min_size) {
    // TODO: windows defines max as a macro :(
    return EightBytesAlignedSize(min_size > sizeof(ObjectHeaderContinuationMessage) + kPrefixSize ? min_size : sizeof(ObjectHeaderContinuationMessage) + kPrefixSize);
}

template<serde::Deserializer D>
__device__
hdf5::expected<void> ObjectHeader::ParseObjectHeaderMessages(ObjectHeader& hd, D& de, uint32_t size_limit, uint16_t total_message_ct) {
    struct StackFrame {
        offset_t return_pos;
        uint32_t size_limit;
        uint32_t bytes_read;
    };

    cstd::inplace_vector<StackFrame, kMaxContinuationDepth> stack;
    stack.push_back({de.GetPosition(), size_limit, 0});

    while (!stack.empty()) {
        auto& frame = stack.back();

        if (frame.bytes_read >= frame.size_limit || hd.messages.size() >= total_message_ct) {
            // Only restore position for continuation frames, not the initial frame
            if (stack.size() > 1) {
                de.SetPosition(frame.return_pos);
            }
            stack.pop_back();
            continue;
        }

        size_t before_read = de.GetPosition();

        auto msg_result = serde::Read<ObjectHeaderMessage>(de);
        if (!msg_result) {
            return cstd::unexpected(msg_result.error());
        }
        hd.messages.push_back(*msg_result);

        frame.bytes_read += de.GetPosition() - before_read;

        if (const auto* cont = cstd::get_if<ObjectHeaderContinuationMessage>(&hd.messages.back().message)) {
            offset_t return_pos = de.GetPosition();

            de.SetPosition(/* TODO: sb.base_addr + */ cont->offset);

            stack.push_back({return_pos, static_cast<uint32_t>(cont->length), 0});
        }
    }

    return {};
}