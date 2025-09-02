#include <numeric>
#include <stdexcept>

#include "object_header.h"
#include "datatype.h"
#include "../serialization/buffer.h"

size_t DataspaceMessage::TotalElements() const {
    return std::accumulate(
        dimensions.begin(), dimensions.end(),
        1,
        [](size_t acc, const DimensionInfo& info) {
            return acc * info.size;
        }
    );
}

size_t DataspaceMessage::MaxElements() const {
    return std::accumulate(
        dimensions.begin(), dimensions.end(),
        1,
        [](size_t acc, const DimensionInfo& info) {
            return acc * info.max_size;
        }
    );
}

void DataspaceMessage::Serialize(Serializer& s) const {
    s.Write(kVersionNumber);

    auto dimensionality = static_cast<uint8_t>(dimensions.size());
    s.Write(dimensionality);

    s.Write(static_cast<uint8_t>(bitset_.to_ulong() & 0b11));

    // reserved
    s.Write<uint32_t>(0);
    s.Write<uint8_t>(0);

    for (const DimensionInfo& d : dimensions) {
        s.Write(d.size);
    }

    for (const DimensionInfo& d : dimensions) {
        s.Write(d.max_size);
    }

    if (PermutationIndicesPresent()) {
        for (const DimensionInfo& d : dimensions) {
            s.Write(d.permutation_index);
        }
    }
}

DataspaceMessage DataspaceMessage::Deserialize(Deserializer& de) {
    if (de.Read<uint8_t>() != kVersionNumber) {
        throw std::runtime_error("Version number was invalid");
    }

    DataspaceMessage msg{};

    auto dimensionality = de.Read<uint8_t>();
    msg.dimensions.resize(dimensionality);

    msg.bitset_ = de.Read<uint8_t>() & 0b11;

    // reserved
    de.Skip<5>();

    for (DimensionInfo& dimension : msg.dimensions) {
        dimension.size = de.Read<len_t>();
    }

    for (DimensionInfo& dimension : msg.dimensions) {
        dimension.max_size = de.Read<len_t>();
    }

    if (msg.PermutationIndicesPresent()) {
        for (DimensionInfo& dimension : msg.dimensions) {
            dimension.permutation_index = de.Read<len_t>();
        }
    }

    return msg;
}

DataspaceMessage::DataspaceMessage(const std::vector<DimensionInfo>& dimensions, bool max_dim_present, bool perm_indices_present) {
    if (dimensions.size() > 255) {
        throw std::logic_error("DataspaceMessage cannot have more than 255 dimensions");
    }

    this->dimensions = dimensions;

    bitset_.set(0, max_dim_present);
    bitset_.set(1, perm_indices_present);
}

void LinkInfoMessage::Serialize(Serializer& s) const {
    s.Write(kVersionNumber);

    std::bitset<2> flags;
    flags.set(0, max_creation_index.has_value());
    flags.set(1, creation_order_btree_addr.has_value());

    s.Write(static_cast<uint8_t>(flags.to_ulong()));

    if (max_creation_index.has_value()) {
        s.Write(*max_creation_index);
    }

    s.Write(fractal_heap_addr);
    s.Write(index_names_btree_addr);

    if (creation_order_btree_addr.has_value()) {
        s.Write(*creation_order_btree_addr);
    }
}

LinkInfoMessage LinkInfoMessage::Deserialize(Deserializer& de) {
    if (de.Read<uint8_t>() != kVersionNumber) {
        throw std::runtime_error("Version number was invalid");
    }

    LinkInfoMessage msg{};

    auto flags = de.Read<uint8_t>();
    std::bitset<2> flag_bits(flags);

    if (flag_bits.test(0)) {
        msg.max_creation_index = de.Read<uint64_t>();
    } else {
        msg.max_creation_index = std::nullopt;
    }

    msg.fractal_heap_addr = de.Read<offset_t>();
    msg.index_names_btree_addr = de.Read<offset_t>();

    if (flag_bits.test(1)) {
        msg.creation_order_btree_addr = de.Read<offset_t>();
    } else {
        msg.creation_order_btree_addr = std::nullopt;
    }

    return msg;
}

void FillValueMessage::Serialize(Serializer& s) const {
    s.Write(kVersionNumber);

    s.Write(static_cast<uint8_t>(space_alloc_time));
    s.Write(static_cast<uint8_t>(write_time));

    if (fill_value.has_value()) {
        s.Write<uint8_t>(1);
        s.Write(static_cast<uint32_t>(fill_value->size()));
        s.WriteBuffer(*fill_value);
    } else {
        s.Write<uint8_t>(0);
    }
}

FillValueMessage FillValueMessage::Deserialize(Deserializer& de) {
    if (de.Read<uint8_t>() != kVersionNumber) {
        throw std::runtime_error("Version number was invalid");
    }

    FillValueMessage msg{};

    // space allocation time
    auto space_alloc = de.Read<uint8_t>();
    if (space_alloc >= 4) {
        throw std::runtime_error("space alloc time was invalid");
    }

    msg.space_alloc_time = static_cast<SpaceAllocTime>(space_alloc);

    // fv write time
    auto write_time = de.Read<uint8_t>();
    if (write_time >= 3) {
        throw std::runtime_error("fill value write time was invalid");
    }

    msg.write_time = static_cast<ValWriteTime>(write_time);

    auto defined = de.Read<uint8_t>();
    if (defined == 0) {
        msg.fill_value = std::nullopt;
    } else if (defined == 1) {
        auto size = de.Read<uint32_t>();

        std::vector<byte_t> fv(size);
        de.ReadBuffer(fv);

        msg.fill_value = fv;
    } else {
        throw std::runtime_error("invalid fill value defined state");
    }

    return msg;
}

void ExternalDataFilesMessage::Serialize(Serializer& s) const {
    s.Write<uint8_t>(kVersionNumber);

    // reserved (zero)
    s.Write<uint8_t>(0);
    s.Write<uint8_t>(0);
    s.Write<uint8_t>(0);

    // allocated slots
    s.Write<uint16_t>(slots.size());
    // used slots
    s.Write<uint16_t>(slots.size());

    s.Write(heap_address);

    for (const ExternalFileSlot& slot: slots) {
        s.WriteComplex(slot);
    }
}

ExternalDataFilesMessage ExternalDataFilesMessage::Deserialize(Deserializer& de) {
    if (de.Read<uint8_t>() != 1) {
        throw std::runtime_error("ExternalDataFilesMessage: unsupported version");
    }

    de.Skip<3>(); // reserved

    ExternalDataFilesMessage msg{};

    auto allocated_slots = de.Read<uint16_t>();
    auto used_slots = de.Read<uint16_t>();

    if (allocated_slots != used_slots) {
        // "The current library simply uses the number of Used Slots for this message"
        throw std::logic_error("ExternalDataFilesMessage: allocated slots does not match used slots");
    }

    msg.heap_address = de.Read<offset_t>();

    msg.slots.reserve(used_slots);

    for (uint16_t i = 0; i < used_slots; ++i) {
        msg.slots.push_back(de.ReadComplex<ExternalFileSlot>());
    }

    return msg;
}

void GroupInfoMessage::Serialize(Serializer& s) const {
    s.Write(kVersionNumber);

    std::bitset<2> flags;
    flags.set(0, max_compact.has_value());

    if (max_compact.has_value() != min_dense.has_value()) {
        throw std::logic_error("max_compact and min_dense must both be present or absent");
    }

    flags.set(1, est_num_entries.has_value());

    if (est_num_entries.has_value() != est_entries_name_len.has_value()) {
        throw std::logic_error("est_num_entries and est_entries_name_len must both be present or absent");
    }

    s.Write(static_cast<uint8_t>(flags.to_ulong()));

    if (max_compact.has_value() && min_dense.has_value()) {
        s.Write(*max_compact);
        s.Write(*min_dense);
    }
    if (est_num_entries.has_value() && est_entries_name_len.has_value()) {
        s.Write(*est_num_entries);
        s.Write(*est_entries_name_len);
    }
}

GroupInfoMessage GroupInfoMessage::Deserialize(Deserializer& de) {
    if (de.Read<uint8_t>() != 0) {
        throw std::runtime_error("Invalid version number for GroupInfoMessage");
    }

    GroupInfoMessage msg;

    auto flags = de.Read<uint8_t>();
    std::bitset<2> flags_bits(flags);

    if (flags_bits.test(0)) {
        msg.max_compact = de.Read<uint16_t>();
        msg.min_dense = de.Read<uint16_t>();
    }
    if (flags_bits.test(1)) {
        msg.est_num_entries = de.Read<uint16_t>();
        msg.est_entries_name_len = de.Read<uint16_t>();
    }

    return msg;
}

void CompactStorageProperty::Serialize(Serializer& s) const {
    s.Write<uint16_t>(raw_data.size());
    s.WriteBuffer(raw_data);
}

CompactStorageProperty CompactStorageProperty::Deserialize(Deserializer& de) {
    auto size = de.Read<uint16_t>();

    CompactStorageProperty msg{};
    msg.raw_data.resize(size);

    de.ReadBuffer(msg.raw_data);

    return msg;
}

void ContiguousStorageProperty::Serialize(Serializer& s) const {
    s.Write(address);
    s.Write(size);
}

ContiguousStorageProperty ContiguousStorageProperty::Deserialize(Deserializer& de) {
    ContiguousStorageProperty prop{};

    prop.address = de.Read<offset_t>();
    prop.size = de.Read<len_t>();

    return prop;
}

void ChunkedStorageProperty::Serialize(Serializer& s) const {
    s.Write<uint8_t>(dimension_sizes.size() + 1);

    s.Write(b_tree_addr);

    for (uint32_t dim_size : dimension_sizes) {
        s.Write(dim_size);
    }

    s.Write(elem_size_bytes);
}

ChunkedStorageProperty ChunkedStorageProperty::Deserialize(Deserializer& de) {
    auto dimensionality = de.Read<uint8_t>() - 1;

    ChunkedStorageProperty prop{};

    prop.b_tree_addr = de.Read<offset_t>();

    prop.dimension_sizes.reserve(dimensionality);
    for (uint8_t i = 0; i < dimensionality; ++i) {
        prop.dimension_sizes.push_back(de.Read<uint32_t>());
    }

    prop.elem_size_bytes = de.Read<uint32_t>();

    return prop;
}

void DataLayoutMessage::Serialize(Serializer& s) const {
    s.Write(kVersionNumber);

    if (std::holds_alternative<CompactStorageProperty>(properties)) {
        s.Write<uint8_t>(kCompact);
        s.WriteComplex(std::get<CompactStorageProperty>(properties));
    } else if (std::holds_alternative<ContiguousStorageProperty>(properties)) {
        s.Write<uint8_t>(kContiguous);
        s.WriteComplex(std::get<ContiguousStorageProperty>(properties));
    } else if (std::holds_alternative<ChunkedStorageProperty>(properties)) {
        s.Write<uint8_t>(kChunked);
        s.WriteComplex(std::get<ChunkedStorageProperty>(properties));
    } else {
        throw std::runtime_error("invalid data layout class");
    }
}

DataLayoutMessage DataLayoutMessage::Deserialize(Deserializer& de) {
    if (de.Read<uint8_t>() != kVersionNumber) {
        throw std::runtime_error("Version number was invalid");
    }

    auto layout_class = de.Read<uint8_t>();

    DataLayoutMessage msg{};

    if (layout_class == kCompact) {
        msg.properties = de.ReadComplex<CompactStorageProperty>();
    } else if (layout_class == kContiguous) {
        msg.properties = de.ReadComplex<ContiguousStorageProperty>();
    } else if (layout_class == kChunked) {
        msg.properties = de.ReadComplex<ChunkedStorageProperty>();
    } else {
        throw std::runtime_error("invalid data layout class");
    }

    return msg;
}

void WriteEightBytePaddedFields(Serializer& s, std::span<const byte_t> buf) {
    s.WriteBuffer(buf);

    size_t leftover = (8 - buf.size() % 8) % 8;

    static std::array<const byte_t, 8> extra_padding{};
    s.WriteBuffer(std::span(extra_padding.data(), leftover));
}

void AttributeMessage::Serialize(Serializer& s) const {
    s.Write(kVersionNumber);
    // reserved (zero)
    s.Write<uint8_t>(0);

    // FIXME: these values aren't correctly calculated in the slightest
    s.Write<uint16_t>(name.size() + 1);
    s.Write<uint16_t>(0 /* TODO: datatype size */);
    s.Write<uint16_t>(0 /* TODO: dataspace size */);

    // write name
    WriteEightBytePaddedFields(s,std::span(
        reinterpret_cast<const byte_t*>(name.data()),
        name.size() + 1
    ));

    // write data
    DynamicBufferSerializer datatype_bufs;
    datatype_bufs.WriteComplex(datatype);
    // FIXME: change this function to write directly into 's'?
    WriteEightBytePaddedFields(s, datatype_bufs.buf);

    // write dataspace
    DynamicBufferSerializer dataspace_bufs;
    dataspace_bufs.WriteComplex(dataspace);
    WriteEightBytePaddedFields(s, dataspace_bufs.buf);

    // write data
    s.WriteBuffer(data);
}

void ReadEightBytePaddedData(Deserializer& de, std::span<byte_t> buffer) {
    de.ReadBuffer(buffer);

    size_t leftover = (8 - buffer.size() % 8) % 8;

    static std::array<byte_t, 8> leftover_buf;
    de.ReadBuffer(std::span(leftover_buf.data(), leftover));
}

AttributeMessage AttributeMessage::Deserialize(Deserializer& de) {
    if (de.Read<uint8_t>() != kVersionNumber) {
        throw std::runtime_error("Version number was invalid");
    }
    // reserved (zero)
    de.Skip<uint8_t>();

    auto name_size = de.Read<uint16_t>();
    auto datatype_size = de.Read<uint16_t>();
    auto dataspace_size = de.Read<uint16_t>();

    auto max_buf_size = std::max(std::max(name_size, datatype_size), dataspace_size);
    std::vector<byte_t> buf(max_buf_size);

    AttributeMessage msg{};

    // read name
    ReadEightBytePaddedData(de, std::span(buf.data(), name_size));

    if (buf.at(name_size - 1) != static_cast<byte_t>('\0')) {
        throw std::runtime_error("string read was not null-terminated");
    }

    msg.name = std::string(reinterpret_cast<const char*>(buf.data()), name_size - 1);

    // read datatype
    std::span datatype_buf(buf.data(), datatype_size);
    ReadEightBytePaddedData(de, datatype_buf);

    msg.datatype = BufferDeserializer(datatype_buf).ReadComplex<DatatypeMessage>();

    // read dataspace
    std::span dataspace_buf(buf.data(), dataspace_size);
    ReadEightBytePaddedData(de, dataspace_buf);

    msg.dataspace = BufferDeserializer(dataspace_buf).ReadComplex<DataspaceMessage>();

    size_t data_size = msg.datatype.Size() * msg.dataspace.MaxElements();
    msg.data.resize(data_size);

    de.ReadBuffer(msg.data);

    return msg;
}

void ObjectCommentMessage::Serialize(Serializer& s) const {
    s.WriteBuffer(std::span(
        reinterpret_cast<const byte_t*>(comment.c_str()),
        comment.size() + 1 // null terminator
    ));
}

ObjectCommentMessage ObjectCommentMessage::Deserialize(Deserializer& de) {
    std::vector<byte_t> buf;

    while (true) {
        auto c = de.Read<byte_t>();

        if (c == static_cast<byte_t>('\0')) {
            break;
        }

        buf.push_back(c);
    }

    return {
        .comment = std::string(reinterpret_cast<const char*>(buf.data()), buf.size())
    };
}

void ObjectModificationTimeMessage::Serialize(Serializer& s) const {
    s.Write(kVersionNumber);

    s.Write<uint8_t>(0);
    s.Write<uint8_t>(0);
    s.Write<uint8_t>(0);

    auto seconds_since_epoch = std::chrono::duration_cast<std::chrono::seconds>(
        modification_time.time_since_epoch()
    ).count();

    s.Write(static_cast<uint32_t>(seconds_since_epoch));
}

ObjectModificationTimeMessage ObjectModificationTimeMessage::Deserialize(Deserializer& de) {
    if (de.Read<uint8_t>() != kVersionNumber) {
        throw std::runtime_error("Version number was invalid");
    }

    // reserved (zero)
    de.Skip<3>();

    auto seconds_since_epoch = de.Read<uint32_t>();

    return {
        .modification_time = std::chrono::system_clock::time_point{ std::chrono::seconds{seconds_since_epoch} }
    };
}

void DriverInfoMessage::Serialize(Serializer& s) const {
    s.Write(kVersionNumber);

    // check that driver_id has len kDriverIdSize and is all ascii
    if (driver_id.size() != kDriverIdSize || !std::ranges::all_of(driver_id, [](char c) { return c >= 0 && c <= 127; })) {
        throw std::runtime_error("DriverInfoMessage: driver_id must be exactly eight ASCII characters");
    }

    s.WriteBuffer(std::span(
        reinterpret_cast<const byte_t*>(driver_id.data()),
        driver_id.size()
    ));

    s.Write<uint16_t>(driver_info.size());

    s.WriteBuffer(driver_info);
}

DriverInfoMessage DriverInfoMessage::Deserialize(Deserializer& de) {
    if (de.Read<uint8_t>() != kVersionNumber) {
        throw std::runtime_error("DriverInfoMessage: unsupported version");
    }

    // read 8 bytes, then make a string out of it
    DriverInfoMessage msg{};

    // read id
    std::array<char, kDriverIdSize> id{};
    de.ReadBuffer(std::span(reinterpret_cast<byte_t*>(id.data()), id.size()));
    msg.driver_id = std::string(id.data(), id.size());


    auto driver_info_size = de.Read<uint16_t>();
    msg.driver_info.resize(driver_info_size);
    de.ReadBuffer(msg.driver_info);

    return msg;
}

void AttributeInfoMessage::Serialize(Serializer& s) const {
    s.Write(kVersionNumber);

    std::bitset<2> flags;
    flags.set(0, max_creation_index.has_value());
    flags.set(1, creation_order_btree_addr.has_value());

    s.Write<uint8_t>(flags.to_ulong());

    if (max_creation_index.has_value()) {
        s.Write(*max_creation_index);
    }

    s.Write(fractal_heap_addr);
    s.Write(name_btree_addr);

    if (creation_order_btree_addr.has_value()) {
        s.Write(*creation_order_btree_addr);
    }
}

AttributeInfoMessage AttributeInfoMessage::Deserialize(Deserializer& de) {
    if (de.Read<uint8_t>() != kVersionNumber) {
        throw std::runtime_error("AttributeInfoMessage: unsupported version");
    }

    AttributeInfoMessage msg{};

    auto flags = de.Read<uint8_t>();
    std::bitset<8> flag_bits(flags);

    if (flag_bits.test(0)) {
        msg.max_creation_index = de.Read<uint16_t>();
    } else {
        msg.max_creation_index = std::nullopt;
    }

    msg.fractal_heap_addr = de.Read<offset_t>();
    msg.name_btree_addr = de.Read<offset_t>();

    if (flag_bits.test(1)) {
        msg.creation_order_btree_addr = de.Read<offset_t>();
    } else {
        msg.creation_order_btree_addr = std::nullopt;
    }

    return msg;
}

void FileSpaceInfoMessage::Serialize(Serializer& s) const {
    s.Write(kVersionNumber);
    s.Write(static_cast<uint8_t>(strategy));

    auto persisting_free_space = PersistingFreeSpace();

    s.Write(static_cast<uint8_t>(persisting_free_space));
    s.Write(free_space_threshold);
    s.Write(file_space_page_size);
    s.Write(page_end_metadata_threshold);
    s.Write(eoa);

    if (persisting_free_space) {
        s.Write(*small_managers);
        s.Write(*large_managers);
    }
}

FileSpaceInfoMessage FileSpaceInfoMessage::Deserialize(Deserializer& de) {
    if (de.Read<uint8_t>() != kVersionNumber) {
        throw std::runtime_error("FileSpaceInfoMessage: invalid version");
    }

    FileSpaceInfoMessage msg{};

    auto strategy = de.Read<uint8_t>();

    constexpr uint16_t kStrategyCt = 4;
    if (strategy >= kStrategyCt) {
        throw std::runtime_error("FileSpaceInfoMessage: invalid strategy");
    }

    msg.strategy = static_cast<Strategy>(strategy);

    auto persisting_free_space = de.Read<uint8_t>() != 0;

    msg.free_space_threshold = de.Read<len_t>();
    msg.file_space_page_size = de.Read<uint32_t>();
    msg.page_end_metadata_threshold = de.Read<uint16_t>();
    msg.eoa = de.Read<offset_t>();

    if (persisting_free_space) {
        msg.small_managers = de.Read<std::array<offset_t, 6>>();
        msg.large_managers = de.Read<std::array<offset_t, 6>>();
    }

    return msg;
}

uint16_t ObjectHeaderMessage::MessageType() const {
    auto index = std::visit([]<typename T>(const T&) { return T::kType; }, message);

    if (index != message.index()) {
        throw std::runtime_error("mismatch between variant index and message type");
    }

    return index;
}

void ObjectHeaderMessage::Serialize(Serializer& s) const {
    s.Write(MessageType());
    s.Write(size);

    s.Write<uint8_t>(flags_.to_ulong());

    // FIXME: Serializer::WriteZero<size_t>
    s.Write<uint8_t>(0);
    s.Write<uint8_t>(0);
    s.Write<uint8_t>(0);

    std::visit([&s](const auto& msg) { s.WriteComplex(msg); }, message);
}

ObjectHeaderMessage ObjectHeaderMessage::Deserialize(Deserializer& de) {
    ObjectHeaderMessage msg{};

    auto type = de.Read<uint16_t>();

    constexpr uint16_t kMessageTypeCt = 0x18;
    if (type >= kMessageTypeCt) {
        throw std::runtime_error("Not a valid message type");
    }

    msg.size = de.Read<uint16_t>();
    msg.flags_ = de.Read<uint8_t>();
    de.Skip<3>(); // reserved (0)

    auto start = de.GetPosition();

    switch (type) {
        case NilMessage::kType: {
            // FIXME: this can be optimized
            for (uint16_t i = 0; i < msg.size; ++i) {
                de.Skip<uint8_t>();
            }

            msg.message = NilMessage { .size = msg.size };
            break;
        }
        case DataspaceMessage::kType: {
            msg.message = de.ReadComplex<DataspaceMessage>();
            break;
        }
        case LinkInfoMessage::kType: {
            msg.message = de.ReadComplex<LinkInfoMessage>();
            break;
        }
        case DatatypeMessage::kType: {
            msg.message = de.ReadComplex<DatatypeMessage>();
            break;
        }
        case FillValueOldMessage::kType: {
            msg.message = de.ReadComplex<FillValueOldMessage>();
            break;
        }
        case FillValueMessage::kType: {
            msg.message = de.ReadComplex<FillValueMessage>();
            break;
        }
        case LinkMessage::kType: {
            msg.message = de.ReadComplex<LinkMessage>();
            break;
        }
        case ExternalDataFilesMessage::kType: {
            msg.message = de.ReadComplex<ExternalDataFilesMessage>();
            break;
        }
        case DataLayoutMessage::kType: {
            msg.message = de.ReadComplex<DataLayoutMessage>();
            break;
        }
        case BogusMessage::kType: {
            msg.message = de.ReadComplex<BogusMessage>();
            break;
        }
        case GroupInfoMessage::kType: {
            msg.message = de.ReadComplex<GroupInfoMessage>();
            break;
        }
        case FilterPipelineMessage::kType: {
            msg.message = de.ReadComplex<FilterPipelineMessage>();
            break;
        }
        case AttributeMessage::kType: {
            msg.message = de.ReadComplex<AttributeMessage>();
            break;
        }
        case ObjectCommentMessage::kType: {
            msg.message = de.ReadComplex<ObjectCommentMessage>();
            break;
        }
        case ObjectModificationTimeOldMessage::kType: {
            msg.message = de.ReadComplex<ObjectModificationTimeOldMessage>();
            break;
        }
        case SharedMessageTableMessage::kType: {
            msg.message = de.ReadComplex<SharedMessageTableMessage>();
            break;
        }
        case ObjectHeaderContinuationMessage::kType: {
            msg.message = de.ReadComplex<ObjectHeaderContinuationMessage>();
            break;
        }
        case SymbolTableMessage::kType: {
            msg.message = de.ReadComplex<SymbolTableMessage>();
            break;
        }
        case ObjectModificationTimeMessage::kType: {
            msg.message = de.ReadComplex<ObjectModificationTimeMessage>();
            break;
        }
        case BTreeKValuesMessage::kType: {
            msg.message = de.ReadComplex<BTreeKValuesMessage>();
            break;
        }
        case DriverInfoMessage::kType: {
            msg.message = de.ReadComplex<DriverInfoMessage>();
            break;
        }
        case AttributeInfoMessage::kType: {
            msg.message = de.ReadComplex<AttributeInfoMessage>();
            break;
        }
        case ObjectReferenceCountMessage::kType: {
            msg.message = de.ReadComplex<ObjectReferenceCountMessage>();
            break;
        }
        case FileSpaceInfoMessage::kType: {
            msg.message = de.ReadComplex<FileSpaceInfoMessage>();
            break;
        }
        default: {
            throw std::logic_error("invalid object header message type");
        }
    }

    auto difference = de.GetPosition() - start;

    if (difference > msg.size) {
        throw std::runtime_error("read an incorrect number of bytes!");
    }

    if (msg.size > difference) {
        auto padding_ct = msg.size - difference;

        if (padding_ct >= 8) {
            throw std::runtime_error("shouldn't be more than 8 bytes to pad to 8 bytes");
        }

        std::array<byte_t, 8> padding{};

        de.ReadBuffer(std::span(padding.data(), padding_ct));
    }

    return msg;
}

void ObjectHeader::Serialize(Serializer& s) const {
    s.Write(kVersionNumber);
    s.Write<uint8_t>(0);
    s.Write<uint16_t>(messages.size());
    s.Write(object_ref_count);
    s.Write(object_header_size);
    // reserved (zero)
    s.Write<uint32_t>(0);

    // FIXME: handle object continuation
    for (const ObjectHeaderMessage& msg: messages) {
        s.WriteComplex(msg);
    }
}

void ParseObjectHeaderMessages(ObjectHeader& hd, Deserializer& de, uint32_t size_limit, uint16_t total_message_ct) { // NOLINT(*-no-recursion)
    uint32_t bytes_read = 0;

    while (bytes_read < size_limit && hd.messages.size() < total_message_ct) {
        size_t before_read = de.GetPosition();

        hd.messages.push_back(de.ReadComplex<ObjectHeaderMessage>());

        bytes_read += de.GetPosition() - before_read;

        if (const auto* cont = std::get_if<ObjectHeaderContinuationMessage>(&hd.messages.back().message)) {
            offset_t return_pos = de.GetPosition();

            de.SetPosition(/* TODO: sb.base_addr + */ cont->offset);

            ParseObjectHeaderMessages(hd, de, cont->length, total_message_ct);

            de.SetPosition(return_pos);
        }
    }
}

ObjectHeader ObjectHeader::Deserialize(Deserializer& de) {
    if (de.Read<uint8_t>() != kVersionNumber) {
        throw std::runtime_error("Version number was invalid");
    }
    // reserved (zero)
    de.Skip<uint8_t>();

    auto message_count = de.Read<uint16_t>();

    ObjectHeader hd{};

    hd.object_ref_count = de.Read<uint32_t>();
    hd.object_header_size = de.Read<uint32_t>();
    // reserved (zero)
    de.Skip<uint32_t>();

    ParseObjectHeaderMessages(hd, de, hd.object_header_size, message_count);

    return hd;
}
