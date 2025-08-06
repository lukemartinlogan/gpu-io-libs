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

    s.Write(static_cast<uint8_t>(bitset_.to_ulong()) & 0b11);

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
    s.Write(static_cast<uint8_t>(dimension_sizes.size()));

    s.Write(b_tree_addr);

    s.WriteBuffer(std::span(
        reinterpret_cast<const byte_t*>(dimension_sizes.data()),
        dimension_sizes.size() * sizeof(uint32_t)
    ));

    s.Write(elem_size_bytes);
}

ChunkedStorageProperty ChunkedStorageProperty::Deserialize(Deserializer& de) {
    auto dimensionality = de.Read<uint8_t>();

    ChunkedStorageProperty prop{};

    prop.b_tree_addr = de.Read<offset_t>();

    prop.dimension_sizes.resize(dimensionality);

    de.ReadBuffer(std::span(
        reinterpret_cast<byte_t*>(prop.dimension_sizes.data()),
        prop.dimension_sizes.size() * sizeof(uint32_t)
    ));

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
    BufferDeserializer datatype_buf_de(std::span(buf.data(), datatype_size));
    ReadEightBytePaddedData(de, datatype_buf_de.buf);

    msg.datatype = datatype_buf_de.ReadComplex<DatatypeMessage>();

    // read dataspace
    BufferDeserializer dataspace_buf_de(std::span(buf.data(), dataspace_size));
    ReadEightBytePaddedData(de, dataspace_buf_de.buf);

    msg.dataspace = dataspace_buf_de.ReadComplex<DataspaceMessage>();

    size_t data_size = msg.datatype.Size() * msg.dataspace.MaxElements();
    msg.data.resize(data_size);

    de.ReadBuffer(msg.data);

    return msg;
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

void ObjectHeaderMessage::Serialize(Serializer& s) const {
    s.Write(type);
    s.Write<uint16_t>(0); // FIXME: object header size!

    // FIXME: Serializer::WriteZero<size_t>
    s.Write<uint8_t>(0);
    s.Write<uint8_t>(0);
    s.Write<uint8_t>(0);

    s.Write(flags);

    switch (type) {
        case Type::kNil: {
            s.Write(std::get<NilMessage>(message));
        }
        case Type::kDataspace: {
            s.Write(std::get<DataspaceMessage>(message));
            break;
        }
        case Type::kLinkInfo: {
            s.Write(std::get<LinkInfoMessage>(message));
            break;
        }
        case Type::kDatatype: {
            s.Write(std::get<DatatypeMessage>(message));
            break;
        }
        case Type::kFillValueOld: {
            s.Write(std::get<FillValueOldMessage>(message));
            break;
        }
        case Type::kFillValue: {
            s.Write(std::get<FillValueMessage>(message));
            break;
        }
        case Type::kLink: {
            s.Write(std::get<LinkMessage>(message));
            break;
        }
        case Type::kDataLayout: {
            s.Write(std::get<DataLayoutMessage>(message));
            break;
        }
        case Type::kAttribute: {
            s.Write(std::get<AttributeMessage>(message));
            break;
        }
        case Type::kObjectHeaderContinuation: {
            s.Write(std::get<ObjectHeaderContinuationMessage>(message));
            break;
        }
        case Type::kSymbolTable: {
            s.Write(std::get<SymbolTableMessage>(message));
            break;
        }
        case Type::kObjectModificationTime: {
            s.Write(std::get<ObjectModificationTimeMessage>(message));
            break;
        }
        default: {
            throw std::logic_error("object header ty not implemented");
        }
    }
}

ObjectHeaderMessage ObjectHeaderMessage::Deserialize(Deserializer& de) {
    ObjectHeaderMessage msg{};

    auto type = de.Read<uint16_t>();

    constexpr uint16_t kMessageTypeCt = 0x18;
    if (type >= kMessageTypeCt) {
        throw std::runtime_error("Not a valid message type");
    }

    msg.type = static_cast<Type>(type);

    auto size = de.Read<uint16_t>();
    msg.flags = de.Read<uint8_t>();
    de.Skip<3>(); // reserved (0)

    auto start = de.GetPosition();

    switch (msg.type) {
        case Type::kNil: {
            // FIXME: this can be optimized
            for (uint16_t i = 0; i < size; ++i) {
                de.Skip<uint8_t>();
            }

            msg.message = NilMessage { .size = size };
            break;
        }
        case Type::kDataspace: {
            msg.message = de.ReadComplex<DataspaceMessage>();
            break;
        }
        case Type::kLinkInfo: {
            msg.message = de.ReadComplex<LinkInfoMessage>();
            break;
        }
        case Type::kDatatype: {
            msg.message = de.ReadComplex<DatatypeMessage>();
            break;
        }
        case Type::kFillValueOld: {
            msg.message = de.ReadComplex<FillValueOldMessage>();
            break;
        }
        case Type::kFillValue: {
            msg.message = de.ReadComplex<FillValueMessage>();
            break;
        }
        case Type::kLink: {
            msg.message = de.ReadComplex<LinkMessage>();
            break;
        }
        case Type::kDataLayout: {
            msg.message = de.ReadComplex<DataLayoutMessage>();
            break;
        }
        case Type::kAttribute: {
            msg.message = de.ReadComplex<AttributeMessage>();
            break;
        }
        case Type::kObjectHeaderContinuation: {
            msg.message = de.ReadComplex<ObjectHeaderContinuationMessage>();
            break;
        }
        case Type::kSymbolTable: {
            msg.message = de.ReadComplex<SymbolTableMessage>();
            break;
        }
        case Type::kObjectModificationTime: {
            msg.message = de.ReadComplex<ObjectModificationTimeMessage>();
            break;
        }
        default: {
            throw std::logic_error("object header ty not implemented");
        }
    }

    auto difference = de.GetPosition() - start;

    if (difference > size) {
        throw std::runtime_error("read an incorrect number of bytes!");
    }

    if (size > difference) {
        auto padding_ct = size - difference;

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
    s.Write(message_count);
    s.Write(object_ref_count);
    s.Write(object_header_size);
    // reserved (zero)
    s.Write<uint32_t>(0);

    for (const ObjectHeaderMessage& msg: messages) {
        s.WriteComplex(msg);
    }
}

ObjectHeader ObjectHeader::Deserialize(Deserializer& de) {
    if (de.Read<uint8_t>() != kVersionNumber) {
        throw std::runtime_error("Version number was invalid");
    }
    // reserved (zero)
    de.Skip<uint8_t>();

    ObjectHeader hd{};

    hd.message_count = de.Read<uint16_t>();
    hd.object_ref_count = de.Read<uint32_t>();
    hd.object_header_size = de.Read<uint32_t>();
    // reserved (zero)
    de.Skip<uint32_t>();

    for (uint16_t m = 0; m < hd.message_count; ++m) {
        hd.messages.push_back(de.ReadComplex<ObjectHeaderMessage>());

        if (const auto* p = std::get_if<ObjectHeaderContinuationMessage>(&hd.messages.back().message)) {
            de.SetPosition(/* TODO: sb.base_addr + */ p->offset);
            // TODO: don't read over size
        }
    }

    return hd;
}
