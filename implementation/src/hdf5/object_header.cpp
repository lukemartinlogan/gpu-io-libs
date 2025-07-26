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

    for (uint8_t d = 0; d < msg.dimensions.size(); ++d) {
        msg.dimensions.at(d).size = de.Read<len_t>();
    }

    for (uint8_t d = 0; d < msg.dimensions.size(); ++d) {
        msg.dimensions.at(d).max_size = de.Read<len_t>();
    }

    if (msg.PermutationIndicesPresent()) {
        for (uint8_t d = 0; d < msg.dimensions.size(); ++d) {
            msg.dimensions.at(d).permutation_index = de.Read<len_t>();
        }
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

void ObjectHeaderMessage::Serialize(Serializer& s) const {
    s.Write(type);
    s.Write(InternalSize());

    // FIXME: Serializer::WriteZero<size_t>
    s.Write<uint8_t>(0);
    s.Write<uint8_t>(0);
    s.Write<uint8_t>(0);

    s.Write(flags);

    switch (type) {
        case Type::kDatatype: {
            s.Write(std::get<DatatypeMessage>(message));
            break;
        }
        case Type::kDataspace: {
            s.Write(std::get<DataspaceMessage>(message));
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
        default: {
            throw std::logic_error("object header ty not implemented");
        }
    }
}

ObjectHeaderMessage ObjectHeaderMessage::Deserialize(Deserializer& de) {
    ObjectHeaderMessage msg{};

    uint16_t type = de.Read<uint16_t>();

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
        case Type::kDatatype: {
            msg.message = de.ReadComplex<DatatypeMessage>();
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

    // FIXME(datatype-impl): readd this check once datatypes are properly parsed
    // uint64_t total_bytes = std::accumulate(
    //     hd.messages.begin(),
    //     hd.messages.end(),
    //     static_cast<uint64_t>(0),
    //     [](uint64_t acc, const ObjectHeaderMessage& msg) {
    //         return acc + msg.InternalSize();
    //     }
    // );
    //
    // if (total_bytes != hd.object_header_size) {
    //     throw std::runtime_error("Failed to read correct number of header bytes");
    // }

    return hd;
}
