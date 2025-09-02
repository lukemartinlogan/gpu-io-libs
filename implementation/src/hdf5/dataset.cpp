#include "dataset.h"

Dataset::Dataset(const Object& object)
    : object_(object)
{
    ObjectHeader header = object_.GetHeader();

    bool found_layout = false, found_type = false, found_space = false;

    for (const ObjectHeaderMessage& msg: header.messages) {
        if (const auto* layout = std::get_if<DataLayoutMessage>(&msg.message)) {
            layout_ = *layout;
            found_layout = true;
        }
        else if (const auto* type = std::get_if<DatatypeMessage>(&msg.message)) {
            type_ = *type;
            found_type = true;
        }
        else if (const auto* space = std::get_if<DataspaceMessage>(&msg.message)) {
            space_ = *space;
            found_space = true;
        }
    }

    if (!found_layout || !found_type || !found_space) {
        throw std::runtime_error("Dataset header does not contain all required messages");
    }
}

void Dataset::Read(std::span<byte_t> buffer, size_t start_index, size_t count) const {
    if (start_index + count > space_.TotalElements()) {
        throw std::out_of_range("Index range out of bounds for dataset");
    }

    size_t element_size = type_.Size();
    size_t total_bytes = count * element_size;

    if (buffer.size() < total_bytes) {
        throw std::invalid_argument("Buffer too small for requested data");
    }
    if (buffer.size() > total_bytes) {
        throw std::invalid_argument("Buffer size exceeds requested data size");
    }

    if (type_.class_v == DatatypeMessage::Class::kVariableLength) {
        throw std::logic_error("Variable length datatypes are not supported yet");
    }

    auto props = layout_.properties;

    if (const auto* compact = std::get_if<CompactStorageProperty>(&props)) {
        auto start = compact->raw_data.begin() + static_cast<ptrdiff_t>(start_index * element_size);

        if (start + static_cast<ptrdiff_t>(total_bytes) > compact->raw_data.end()) {
            throw std::out_of_range("Index range out of bounds for compact storage dataset");
        }

        std::copy_n(
            start,
            total_bytes,
            buffer.data()
        );

    } else if (const auto* contiguous = std::get_if<ContiguousStorageProperty>(&props)) {
        if ((start_index + count) * element_size > contiguous->size) {
            throw std::out_of_range("Index range out of bounds for contiguous storage dataset");
        }

        object_.file->io.SetPosition(contiguous->address + start_index * element_size);
        object_.file->io.ReadBuffer(std::span(buffer.data(), total_bytes));

    } else if (const auto* chunked = std::get_if<ChunkedStorageProperty>(&props)) {
        throw std::logic_error("chunked read not implemented yet");
    } else {
        throw std::logic_error("unknown storage type in dataset");
    }
}

void Dataset::Write(std::span<const byte_t> data, size_t start_index) const {
    if (data.size() % type_.Size() != 0) {
        throw std::invalid_argument("Buffer size must be a multiple of the datatype size");
    }

    size_t count = data.size() / type_.Size();

    if (start_index + count > space_.TotalElements()) {
        throw std::out_of_range("Index range out of bounds for dataset");
    }

    size_t element_size = type_.Size();

    if (type_.class_v == DatatypeMessage::Class::kVariableLength) {
        throw std::logic_error("Variable length datatypes are not supported yet");
    }

    auto props = layout_.properties;

    if (auto* compact = std::get_if<CompactStorageProperty>(&props)) {
        auto start = compact->raw_data.begin() + static_cast<ptrdiff_t>(start_index * element_size);

        if (start + static_cast<ptrdiff_t>(data.size()) > compact->raw_data.end()) {
            throw std::out_of_range("Index range out of bounds for compact storage dataset");
        }

        std::copy_n(
            data.begin(),
            data.size(),
            start
        );

    } else if (const auto* contiguous = std::get_if<ContiguousStorageProperty>(&props)) {
        if ((start_index + count) * element_size > contiguous->size) {
            throw std::out_of_range("Index range out of bounds for contiguous storage dataset");
        }

        object_.file->io.SetPosition(contiguous->address + start_index * element_size);
        object_.file->io.WriteBuffer(data);

    } else if (const auto* chunked = std::get_if<ChunkedStorageProperty>(&props)) {
        throw std::logic_error("chunked write not implemented yet");
    } else {
        throw std::logic_error("unknown storage type in dataset");
    }
}

std::vector<std::tuple<ChunkCoordinates, offset_t, len_t>> Dataset::RawOffsets() const {
    auto props = layout_.properties;

    if (const auto* compact = std::get_if<CompactStorageProperty>(&props)) {
        // For compact storage, return a single entry with zero coordinates matching dataset dimensionality
        ChunkCoordinates coords;
        coords.coords = std::vector<uint64_t>(space_.dimensions.size(), 0);
        return { {coords, 0, static_cast<len_t>(compact->raw_data.size())} };

    } else if (const auto* contiguous = std::get_if<ContiguousStorageProperty>(&props)) {
        // For contiguous storage, return a single entry with zero coordinates matching dataset dimensionality
        ChunkCoordinates coords;
        coords.coords = std::vector<uint64_t>(space_.dimensions.size(), 0);
        return { {coords, contiguous->address, contiguous->size} };

    } else if (const auto* chunked = std::get_if<ChunkedStorageProperty>(&props)) {
        // For chunked storage, use the B-tree to get all chunk offsets
        ChunkedBTree chunked_tree(
            chunked->b_tree_addr,
            object_.file,
            static_cast<uint8_t>(chunked->dimension_sizes.size())
        );
        return chunked_tree.Offsets();
    } else {
        throw std::logic_error("unknown storage type in dataset");
    }
}