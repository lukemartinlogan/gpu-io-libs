#include "dataset.h"
#include "hyperslab.h"

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

template<typename Visitor>
void ProcessChunkedHyperslab(
    const ChunkedStorageProperty* chunked,
    HyperslabIterator& iterator,
    size_t element_size,
    std::shared_ptr<FileLink> file,
    Visitor&& visitor
) {
    ChunkedBTree chunked_tree(
        chunked->b_tree_addr,
        std::move(file),
        static_cast<uint8_t>(chunked->dimension_sizes.size())
    );

    size_t buffer_offset = 0;

    while (!iterator.IsAtEnd()) {
        const auto& current_coord = iterator.GetCurrentCoordinate();

        ChunkCoordinates chunk_coords;
        chunk_coords.coords.resize(chunked->dimension_sizes.size());

        uint64_t within_chunk_offset = 0;
        uint64_t chunk_stride = 1;

        // process dimensions in reverse order
        for (int dim = static_cast<int>(chunked->dimension_sizes.size()) - 1; dim >= 0; --dim) {
            uint64_t chunk_size = chunked->dimension_sizes[dim];
            uint64_t coord = current_coord[dim];

            uint64_t chunk_offset = (coord / chunk_size) * chunk_size;
            uint64_t within_chunk = coord % chunk_size;

            // calculate linear offset
            within_chunk_offset += within_chunk * chunk_stride;
            chunk_stride *= chunk_size;

            chunk_coords.coords[dim] = chunk_offset;
        }

        std::optional<offset_t> chunk_file_offset = chunked_tree.GetChunk(chunk_coords);

        // Calculate element file offset if chunk exists
        std::optional<offset_t> element_file_offset;
        if (chunk_file_offset.has_value()) {
            element_file_offset = *chunk_file_offset + within_chunk_offset * element_size;
        }

        // Process this element using the provided processor
        visitor(element_file_offset, buffer_offset);

        buffer_offset += element_size;
        iterator.Advance();
    }
}

void Dataset::ReadHyperslab(
    std::span<byte_t> buffer,
    const std::vector<uint64_t>& start,
    const std::vector<uint64_t>& count,
    const std::vector<uint64_t>& stride,
    const std::vector<uint64_t>& block
) const {
    if (type_.class_v == DatatypeMessage::Class::kVariableLength) {
        throw std::logic_error("Variable length datatypes are not supported yet");
    }

    std::vector<uint64_t> dataset_dims(space_.dimensions.size());

    std::ranges::transform(
        space_.dimensions,
       dataset_dims.begin(),
       [](const auto& dim_info) { return dim_info.size; }
    );

    HyperslabIterator iterator(start, count, stride, block, dataset_dims);

    size_t element_size = type_.Size();
    uint64_t total_elements = iterator.GetTotalElements();

    if (buffer.size() < total_elements * element_size) {
        throw std::invalid_argument("Buffer too small for hyperslab data");
    }
    if (buffer.size() > total_elements * element_size) {
        throw std::invalid_argument("Buffer size exceeds hyperslab data size");
    }

    auto props = layout_.properties;

    if (const auto* compact = std::get_if<CompactStorageProperty>(&props)) {
        size_t buffer_offset = 0;

        while (!iterator.IsAtEnd()) {
            uint64_t linear_index = iterator.GetLinearIndex();
            size_t data_offset = linear_index * element_size;

            if (data_offset + element_size > compact->raw_data.size()) {
                throw std::out_of_range("Hyperslab selection exceeds compact storage bounds");
            }

            std::copy_n(
                compact->raw_data.begin() + static_cast<ptrdiff_t>(data_offset),
                element_size,
                buffer.data() + buffer_offset
            );

            buffer_offset += element_size;
            iterator.Advance();
        }

    } else if (const auto* contiguous = std::get_if<ContiguousStorageProperty>(&props)) {
        size_t buffer_offset = 0;

        while (!iterator.IsAtEnd()) {
            uint64_t linear_index = iterator.GetLinearIndex();
            offset_t file_offset = contiguous->address + linear_index * element_size;

            object_.file->io.SetPosition(file_offset);
            object_.file->io.ReadBuffer(std::span(buffer.data() + buffer_offset, element_size));

            buffer_offset += element_size;
            iterator.Advance();
        }

    } else if (const auto* chunked = std::get_if<ChunkedStorageProperty>(&props)) {
        ProcessChunkedHyperslab(
            chunked, iterator, element_size, object_.file,
            [&](const std::optional<offset_t>& element_file_offset, size_t buffer_offset) {
                if (!element_file_offset.has_value()) {
                    // chunk doesn't exist (sparse dataset)
                    std::fill_n(buffer.data() + buffer_offset, element_size, byte_t{0});
                } else {
                    object_.file->io.SetPosition(*element_file_offset);
                    object_.file->io.ReadBuffer(std::span(buffer.data() + buffer_offset, element_size));
                }
            });
    } else {
        throw std::logic_error("Unknown storage type in dataset");
    }
}