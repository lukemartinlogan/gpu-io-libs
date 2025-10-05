#include "dataset.h"
#include "hyperslab.h"
#include <unordered_set>
#include <numeric>

hdf5::expected<Dataset> Dataset::New(const Object& object) {
    ObjectHeader header = object.GetHeader();

    bool found_layout = false, found_type = false, found_space = false;
    DataLayoutMessage layout{};
    DatatypeMessage type{};
    DataspaceMessage space{};

    for (const ObjectHeaderMessage& msg: header.messages) {
        if (const auto* layout_ptr = cstd::get_if<DataLayoutMessage>(&msg.message)) {
            layout = *layout_ptr;
            found_layout = true;
        }
        else if (const auto* type_ptr = cstd::get_if<DatatypeMessage>(&msg.message)) {
            type = *type_ptr;
            found_type = true;
        }
        else if (const auto* space_ptr = cstd::get_if<DataspaceMessage>(&msg.message)) {
            space = *space_ptr;
            found_space = true;
        }
    }

    if (!found_layout || !found_type || !found_space) {
        return hdf5::error(hdf5::HDF5ErrorCode::InvalidDataValue, "Dataset header does not contain all required messages");
    }

    return Dataset(object, layout, type, space);
}

hdf5::expected<void> Dataset::Read(std::span<byte_t> buffer, size_t start_index, size_t count) const {
    if (start_index + count > space_.TotalElements()) {
        return hdf5::error(hdf5::HDF5ErrorCode::IndexOutOfBounds, "Index range out of bounds for dataset");
    }

    size_t element_size = type_.Size();
    size_t total_bytes = count * element_size;

    if (buffer.size() < total_bytes) {
        return hdf5::error(hdf5::HDF5ErrorCode::BufferTooSmall, "Buffer too small for requested data");
    }
    if (buffer.size() > total_bytes) {
        return hdf5::error(hdf5::HDF5ErrorCode::BufferTooLarge, "Buffer size exceeds requested data size");
    }

    if (type_.class_v == DatatypeMessage::Class::kVariableLength) {
        return hdf5::error(hdf5::HDF5ErrorCode::FeatureNotSupported, "Variable length datatypes are not supported yet");
    }

    auto props = layout_.properties;

    if (const auto* compact = cstd::get_if<CompactStorageProperty>(&props)) {
        auto start = compact->raw_data.begin() + static_cast<ptrdiff_t>(start_index * element_size);

        if (start + static_cast<ptrdiff_t>(total_bytes) > compact->raw_data.end()) {
            return hdf5::error(hdf5::HDF5ErrorCode::IndexOutOfBounds, "Index range out of bounds for compact storage dataset");
        }

        std::copy_n(
            start,
            total_bytes,
            buffer.data()
        );

    } else if (const auto* contiguous = cstd::get_if<ContiguousStorageProperty>(&props)) {
        if ((start_index + count) * element_size > contiguous->size) {
            return hdf5::error(hdf5::HDF5ErrorCode::IndexOutOfBounds, "Index range out of bounds for contiguous storage dataset");
        }

        object_.file->io.SetPosition(contiguous->address + start_index * element_size);
        object_.file->io.ReadBuffer(std::span(buffer.data(), total_bytes));

    } else if (const auto* chunked = cstd::get_if<ChunkedStorageProperty>(&props)) {
        return hdf5::error(hdf5::HDF5ErrorCode::NotImplemented, "chunked read not implemented yet");
    } else {
        return hdf5::error(hdf5::HDF5ErrorCode::InvalidVariantState, "unknown storage type in dataset");
    }

    return {};
}

hdf5::expected<void> Dataset::Write(std::span<const byte_t> data, size_t start_index) const {
    if (data.size() % type_.Size() != 0) {
        return hdf5::error(hdf5::HDF5ErrorCode::BufferNotAligned, "Buffer size must be a multiple of the datatype size");
    }

    size_t count = data.size() / type_.Size();

    if (start_index + count > space_.TotalElements()) {
        return hdf5::error(hdf5::HDF5ErrorCode::IndexOutOfBounds, "Index range out of bounds for dataset");
    }

    size_t element_size = type_.Size();

    if (type_.class_v == DatatypeMessage::Class::kVariableLength) {
        return hdf5::error(hdf5::HDF5ErrorCode::FeatureNotSupported, "Variable length datatypes are not supported yet");
    }

    auto props = layout_.properties;

    if (auto* compact = cstd::get_if<CompactStorageProperty>(&props)) {
        auto start = compact->raw_data.begin() + static_cast<ptrdiff_t>(start_index * element_size);

        if (start + static_cast<ptrdiff_t>(data.size()) > compact->raw_data.end()) {
            return hdf5::error(hdf5::HDF5ErrorCode::IndexOutOfBounds, "Index range out of bounds for compact storage dataset");
        }

        std::copy_n(
            data.begin(),
            data.size(),
            start
        );

    } else if (const auto* contiguous = cstd::get_if<ContiguousStorageProperty>(&props)) {
        if ((start_index + count) * element_size > contiguous->size) {
            return hdf5::error(hdf5::HDF5ErrorCode::IndexOutOfBounds, "Index range out of bounds for contiguous storage dataset");
        }

        object_.file->io.SetPosition(contiguous->address + start_index * element_size);
        object_.file->io.WriteBuffer(data);

    } else if (const auto* chunked = cstd::get_if<ChunkedStorageProperty>(&props)) {
        return hdf5::error(hdf5::HDF5ErrorCode::NotImplemented, "chunked write not implemented yet");
    } else {
        return hdf5::error(hdf5::HDF5ErrorCode::InvalidVariantState, "unknown storage type in dataset");
    }

    return {};
}

hdf5::expected<std::vector<cstd::tuple<ChunkCoordinates, offset_t, len_t>>> Dataset::RawOffsets() const {
    auto props = layout_.properties;

    if (const auto* compact = cstd::get_if<CompactStorageProperty>(&props)) {
        // For compact storage, return a single entry with zero coordinates matching dataset dimensionality
        ChunkCoordinates coords;
        coords.coords = hdf5::dim_vector<uint64_t>(space_.dimensions.size(), 0);
        return std::vector{ cstd::tuple{coords, offset_t{0}, static_cast<len_t>(compact->raw_data.size())} };

    } else if (const auto* contiguous = cstd::get_if<ContiguousStorageProperty>(&props)) {
        // For contiguous storage, return a single entry with zero coordinates matching dataset dimensionality
        ChunkCoordinates coords;
        coords.coords = hdf5::dim_vector<uint64_t>(space_.dimensions.size(), 0);
        return std::vector{ cstd::tuple{coords, contiguous->address, contiguous->size} };

    } else if (const auto* chunked = cstd::get_if<ChunkedStorageProperty>(&props)) {
        // For chunked storage, use the B-tree to get all chunk offsets
        ChunkedBTree chunked_tree(
            chunked->b_tree_addr,
            object_.file,
            {
                .dimensionality = static_cast<uint8_t>(chunked->dimension_sizes.size()),
                .elem_byte_size = type_.Size(),
            }
        );
        return chunked_tree.Offsets();
    } else {
        return hdf5::error(hdf5::HDF5ErrorCode::InvalidVariantState, "unknown storage type in dataset");
    }
}

template<typename Visitor>
hdf5::expected<void> ProcessChunkedHyperslab(
    const ChunkedStorageProperty* chunked,
    HyperslabIterator& iterator,
    size_t element_size,
    std::shared_ptr<FileLink> file,
    Visitor&& visitor
) {
    ChunkedBTree chunked_tree(
        chunked->b_tree_addr,
        std::move(file),
        {
            .dimensionality = static_cast<uint8_t>(chunked->dimension_sizes.size()),
            .elem_byte_size = element_size,
        }
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

        cstd::optional<offset_t> chunk_file_offset = chunked_tree.GetChunk(chunk_coords);

        // Calculate element file offset if chunk exists
        cstd::optional<offset_t> element_file_offset;
        if (chunk_file_offset.has_value()) {
            element_file_offset = *chunk_file_offset + within_chunk_offset * element_size;
        }

        // Process this element using the provided processor
        auto visitor_result = visitor(element_file_offset, buffer_offset, chunk_coords);
        if (!visitor_result) {
            return cstd::unexpected(visitor_result.error());
        }

        buffer_offset += element_size;
        iterator.Advance();
    }

    return {};
}

hdf5::expected<void> Dataset::ReadHyperslab(
    std::span<byte_t> buffer,
    const hdf5::dim_vector<uint64_t>& start,
    const hdf5::dim_vector<uint64_t>& count,
    const hdf5::dim_vector<uint64_t>& stride,
    const hdf5::dim_vector<uint64_t>& block
) const {
    if (type_.class_v == DatatypeMessage::Class::kVariableLength) {
        return hdf5::error(hdf5::HDF5ErrorCode::FeatureNotSupported, "Variable length datatypes are not supported yet");
    }

    hdf5::dim_vector<uint64_t> dataset_dims(space_.dimensions.size());

    std::ranges::transform(
        space_.dimensions,
       dataset_dims.begin(),
       [](const auto& dim_info) { return dim_info.size; }
    );

    auto iterator_result = HyperslabIterator::New(start, count, stride, block, dataset_dims);
    if (!iterator_result) {
        return cstd::unexpected(iterator_result.error());
    }
    HyperslabIterator iterator = std::move(*iterator_result);

    size_t element_size = type_.Size();
    auto total_elements_result = iterator.GetTotalElements();

    if (!total_elements_result) {
        return cstd::unexpected(total_elements_result.error());
    }

    uint64_t total_elements = *total_elements_result;

    if (buffer.size() < total_elements * element_size) {
        return hdf5::error(hdf5::HDF5ErrorCode::BufferTooSmall, "Buffer too small for hyperslab data");
    }
    if (buffer.size() > total_elements * element_size) {
        return hdf5::error(hdf5::HDF5ErrorCode::BufferTooLarge, "Buffer size exceeds hyperslab data size");
    }

    auto props = layout_.properties;

    if (const auto* compact = cstd::get_if<CompactStorageProperty>(&props)) {
        size_t buffer_offset = 0;

        while (!iterator.IsAtEnd()) {
            auto linear_index_result = iterator.GetLinearIndex();

            if (!linear_index_result) {
                return cstd::unexpected(linear_index_result.error());
            }

            size_t data_offset = *linear_index_result * element_size;

            if (data_offset + element_size > compact->raw_data.size()) {
                return hdf5::error(hdf5::HDF5ErrorCode::SelectionOutOfBounds, "Hyperslab selection exceeds compact storage bounds");
            }

            std::copy_n(
                compact->raw_data.begin() + static_cast<ptrdiff_t>(data_offset),
                element_size,
                buffer.data() + buffer_offset
            );

            buffer_offset += element_size;
            iterator.Advance();
        }

    } else if (const auto* contiguous = cstd::get_if<ContiguousStorageProperty>(&props)) {
        size_t buffer_offset = 0;

        while (!iterator.IsAtEnd()) {
            auto linear_index = iterator.GetLinearIndex();

            if (!linear_index) {
                return cstd::unexpected(linear_index.error());
            }

            offset_t file_offset = contiguous->address + *linear_index * element_size;

            object_.file->io.SetPosition(file_offset);
            object_.file->io.ReadBuffer(std::span(buffer.data() + buffer_offset, element_size));

            buffer_offset += element_size;
            iterator.Advance();
        }

    } else if (const auto* chunked = cstd::get_if<ChunkedStorageProperty>(&props)) {
        auto process_result = ProcessChunkedHyperslab(
            chunked, iterator, element_size, object_.file,
            [&](const cstd::optional<offset_t>& element_file_offset, size_t buffer_offset, const ChunkCoordinates& /* chunk_coords */) -> hdf5::expected<void> {
                if (!element_file_offset.has_value()) {
                    // chunk doesn't exist (sparse dataset)
                    std::fill_n(buffer.data() + buffer_offset, element_size, byte_t{0});
                } else {
                    object_.file->io.SetPosition(*element_file_offset);
                    object_.file->io.ReadBuffer(std::span(buffer.data() + buffer_offset, element_size));
                }
                return {};
            });
        if (!process_result) {
            return cstd::unexpected(process_result.error());
        }
    } else {
        return hdf5::error(hdf5::HDF5ErrorCode::InvalidVariantState, "Unknown storage type in dataset");
    }

    return {};
}

hdf5::expected<void> Dataset::WriteHyperslab(
    std::span<const byte_t> data,
    const hdf5::dim_vector<uint64_t>& start,
    const hdf5::dim_vector<uint64_t>& count,
    const hdf5::dim_vector<uint64_t>& stride,
    const hdf5::dim_vector<uint64_t>& block
) const {
    if (type_.class_v == DatatypeMessage::Class::kVariableLength) {
        return hdf5::error(hdf5::HDF5ErrorCode::FeatureNotSupported, "Variable length datatypes are not supported yet");
    }

    hdf5::dim_vector<uint64_t> dataset_dims(space_.dimensions.size());

    std::ranges::transform(
        space_.dimensions,
       dataset_dims.begin(),
       [](const auto& dim_info) { return dim_info.size; }
    );

    auto iterator_result = HyperslabIterator::New(start, count, stride, block, dataset_dims);
    if (!iterator_result) {
        return cstd::unexpected(iterator_result.error());
    }
    HyperslabIterator iterator = std::move(*iterator_result);

    size_t element_size = type_.Size();
    auto total_elements_result = iterator.GetTotalElements();

    if (!total_elements_result) {
        return cstd::unexpected(total_elements_result.error());
    }

    uint64_t total_elements = *total_elements_result;

    if (data.size() < total_elements * element_size) {
        return hdf5::error(hdf5::HDF5ErrorCode::BufferTooSmall, "Data buffer too small for hyperslab");
    }
    if (data.size() > total_elements * element_size) {
        return hdf5::error(hdf5::HDF5ErrorCode::BufferTooLarge, "Data buffer size exceeds hyperslab size");
    }

    auto props = layout_.properties;

    if (const auto* compact = cstd::get_if<CompactStorageProperty>(&props)) {
        // Compact storage is read-only after creation, so writing is not supported
        return hdf5::error(hdf5::HDF5ErrorCode::FeatureNotSupported, "Cannot write to compact storage dataset");

    } else if (const auto* contiguous = cstd::get_if<ContiguousStorageProperty>(&props)) {
        size_t data_offset = 0;

        while (!iterator.IsAtEnd()) {
            auto linear_index = iterator.GetLinearIndex();

            if (!linear_index) {
                return cstd::unexpected(linear_index.error());
            }

            offset_t file_offset = contiguous->address + *linear_index * element_size;

            object_.file->io.SetPosition(file_offset);
            object_.file->io.WriteBuffer(std::span(data.data() + data_offset, element_size));

            data_offset += element_size;
            iterator.Advance();
        }

    } else if (const auto* chunked = cstd::get_if<ChunkedStorageProperty>(&props)) {
        auto process_result = ProcessChunkedHyperslab(
            chunked, iterator, element_size, object_.file,
            [&](const cstd::optional<offset_t>& element_file_offset, size_t buffer_offset, const ChunkCoordinates& /* chunk_coords */) -> hdf5::expected<void> {
                if (!element_file_offset.has_value()) {
                    // Chunk doesn't exist (sparse dataset) - this is an error for writing
                    // In HDF5, writing to a non-existent chunk should create the chunk,
                    // but our current implementation doesn't support chunk creation
                    return hdf5::error(hdf5::HDF5ErrorCode::NotImplemented, "Cannot write to non-existent chunk (chunk creation not implemented)");
                }

                object_.file->io.SetPosition(*element_file_offset);
                object_.file->io.WriteBuffer(std::span(data.data() + buffer_offset, element_size));
                return {};
            });
        if (!process_result) {
            return cstd::unexpected(process_result.error());
        }
    } else {
        return hdf5::error(hdf5::HDF5ErrorCode::InvalidVariantState, "Unknown storage type in dataset");
    }

    return {};
}

hdf5::expected<std::vector<cstd::tuple<ChunkCoordinates, offset_t, len_t>>> Dataset::GetHyperslabChunkRawOffsets(
    const hdf5::dim_vector<uint64_t>& start,
    const hdf5::dim_vector<uint64_t>& count,
    const hdf5::dim_vector<uint64_t>& stride,
    const hdf5::dim_vector<uint64_t>& block
) const {
    std::vector<cstd::tuple<ChunkCoordinates, offset_t, len_t>> result;

    if (!cstd::holds_alternative<ChunkedStorageProperty>(layout_.properties)) {
        return RawOffsets();
    }

    auto chunked = cstd::get<ChunkedStorageProperty>(layout_.properties);

    size_t dimensionality = chunked.dimension_sizes.size();

    ChunkedBTree chunked_tree(chunked.b_tree_addr, object_.file, {
        .dimensionality = static_cast<uint8_t>(dimensionality),
        .elem_byte_size = type_.Size(),
    });

    const len_t chunk_size_bytes = std::accumulate(
        chunked.dimension_sizes.begin(),
        chunked.dimension_sizes.end(),
        type_.Size(),
        std::multiplies{}
    );

    hdf5::dim_vector<std::vector<uint64_t>> chunks_per_dim(dimensionality);
    
    for (size_t dim = 0; dim < dimensionality; ++dim) {
        uint64_t chunk_size = chunked.dimension_sizes[dim];
        uint64_t dim_start = start[dim];
        uint64_t dim_count = count[dim];
        uint64_t dim_stride = stride.empty() ? 1 : stride[dim];
        uint64_t dim_block = block.empty() ? 1 : block[dim];
        
        std::unordered_set<uint64_t> unique_chunks;
        unique_chunks.reserve(dim_count * 2); // avoid hash table rehashing
        
        // for each count iteration
        for (uint64_t count_idx = 0; count_idx < dim_count; ++count_idx) {
            uint64_t block_start = dim_start + count_idx * dim_stride;
            uint64_t block_end = block_start + dim_block - 1;
            
            // find all chunks touched by this block
            uint64_t start_chunk = (block_start / chunk_size) * chunk_size;
            uint64_t end_chunk = (block_end / chunk_size) * chunk_size;
            
            for (uint64_t chunk_coord = start_chunk; chunk_coord <= end_chunk; chunk_coord += chunk_size) {
                unique_chunks.insert(chunk_coord);
            }
        }
        
        if (unique_chunks.empty()) {
            return result;
        }
        
        // convert to sorted vector for cartesian products
        auto& dim_chunks = chunks_per_dim[dim];
        dim_chunks.reserve(unique_chunks.size());
        dim_chunks.assign(unique_chunks.begin(), unique_chunks.end());
        std::ranges::sort(dim_chunks); // sort for consistent ordering
    }

    hdf5::dim_vector<uint64_t> current_combination(dimensionality);
    hdf5::dim_vector<size_t> indices(dimensionality, 0);
    
    for (;;) {
        // get current combination
        for (size_t dim = 0; dim < dimensionality; ++dim) {
            current_combination[dim] = chunks_per_dim[dim][indices[dim]];
        }
        
        ChunkCoordinates chunk_coords(current_combination);
        cstd::optional<offset_t> chunk_file_offset = chunked_tree.GetChunk(chunk_coords);
        
        if (chunk_file_offset.has_value()) {
            result.emplace_back(std::move(chunk_coords), *chunk_file_offset, chunk_size_bytes);
        }
        
        // find first dimension that can increment
        size_t dim = 0;
        while (dim < dimensionality && ++indices[dim] >= chunks_per_dim[dim].size()) {
            indices[dim] = 0;
            ++dim;
        }
        if (dim == dimensionality) {
            break;
        }
    }

    return result;
}