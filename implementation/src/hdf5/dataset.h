#pragma once

#include <utility>

#include "file_link.h"
#include "object.h"
#include "object_header.h"
#include "tree.h"
#include "../serialization/serialization.h"

class Dataset {
public:
    static hdf5::expected<Dataset> New(const Object& object);

    // TODO: multidimensional coords
    template <typename T>
    hdf5::expected<T> Get(size_t index) const {
        if (index >= space_.TotalElements()) {
            return hdf5::error(hdf5::HDF5ErrorCode::IndexOutOfBounds, "Index out of bounds for dataset");
        }

        if (type_.class_v == DatatypeMessage::Class::kVariableLength) {
            return hdf5::error(hdf5::HDF5ErrorCode::FeatureNotSupported, "Variable length datatypes are not supported yet");
        }

        const auto* props = cstd::get_if<ContiguousStorageProperty>(&layout_.properties);

        if (!props) {
            return hdf5::error(hdf5::HDF5ErrorCode::FeatureNotSupported, "only contiguous storage currently supported");
        }

        size_t size = type_.Size();

        object_.file->io.SetPosition(object_.file->superblock.base_addr + props->address + index * size);

        return object_.file->io.ReadComplex<T>();
    }

    // FIXME: implement datatype
    hdf5::expected<void> Read(std::span<byte_t> buffer, size_t start_index, size_t count) const;

    // FIXME: impl datatype instead
    template<typename T>
    hdf5::expected<std::vector<T>> Read(size_t start_index, size_t count) const {
        std::vector<T> out(count);

        hdf5::expected<void> result = Read(
            std::span(
                reinterpret_cast<byte_t*>(out.data()),
                out.size() * sizeof(T)
            ),
            start_index,
            count
        );

        if (!result) {
            return cstd::unexpected(result.error());
        }

        return out;
    }

    hdf5::expected<void> ReadHyperslab(
        std::span<byte_t> buffer,
        const hdf5::dim_vector<uint64_t>& start,
        const hdf5::dim_vector<uint64_t>& count,
        const hdf5::dim_vector<uint64_t>& stride = {},
        const hdf5::dim_vector<uint64_t>& block = {}
    ) const;

    template<typename T>
    hdf5::expected<std::vector<T>> ReadHyperslab(
        const hdf5::dim_vector<uint64_t>& start,
        const hdf5::dim_vector<uint64_t>& count,
        const hdf5::dim_vector<uint64_t>& stride = {},
        const hdf5::dim_vector<uint64_t>& block = {}
    ) const {
        size_t total_elements = TotalElements(count, block);

        std::vector<T> result(total_elements);

        hdf5::expected<void> res = ReadHyperslab(
            std::span(
                reinterpret_cast<byte_t*>(result.data()),
                result.size() * sizeof(T)
            ),
            start,
            count,
            stride,
            block
        );

        if (!res) {
            return cstd::unexpected(res.error());
        }

        return result;
    }

    hdf5::expected<void> Write(std::span<const byte_t> data, size_t start_index) const;

    template<typename T>
    hdf5::expected<void> Write(std::span<const T> data, size_t start_index) const {
        return Write(
            std::span(
                reinterpret_cast<const byte_t*>(data.data()),
                data.size_bytes()
            ),
            start_index
        );
    }

    hdf5::expected<void> WriteHyperslab(
        std::span<const byte_t> data,
        const hdf5::dim_vector<uint64_t>& start,
        const hdf5::dim_vector<uint64_t>& count,
        const hdf5::dim_vector<uint64_t>& stride = {},
        const hdf5::dim_vector<uint64_t>& block = {}
    ) const;

    template<typename T>
    hdf5::expected<void> WriteHyperslab(
        std::span<const T> data,
        const hdf5::dim_vector<uint64_t>& start,
        const hdf5::dim_vector<uint64_t>& count,
        const hdf5::dim_vector<uint64_t>& stride = {},
        const hdf5::dim_vector<uint64_t>& block = {}
    ) const {
        return WriteHyperslab(
            std::span(
                reinterpret_cast<const byte_t*>(data.data()),
                data.size_bytes()
            ),
            start,
            count,
            stride,
            block
        );
    }

    [[nodiscard]] hdf5::expected<std::vector<cstd::tuple<ChunkCoordinates, offset_t, len_t>>> RawOffsets() const;

    [[nodiscard]] hdf5::expected<std::vector<cstd::tuple<ChunkCoordinates, offset_t, len_t>>> GetHyperslabChunkRawOffsets(
        const hdf5::dim_vector<uint64_t>& start,
        const hdf5::dim_vector<uint64_t>& count,
        const hdf5::dim_vector<uint64_t>& stride = {},
        const hdf5::dim_vector<uint64_t>& block = {}
    ) const;

private:
    Dataset(Object object, DataLayoutMessage layout, DatatypeMessage type, const DataspaceMessage& space)
        : object_(std::move(object)), layout_(std::move(layout)), type_(std::move(type)), space_(space) {}

    static size_t TotalElements(const hdf5::dim_vector<uint64_t>& count, const hdf5::dim_vector<uint64_t>& block) {
        size_t total_elements = 1;
        for (size_t i = 0; i < count.size(); ++i) {
            size_t effective_block = block.empty() ? 1 : block[i];
            total_elements *= count[i] * effective_block;
        }

        return total_elements;
    }


private:
    Object object_;

    DataLayoutMessage layout_{};
    DatatypeMessage type_{};
    DataspaceMessage space_{};
};
