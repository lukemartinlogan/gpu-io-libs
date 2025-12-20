#pragma once

#include "types.h"
#include "file_link.h"
#include "object.h"
#include "object_header.h"
#include "tree.h"
#include "../serialization/serialization.h"

class Dataset {
public:
    __device__
    static hdf5::expected<Dataset> New(const Object& object);

    // TODO: multidimensional coords
    template <typename T>
    __device__ hdf5::expected<T> Get(size_t index) const;

    // FIXME: implement datatype
    __device__
    hdf5::expected<void> Read(cstd::span<byte_t> buffer, size_t start_index, size_t count) const;

    __device__
    hdf5::expected<void> ReadHyperslab(
        cstd::span<byte_t> buffer,
        const hdf5::dim_vector<uint64_t>& start,
        const hdf5::dim_vector<uint64_t>& count,
        const hdf5::dim_vector<uint64_t>& stride = {},
        const hdf5::dim_vector<uint64_t>& block = {}
    ) const;

    __device__
    hdf5::expected<void> Write(cstd::span<const byte_t> data, size_t start_index) const;

    template<typename T>
    __device__ hdf5::expected<void> Write(cstd::span<const T> data, size_t start_index) const;

    __device__
    hdf5::expected<void> WriteHyperslab(
        cstd::span<const byte_t> data,
        const hdf5::dim_vector<uint64_t>& start,
        const hdf5::dim_vector<uint64_t>& count,
        const hdf5::dim_vector<uint64_t>& stride = {},
        const hdf5::dim_vector<uint64_t>& block = {}
    ) const;

    template<typename T>
    __device__ hdf5::expected<void> WriteHyperslab(
        cstd::span<const T> data,
        const hdf5::dim_vector<uint64_t>& start,
        const hdf5::dim_vector<uint64_t>& count,
        const hdf5::dim_vector<uint64_t>& stride = {},
        const hdf5::dim_vector<uint64_t>& block = {}
    ) const {
        return WriteHyperslab(
            cstd::span(
                reinterpret_cast<const byte_t*>(data.data()),
                data.size_bytes()
            ),
            start,
            count,
            stride,
            block
        );
    }

private:
    __device__
    Dataset(Object object, DataLayoutMessage layout, DatatypeMessage type, const DataspaceMessage& space)
        : object_(std::move(object)), layout_(std::move(layout)), type_(std::move(type)), space_(space) {}

    __device__
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

template <typename T>
__device__
inline hdf5::expected<T> Dataset::Get(size_t index) const {
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

    auto io = object_.file->MakeRW();
    io.SetPosition(object_.file->superblock.base_addr + props->address + index * size);

    return serde::Read<T>(io);
}

template<typename T>
__device__
inline hdf5::expected<void> Dataset::Write(cstd::span<const T> data, size_t start_index) const {
    return Write(
        cstd::span(
            reinterpret_cast<const byte_t*>(data.data()),
            data.size_bytes()
        ),
        start_index
    );
}
