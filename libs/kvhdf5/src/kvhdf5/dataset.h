#pragma once

#include "defines.h"
#include "types.h"
#include "datatype.h"
#include "allocator.h"
#include "context.h"
#include "error.h"
#include <utils/gpu_string.h>
#include <cuda/std/span>
#include <cuda/std/inplace_vector>
#include <cuda/std/initializer_list>

namespace kvhdf5 {

inline constexpr uint8_t MAX_DATASET_DIMS = 8;

struct DatasetShape {
    cstd::array<uint64_t, MAX_DATASET_DIMS> dims;
    cstd::array<uint64_t, MAX_DATASET_DIMS> chunk_dims;
    uint8_t ndims_;
    // 7 bytes padding to 8-byte boundary

    static expected<DatasetShape> Create(
        cstd::initializer_list<uint64_t> dims_list,
        cstd::initializer_list<uint64_t> chunk_dims_list
    ) {
        if (dims_list.size() != chunk_dims_list.size()) {
            return make_error(ErrorCode::InvalidArgument, "dims and chunk_dims must have same size");
        }

        if (dims_list.size() == 0 || dims_list.size() > MAX_DATASET_DIMS) {
            return make_error(ErrorCode::InvalidArgument, "ndims must be between 1 and MAX_DATASET_DIMS");
        }

        DatasetShape shape;
        shape.ndims_ = static_cast<uint8_t>(dims_list.size());

        size_t i = 0;
        auto dims_it = dims_list.begin();
        auto chunks_it = chunk_dims_list.begin();
        for (; i < shape.ndims_; ++i, ++dims_it, ++chunks_it) {
            uint64_t dim = *dims_it;
            uint64_t chunk = *chunks_it;

            if (chunk == 0) {
                return make_error(ErrorCode::InvalidArgument, "chunk_dims must be > 0");
            }
            if (chunk > dim) {
                return make_error(ErrorCode::InvalidArgument, "chunk_dims must be <= dims");
            }

            shape.dims[i] = dim;
            shape.chunk_dims[i] = chunk;
        }

        // Zero out unused dimensions
        for (; i < MAX_DATASET_DIMS; ++i) {
            shape.dims[i] = 0;
            shape.chunk_dims[i] = 0;
        }

        return shape;
    }

    CROSS_FUN uint8_t Ndims() const { return ndims_; }

    CROSS_FUN cstd::span<const uint64_t> Dims() const {
        return cstd::span(dims.data(), ndims_);
    }

    CROSS_FUN cstd::span<uint64_t> Dims() {
        return cstd::span(dims.data(), ndims_);
    }

    CROSS_FUN cstd::span<const uint64_t> ChunkDims() const {
        return cstd::span(chunk_dims.data(), ndims_);
    }

    CROSS_FUN cstd::span<uint64_t> ChunkDims() {
        return cstd::span(chunk_dims.data(), ndims_);
    }

    CROSS_FUN bool IsSingleChunk() const {
        return cstd::equal(Dims().begin(), Dims().end(), ChunkDims().begin());
    }

    CROSS_FUN constexpr bool operator==(const DatasetShape&) const = default;
};

struct Attribute {
    static constexpr size_t kMaxValueSize = 128;

    gpu_string<255> name;
    DatatypeRef datatype;
    // For Phase 1: store small inline values (up to kMaxValueSize bytes)
    // Future: use CTE blob storage for large attributes
    cstd::inplace_vector<byte_t, kMaxValueSize> value;
    
    CROSS_FUN Attribute(gpu_string_view n, DatatypeRef dt, cstd::span<const byte_t> val)
        : name(n), datatype(dt), value()
    {
        KVHDF5_ASSERT(val.size() <= kMaxValueSize, "Attribute value too large (max 128 bytes)");
        value.insert(value.end(), val.begin(), val.end());
    }

    template<serde::Serializer S>
    CROSS_FUN void Serialize(S& s) const {
        name.Serialize(s);
        datatype.Serialize(s);
        uint32_t size = static_cast<uint32_t>(value.size());
        serde::Write(s, size);
        s.WriteBuffer(cstd::span(value.data(), value.size()));
    }

    template<serde::Deserializer D>
    CROSS_FUN static Attribute Deserialize(D& d) {
        gpu_string<255> n = gpu_string<255>::Deserialize(d);
        DatatypeRef dt = DatatypeRef::Deserialize(d);
        uint32_t size = serde::Read<uint32_t>(d);

        KVHDF5_ASSERT(size <= kMaxValueSize, "Attribute value too large");

        cstd::array<byte_t, kMaxValueSize> temp_buffer;
        d.ReadBuffer(cstd::span(temp_buffer.data(), size));

        return Attribute(n, dt, cstd::span<const byte_t>(temp_buffer.data(), size));
    }

    CROSS_FUN constexpr bool operator==(const Attribute&) const = default;
};

struct DatasetMetadata {
    DatasetId id;
    DatatypeRef datatype;
    DatasetShape shape;
    vector<Attribute> attributes;

    template<serde::Serializer S>
    CROSS_FUN void Serialize(S& s) const {
        serde::Write(s, id);
        datatype.Serialize(s);
        serde::Write(s, shape);

        uint32_t count = static_cast<uint32_t>(attributes.size());
        serde::Write(s, count);
        for (uint32_t i = 0; i < count; ++i) {
            attributes[i].Serialize(s);
        }
    }

    template<serde::Deserializer D, ProvidesAllocator Ctx>
    CROSS_FUN static DatasetMetadata Deserialize(D& d, Ctx& ctx) {
        DatasetId id = serde::Read<DatasetId>(d);
        DatatypeRef datatype = DatatypeRef::Deserialize(d);
        DatasetShape shape = serde::Read<DatasetShape>(d);

        uint32_t count = serde::Read<uint32_t>(d);
        vector<Attribute> attributes(&ctx.GetAllocator());
        for (uint32_t i = 0; i < count; ++i) {
            attributes.push_back(Attribute::Deserialize(d));
        }

        return DatasetMetadata{id, datatype, shape, attributes};
    }

    CROSS_FUN bool operator==(const DatasetMetadata& other) const {
        if (id != other.id) return false;
        if (datatype != other.datatype) return false;
        if (shape != other.shape) return false;
        if (attributes.size() != other.attributes.size()) return false;
        for (size_t i = 0; i < attributes.size(); ++i) {
            if (!(attributes[i] == other.attributes[i])) return false;
        }
        return true;
    }
};

} // namespace kvhdf5

KVHDF5_AUTO_SERDE(kvhdf5::DatasetShape);
