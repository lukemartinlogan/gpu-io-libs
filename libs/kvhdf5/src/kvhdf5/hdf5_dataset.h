#pragma once

#include "hdf5_group.h"  // for DatasetCreateProps, Group forward decl
#include "ref.h"
#include "error.h"
#include "dataspace.h"
#include "hdf5_datatype.h"
#include <cuda/std/span>

namespace kvhdf5 {

// Forward declare AttributeHandle<B> - will be defined in hdf5_attribute.h
template<RawBlobStore B>
class AttributeHandle;

template<RawBlobStore B>
class Dataset {
    DatasetId id_;
    Ref<Container<B>> container_;

    // --- Hyperslab helpers ---

    /// Decompose the Nth selected point in a hyperslab into per-dimension
    /// coordinates within the full dataspace, then linearize to a flat
    /// row-major index.  When the selection is SelectAll, the Nth point is
    /// simply index N.
    static CROSS_FUN uint64_t GetNthSelectedPointFlat(
        const Dataspace& space, uint64_t n, uint8_t ndims
    ) {
        if (space.GetSelectionType() == SelectionType::All) {
            return n;
        }

        auto& hyp = space.GetHyperslab();
        // Decompose n into per-dimension (count*block) indices, then map to
        // full-space coordinates.
        uint64_t coords[MAX_DIMS];
        uint64_t remainder = n;
        for (int d = ndims - 1; d >= 0; --d) {
            uint64_t dim_selected = hyp.count[d] * hyp.block[d];
            uint64_t dim_idx = remainder % dim_selected;
            remainder /= dim_selected;

            uint64_t ci = dim_idx / hyp.block[d];
            uint64_t bi = dim_idx % hyp.block[d];
            coords[d] = hyp.start[d] + ci * hyp.stride[d] + bi;
        }

        // Linearize using the dataspace dims
        auto dims_span = space.GetDimsSpan();

        uint64_t flat = 0;
        uint64_t stride = 1;
        for (int d = ndims - 1; d >= 0; --d) {
            flat += coords[d] * stride;
            stride *= dims_span[d];
        }
        return flat;
    }

    // TODO: return a dim_vector
    /// Decompose a flat dataset index into per-dimension coordinates.
    static CROSS_FUN void FlatToCoords(
        uint64_t flat, cstd::span<const uint64_t> dims,
        cstd::span<uint64_t> coords_out
    ) {
        KVHDF5_ASSERT(
            dims.size() == coords_out.size(),
            "FlatToCoords: dims and coords_out must have same rank"
        );
        KVHDF5_ASSERT(
            !dims.empty(),
            "FlatToCoords: dims must not be empty"
        );
        
        uint64_t remainder = flat;
        for (size_t d = dims.size(); d > 0; --d) {
            size_t i = d - 1;
            coords_out[i] = remainder % dims[i];
            remainder /= dims[i];
        }
    }

    /// Linearize per-dimension coordinates to flat row-major index.
    static CROSS_FUN uint64_t CoordsToFlat(
        const uint64_t* coords, const uint64_t* dims, uint8_t ndims
    ) {
        uint64_t flat = 0;
        uint64_t stride = 1;
        for (int d = ndims - 1; d >= 0; --d) {
            flat += coords[d] * stride;
            stride *= dims[d];
        }
        return flat;
    }

    /// Copy `elem_size` bytes between two pointers.
    static CROSS_FUN void CopyElement(
        byte_t* dst, const byte_t* src, uint32_t elem_size
    ) {
        cuda::std::memcpy(dst, src, elem_size);
    }

public:
    CROSS_FUN Dataset(DatasetId id, Ref<Container<B>> container)
        : id_(id), container_(container) {}

    // Maximum stack buffer size for chunk data (64KB)
    static constexpr size_t kMaxChunkBytes = 64 * 1024;

    // --- Write (SelectAll + Hyperslab, multi-chunk) ---
    expected<void> Write(
        const Datatype& mem_type,
        const Dataspace& mem_space,
        const Dataspace& file_space,
        const void* buf
    ) {
        // For SelectNone, do nothing
        if (file_space.GetSelectionType() == SelectionType::None) {
            return {};
        }

        // Load dataset metadata
        auto meta_result = container_->GetDataset(id_);
        if (!meta_result.has_value()) {
            return make_error(ErrorCode::InvalidArgument, "dataset not found");
        }
        auto& meta = meta_result.value();
        auto& shape = meta.shape;

        uint32_t elem_size = mem_type.GetSize();
        uint8_t ndims = shape.Ndims();
        const auto* src = static_cast<const byte_t*>(buf);

        // Compute number of chunks per dimension
        uint64_t num_chunks[MAX_DIMS];
        for (uint8_t d = 0; d < ndims; ++d) {
            num_chunks[d] = (shape.dims[d] + shape.chunk_dims[d] - 1)
                            / shape.chunk_dims[d];
        }

        // Determine chunk range to iterate
        uint64_t first_chunk[MAX_DIMS] = {};
        uint64_t last_chunk[MAX_DIMS] = {};  // inclusive

        if (file_space.GetSelectionType() == SelectionType::Hyperslab) {
            auto& hyp = file_space.GetHyperslab();
            for (uint8_t d = 0; d < ndims; ++d) {
                first_chunk[d] = hyp.start[d] / shape.chunk_dims[d];
                uint64_t last_point = hyp.start[d]
                    + (hyp.count[d] - 1) * hyp.stride[d]
                    + hyp.block[d] - 1;
                last_chunk[d] = last_point / shape.chunk_dims[d];
            }
        } else {
            // SelectAll
            for (uint8_t d = 0; d < ndims; ++d) {
                first_chunk[d] = 0;
                last_chunk[d] = num_chunks[d] - 1;
            }
        }

        uint64_t num_selected = file_space.GetSelectedPointCount();
        uint8_t mem_ndims = mem_space.GetNDims();

        // Iterate chunks in range
        uint64_t chunk_coords[MAX_DIMS];
        for (uint8_t d = 0; d < ndims; ++d) chunk_coords[d] = first_chunk[d];

        bool done = false;
        while (!done) {
            // Compute chunk start position and actual size
            uint64_t chunk_start[MAX_DIMS];
            uint64_t chunk_end[MAX_DIMS];
            uint64_t chunk_actual[MAX_DIMS];
            uint64_t chunk_total_elems = 1;
            for (uint8_t d = 0; d < ndims; ++d) {
                chunk_start[d] = chunk_coords[d] * shape.chunk_dims[d];
                chunk_end[d] = chunk_start[d] + shape.chunk_dims[d];
                if (chunk_end[d] > shape.dims[d]) chunk_end[d] = shape.dims[d];
                chunk_actual[d] = chunk_end[d] - chunk_start[d];
                chunk_total_elems *= chunk_actual[d];
            }

            uint64_t chunk_bytes = chunk_total_elems * elem_size;
            KVHDF5_ASSERT(chunk_bytes <= kMaxChunkBytes,
                          "chunk too large for stack buffer");

            byte_t chunk_buf[kMaxChunkBytes];

            if (file_space.GetSelectionType() == SelectionType::All) {
                // SelectAll: every element in this chunk is written from the
                // user buffer.  No need to load existing data.
                for (uint64_t flat = 0; flat < chunk_total_elems; ++flat) {
                    uint64_t local_coords[MAX_DIMS];
                    uint64_t remainder = flat;
                    for (int d = ndims - 1; d >= 0; --d) {
                        local_coords[d] = remainder % chunk_actual[d];
                        remainder /= chunk_actual[d];
                    }

                    // Compute global coords and flatten
                    uint64_t global[MAX_DIMS];
                    for (uint8_t d = 0; d < ndims; ++d)
                        global[d] = chunk_start[d] + local_coords[d];
                    uint64_t ds_flat = CoordsToFlat(global, shape.dims.data(), ndims);

                    // Map through mem_space to get buffer offset
                    uint64_t mem_flat = GetNthSelectedPointFlat(
                        mem_space, ds_flat, mem_ndims);

                    CopyElement(chunk_buf + flat * elem_size,
                                src + mem_flat * elem_size, elem_size);
                }
            } else {
                // Hyperslab: only some elements in this chunk are written.
                // First load existing chunk data (or zero-init).
                ChunkKey load_key(id_,
                    cstd::span<const uint64_t>(chunk_coords, ndims));
                bool chunk_exists = container_->ChunkExists(load_key);
                if (chunk_exists) {
                    auto chunk_span = cstd::span<byte_t>(chunk_buf, chunk_bytes);
                    auto result = container_->GetChunk(load_key, chunk_span);
                    if (!result.has_value()) {
                        return make_error(ErrorCode::InvalidArgument,
                                          "failed to read chunk for partial write");
                    }
                } else {
                    for (uint64_t i = 0; i < chunk_bytes; ++i)
                        chunk_buf[i] = byte_t{0};
                }

                // Iterate all selected points and check if they fall in this chunk
                bool any_written = false;
                for (uint64_t n = 0; n < num_selected; ++n) {
                    uint64_t file_flat = GetNthSelectedPointFlat(
                        file_space, n, ndims);
                    uint64_t file_coords[MAX_DIMS];
                    FlatToCoords(file_flat, shape.dims.data(), ndims, file_coords);

                    // Check if this point is in the current chunk
                    bool in_chunk = true;
                    uint64_t local[MAX_DIMS];
                    for (uint8_t d = 0; d < ndims; ++d) {
                        if (file_coords[d] < chunk_start[d] ||
                            file_coords[d] >= chunk_end[d]) {
                            in_chunk = false;
                            break;
                        }
                        local[d] = file_coords[d] - chunk_start[d];
                    }
                    if (!in_chunk) continue;

                    any_written = true;

                    // Compute local flat index within chunk
                    uint64_t local_flat = CoordsToFlat(local, chunk_actual, ndims);

                    // Compute memory buffer offset
                    uint64_t mem_flat = GetNthSelectedPointFlat(
                        mem_space, n, mem_ndims);

                    CopyElement(chunk_buf + local_flat * elem_size,
                                src + mem_flat * elem_size, elem_size);
                }

                // Only write the chunk if we actually modified it
                if (!any_written && !chunk_exists) {
                    // Advance to next chunk
                    goto advance_write;
                }
            }

            {
                // Store chunk
                ChunkKey key(id_,
                    cstd::span<const uint64_t>(chunk_coords, ndims));
                container_->PutChunk(key,
                    cstd::span<const byte_t>(chunk_buf, chunk_bytes));
            }

advance_write:
            // Advance to next chunk in range (row-major increment)
            done = true;
            for (int d = ndims - 1; d >= 0; --d) {
                chunk_coords[d]++;
                if (chunk_coords[d] <= last_chunk[d]) {
                    done = false;
                    break;
                }
                chunk_coords[d] = first_chunk[d];
            }
        }

        return {};
    }

    // --- Read (SelectAll + Hyperslab, multi-chunk) ---
    expected<void> Read(
        const Datatype& mem_type,
        const Dataspace& mem_space,
        const Dataspace& file_space,
        void* buf
    ) const {
        if (file_space.GetSelectionType() == SelectionType::None) {
            return {};
        }

        auto meta_result = container_->GetDataset(id_);
        if (!meta_result.has_value()) {
            return make_error(ErrorCode::InvalidArgument, "dataset not found");
        }
        auto& meta = meta_result.value();
        auto& shape = meta.shape;

        uint32_t elem_size = mem_type.GetSize();
        uint8_t ndims = shape.Ndims();
        auto* dst = static_cast<byte_t*>(buf);

        // Compute number of chunks per dimension
        uint64_t num_chunks[MAX_DIMS];
        for (uint8_t d = 0; d < ndims; ++d) {
            num_chunks[d] = (shape.dims[d] + shape.chunk_dims[d] - 1)
                            / shape.chunk_dims[d];
        }

        // Determine chunk range to iterate
        uint64_t first_chunk[MAX_DIMS] = {};
        uint64_t last_chunk[MAX_DIMS] = {};

        if (file_space.GetSelectionType() == SelectionType::Hyperslab) {
            auto& hyp = file_space.GetHyperslab();
            for (uint8_t d = 0; d < ndims; ++d) {
                first_chunk[d] = hyp.start[d] / shape.chunk_dims[d];
                uint64_t last_point = hyp.start[d]
                    + (hyp.count[d] - 1) * hyp.stride[d]
                    + hyp.block[d] - 1;
                last_chunk[d] = last_point / shape.chunk_dims[d];
            }
        } else {
            for (uint8_t d = 0; d < ndims; ++d) {
                first_chunk[d] = 0;
                last_chunk[d] = num_chunks[d] - 1;
            }
        }

        uint64_t num_selected = file_space.GetSelectedPointCount();
        uint8_t mem_ndims = mem_space.GetNDims();

        // Iterate chunks in range
        uint64_t chunk_coords[MAX_DIMS];
        for (uint8_t d = 0; d < ndims; ++d) chunk_coords[d] = first_chunk[d];

        bool done = false;
        while (!done) {
            // Compute chunk start position and actual size
            uint64_t chunk_start[MAX_DIMS];
            uint64_t chunk_end[MAX_DIMS];
            uint64_t chunk_actual[MAX_DIMS];
            uint64_t chunk_total_elems = 1;
            for (uint8_t d = 0; d < ndims; ++d) {
                chunk_start[d] = chunk_coords[d] * shape.chunk_dims[d];
                chunk_end[d] = chunk_start[d] + shape.chunk_dims[d];
                if (chunk_end[d] > shape.dims[d]) chunk_end[d] = shape.dims[d];
                chunk_actual[d] = chunk_end[d] - chunk_start[d];
                chunk_total_elems *= chunk_actual[d];
            }

            uint64_t chunk_bytes = chunk_total_elems * elem_size;
            KVHDF5_ASSERT(chunk_bytes <= kMaxChunkBytes,
                          "chunk too large for stack buffer");

            ChunkKey key(id_, cstd::span<const uint64_t>(chunk_coords, ndims));

            byte_t chunk_buf[kMaxChunkBytes];

            // Load chunk data (or zero-init)
            bool chunk_exists = container_->ChunkExists(key);
            if (chunk_exists) {
                auto chunk_span = cstd::span<byte_t>(chunk_buf, chunk_bytes);
                auto result = container_->GetChunk(key, chunk_span);
                if (!result.has_value()) {
                    return make_error(ErrorCode::InvalidArgument,
                                      "failed to read chunk");
                }
            } else {
                for (uint64_t i = 0; i < chunk_bytes; ++i) {
                    chunk_buf[i] = byte_t{0};
                }
            }

            if (file_space.GetSelectionType() == SelectionType::All) {
                // SelectAll: scatter every element from chunk to user buffer
                for (uint64_t flat = 0; flat < chunk_total_elems; ++flat) {
                    uint64_t local_coords[MAX_DIMS];
                    uint64_t remainder = flat;
                    for (int d = ndims - 1; d >= 0; --d) {
                        local_coords[d] = remainder % chunk_actual[d];
                        remainder /= chunk_actual[d];
                    }

                    uint64_t global[MAX_DIMS];
                    for (uint8_t d = 0; d < ndims; ++d)
                        global[d] = chunk_start[d] + local_coords[d];
                    uint64_t ds_flat = CoordsToFlat(global, shape.dims.data(), ndims);

                    // Map through mem_space to get buffer offset
                    uint64_t mem_flat = GetNthSelectedPointFlat(
                        mem_space, ds_flat, mem_ndims);

                    CopyElement(dst + mem_flat * elem_size,
                                chunk_buf + flat * elem_size, elem_size);
                }
            } else {
                // Hyperslab: only scatter selected points from this chunk
                for (uint64_t n = 0; n < num_selected; ++n) {
                    uint64_t file_flat = GetNthSelectedPointFlat(
                        file_space, n, ndims);
                    uint64_t file_coords[MAX_DIMS];
                    FlatToCoords(file_flat, shape.dims.data(), ndims, file_coords);

                    bool in_chunk = true;
                    uint64_t local[MAX_DIMS];
                    for (uint8_t d = 0; d < ndims; ++d) {
                        if (file_coords[d] < chunk_start[d] ||
                            file_coords[d] >= chunk_end[d]) {
                            in_chunk = false;
                            break;
                        }
                        local[d] = file_coords[d] - chunk_start[d];
                    }
                    if (!in_chunk) continue;

                    uint64_t local_flat = CoordsToFlat(local, chunk_actual, ndims);
                    uint64_t mem_flat = GetNthSelectedPointFlat(
                        mem_space, n, mem_ndims);

                    CopyElement(dst + mem_flat * elem_size,
                                chunk_buf + local_flat * elem_size, elem_size);
                }
            }

            // Advance to next chunk in range (row-major increment)
            done = true;
            for (int d = ndims - 1; d >= 0; --d) {
                chunk_coords[d]++;
                if (chunk_coords[d] <= last_chunk[d]) {
                    done = false;
                    break;
                }
                chunk_coords[d] = first_chunk[d];
            }
        }

        return {};
    }

    // --- Query ---

    expected<Dataspace> GetSpace() const {
        auto meta_result = container_->GetDataset(id_);
        if (!meta_result.has_value()) {
            return make_error(ErrorCode::InvalidArgument, "dataset not found");
        }
        auto& meta = meta_result.value();
        auto dims_span = meta.shape.Dims();
        return Dataspace::CreateSimple(dims_span);
    }

    Datatype GetType() const {
        auto meta_result = container_->GetDataset(id_);
        KVHDF5_ASSERT(meta_result.has_value(), "Dataset::GetType: dataset not found");
        auto& meta = meta_result.value();
        // The internal DatatypeRef wraps a PrimitiveType
        // Convert back to public Datatype
        auto prim = meta.datatype.GetPrimitive();
        KVHDF5_ASSERT(prim.has_value(), "Dataset::GetType: only primitive types supported");
        // Use the appropriate factory based on kind
        switch (prim.value().kind) {
            case PrimitiveType::Kind::Int8:    return Datatype::Int8();
            case PrimitiveType::Kind::Int16:   return Datatype::Int16();
            case PrimitiveType::Kind::Int32:   return Datatype::Int32();
            case PrimitiveType::Kind::Int64:   return Datatype::Int64();
            case PrimitiveType::Kind::Uint8:   return Datatype::Uint8();
            case PrimitiveType::Kind::Uint16:  return Datatype::Uint16();
            case PrimitiveType::Kind::Uint32:  return Datatype::Uint32();
            case PrimitiveType::Kind::Uint64:  return Datatype::Uint64();
            case PrimitiveType::Kind::Float32: return Datatype::Float32();
            case PrimitiveType::Kind::Float64: return Datatype::Float64();
        }
        KVHDF5_ASSERT(false, "Dataset::GetType: unreachable");
        return Datatype::Int8();  // unreachable
    }

    CROSS_FUN DatasetId GetId() const { return id_; }

    // --- SetExtent ---

    /// Grow or shrink the dataset dimensions.
    /// Updates metadata only; existing chunks are not modified.
    /// Growing adds empty space (reads return zero).
    /// Shrinking leaves orphaned chunks (acceptable for now).
    expected<void> SetExtent(cstd::span<const uint64_t> new_dims) {
        auto meta_result = container_->GetDataset(id_);
        if (!meta_result.has_value()) {
            return make_error(ErrorCode::InvalidArgument, "dataset not found");
        }
        auto meta = meta_result.value();

        if (new_dims.size() != meta.shape.Ndims()) {
            return make_error(ErrorCode::InvalidArgument, "rank mismatch");
        }

        for (uint8_t d = 0; d < meta.shape.Ndims(); ++d) {
            meta.shape.dims[d] = new_dims[d];
        }

        container_->PutDataset(id_, meta);
        return {};
    }

    // --- Attributes (declarations; defined in hdf5_attribute.h) ---
    expected<void> SetAttribute(gpu_string_view name, const Datatype& type, const void* data);
    expected<void> GetAttribute(gpu_string_view name, const Datatype& type, void* data) const;
    bool HasAttribute(gpu_string_view name) const;
    AttributeHandle<B> OpenAttribute(gpu_string_view name);

    // --- ChunkIter ---

    /// Callback type for ChunkIter.
    /// Returns true to continue iteration, false to stop early.
    using ChunkIterCallback = bool(*)(const ChunkKey& key, uint64_t size, void* user_data);

    /// Iterate over all existing chunks in the dataset.
    /// For each chunk that exists, calls the callback with the ChunkKey and
    /// the chunk data size in bytes.
    expected<void> ChunkIter(ChunkIterCallback callback, void* user_data) const {
        auto meta_result = container_->GetDataset(id_);
        if (!meta_result.has_value()) {
            return make_error(ErrorCode::InvalidArgument, "dataset not found");
        }
        auto& meta = meta_result.value();
        auto& shape = meta.shape;
        uint8_t ndims = shape.Ndims();
        uint32_t elem_size = meta.datatype.GetPrimitive().value().GetSize();

        uint64_t num_chunks[MAX_DIMS];
        for (uint8_t d = 0; d < ndims; ++d) {
            num_chunks[d] = (shape.dims[d] + shape.chunk_dims[d] - 1)
                            / shape.chunk_dims[d];
        }

        uint64_t chunk_coords[MAX_DIMS] = {};
        bool done = false;
        while (!done) {
            ChunkKey key(id_, cstd::span<const uint64_t>(chunk_coords, ndims));
            bool exists = container_->ChunkExists(key);

            if (exists) {
                // Compute chunk size in bytes
                uint64_t chunk_elems = 1;
                for (uint8_t d = 0; d < ndims; ++d) {
                    uint64_t chunk_start = chunk_coords[d] * shape.chunk_dims[d];
                    uint64_t chunk_end = chunk_start + shape.chunk_dims[d];
                    if (chunk_end > shape.dims[d]) chunk_end = shape.dims[d];
                    chunk_elems *= (chunk_end - chunk_start);
                }
                uint64_t chunk_bytes = chunk_elems * elem_size;

                if (!callback(key, chunk_bytes, user_data)) {
                    break;  // early termination
                }
            }

            // Advance chunk coords (row-major)
            done = true;
            for (int d = ndims - 1; d >= 0; --d) {
                chunk_coords[d]++;
                if (chunk_coords[d] < num_chunks[d]) {
                    done = false;
                    break;
                }
                chunk_coords[d] = 0;
            }
        }

        return {};
    }
};

// --- Group<B>::CreateDataset and OpenDataset ---
// Defined here because they return Dataset<B>, which is now complete.

template<RawBlobStore B>
expected<Dataset<B>> Group<B>::CreateDataset(
    gpu_string_view name, const Datatype& type,
    const Dataspace& space, const DatasetCreateProps& props
) {
    auto meta_result = container_->GetGroup(id_);
    if (!meta_result.has_value()) {
        return make_error(ErrorCode::InvalidArgument, "parent group not found");
    }
    auto meta = meta_result.value();

    // Check no duplicate name
    for (size_t i = 0; i < meta.children.size(); ++i) {
        if (meta.children[i].name == name) {
            return make_error(ErrorCode::InvalidArgument, "child with this name already exists");
        }
    }

    // Build DatasetShape from Dataspace dims and props chunk_dims
    uint8_t ndims = space.GetNDims();
    uint64_t dims_arr[MAX_DIMS];
    space.GetDims(cstd::span<uint64_t>(dims_arr, ndims));

    // Construct DatasetShape manually (DatasetShape::Create uses initializer_list
    // which cannot be built dynamically at runtime)
    DatasetShape shape;
    shape.ndims_ = ndims;
    for (uint8_t i = 0; i < ndims; ++i) {
        shape.dims[i] = dims_arr[i];
        if (!props.chunk_dims.empty()) {
            shape.chunk_dims[i] = props.chunk_dims[i];
        } else {
            shape.chunk_dims[i] = dims_arr[i];  // single chunk
        }
    }
    for (uint8_t i = ndims; i < MAX_DATASET_DIMS; ++i) {
        shape.dims[i] = 0;
        shape.chunk_dims[i] = 0;
    }

    // Allocate dataset ID and create metadata
    auto dataset_id = DatasetId(container_->AllocateId());
    DatasetMetadata ds_meta(dataset_id, type.ToRef(), shape, container_->GetAllocator());
    container_->PutDataset(dataset_id, ds_meta);

    // Add to parent group children
    meta.children.push_back(GroupEntry::NewDataset(dataset_id, name));
    container_->PutGroup(id_, meta);

    return Dataset<B>(dataset_id, container_);
}

template<RawBlobStore B>
expected<Dataset<B>> Group<B>::OpenDataset(gpu_string_view name) {
    auto meta_result = container_->GetGroup(id_);
    if (!meta_result.has_value()) {
        return make_error(ErrorCode::InvalidArgument, "group not found");
    }
    auto& meta = meta_result.value();

    for (size_t i = 0; i < meta.children.size(); ++i) {
        if (meta.children[i].kind == ChildKind::Dataset &&
            meta.children[i].name == name) {
            return Dataset<B>(DatasetId(meta.children[i].object_id), container_);
        }
    }

    return make_error(ErrorCode::InvalidArgument, "dataset not found");
}

} // namespace kvhdf5
