#include "hyperslab.h"

#include <algorithm>

hdf5::expected<HyperslabIterator> HyperslabIterator::New(
    const coord_t& start,
    const coord_t& count,
    const coord_t& stride,
    const coord_t& block,
    const coord_t& dataset_dims
) {
    size_t n_dims = dataset_dims.size();

    coord_t norm_stride = stride.empty() ? coord_t(n_dims, 1) : stride;
    coord_t norm_block  = block.empty() ? coord_t(n_dims, 1) : block;

    if (auto error = ValidateParams(start, count, norm_stride, norm_block, dataset_dims)) {
        return cstd::unexpected(*error);
    }

    return HyperslabIterator(start, count, std::move(norm_stride), std::move(norm_block), dataset_dims);
}

HyperslabIterator::HyperslabIterator(
    const coord_t& start,
    const coord_t& count,
    const coord_t& stride,
    const coord_t& block,
    const coord_t& dataset_dims
) :
    start_(start),
    count_(count),
    stride_(stride),
    block_(block),
    dataset_dims_(dataset_dims),
    current_coord_(start),
    at_end_(false)
{
    size_t n_dims = dataset_dims.size();
    block_index_.resize(n_dims, 0);
    count_index_.resize(n_dims, 0);
}

bool HyperslabIterator::Advance() {
    if (at_end_) {
        return false;
    }

    const size_t n_dims = dataset_dims_.size();

    // increment block position starting from the last dimension (row-major order)
    // int here to allow negative for bounds check
    for (int dim = static_cast<int>(n_dims) - 1; dim >= 0; --dim) {
        block_index_[dim]++;

        if (block_index_[dim] < block_[dim]) {
            // still within current block in this dimension
            current_coord_[dim] = start_[dim] + count_index_[dim] * stride_[dim] + block_index_[dim];
            return true;
        }

        // finished this block, move to next block in this dimension
        block_index_[dim] = 0;
        count_index_[dim]++;

        if (count_index_[dim] < count_[dim]) {
            // still have more blocks in this dimension
            current_coord_[dim] = start_[dim] + count_index_[dim] * stride_[dim];
            return true;
        }

        // finished all blocks in this dimension, reset and carry to next dimension
        count_index_[dim] = 0;
        current_coord_[dim] = start_[dim];
    }

    // finished when past the first dimension
    at_end_ = true;
    return false;
}

hdf5::expected<uint64_t> HyperslabIterator::GetLinearIndex() const {
    if (at_end_) {
        return hdf5::error(hdf5::HDF5ErrorCode::IteratorAtEnd, "Iterator is at end");
    }

    const coord_t& coords = current_coord_;
    const coord_t& dims = dataset_dims_;

    uint64_t linear_index = 0;
    uint64_t multiplier = 1;

    // last dimension changes fastest
    for (int dim = static_cast<int>(dims.size()) - 1; dim >= 0; --dim) {
        if (coords[dim] >= dims[dim]) {
            return hdf5::error(hdf5::HDF5ErrorCode::CoordinateOutOfBounds, "Coordinate exceeds dimension bounds");
        }

        linear_index += coords[dim] * multiplier;
        multiplier *= dims[dim];
    }

    return linear_index;
}

hdf5::expected<uint64_t> HyperslabIterator::GetTotalElements() const {
    if (count_.empty()) {
        return hdf5::error(hdf5::HDF5ErrorCode::EmptyParameter, "Count is empty");
    }

    uint64_t total_elements = 1;

    for (size_t dim = 0; dim < count_.size(); ++dim) {
        // TODO: windows defines max as a macro :(
        if (total_elements > static_cast<uint64_t>(-1) / (count_[dim] * block_[dim])) {
            return hdf5::error(hdf5::HDF5ErrorCode::SelectionOverflow, "Hyperslab selection too large");
        }

        total_elements *= count_[dim] * block_[dim];
    }

    return total_elements;
}

void HyperslabIterator::Reset() {
    at_end_ = false;

    cstd::fill(block_index_.begin(), block_index_.end(), 0);
    cstd::fill(count_index_.begin(), count_index_.end(), 0);

    current_coord_ = start_;
}

cstd::optional<hdf5::HDF5Error> HyperslabIterator::ValidateParams(
    const coord_t& start,
    const coord_t& count,
    const coord_t& stride,
    const coord_t& block,
    const coord_t& dataset_dims
) {
    const size_t n_dims = dataset_dims.size();

    if (n_dims == 0) {
        return hdf5::HDF5Error{hdf5::HDF5ErrorCode::EmptyParameter, "Dataset must have at least one dimension"};
    }

    if (start.size() != n_dims) {
        return hdf5::HDF5Error{hdf5::HDF5ErrorCode::DimensionMismatch, "Start must have same dimensionality as dataset"};
    }

    if (count.size() != n_dims) {
        return hdf5::HDF5Error{hdf5::HDF5ErrorCode::DimensionMismatch, "Count must have same dimensionality as dataset"};
    }

    if (stride.size() != n_dims) {
        return hdf5::HDF5Error{hdf5::HDF5ErrorCode::DimensionMismatch, "Stride must have same dimensionality as dataset"};
    }

    if (block.size() != n_dims) {
        return hdf5::HDF5Error{hdf5::HDF5ErrorCode::DimensionMismatch, "Block must have same dimensionality as dataset"};
    }

    for (size_t dim = 0; dim < n_dims; ++dim) {
        if (dataset_dims[dim] == 0) {
            return hdf5::HDF5Error{hdf5::HDF5ErrorCode::ZeroParameter, "Dataset dimension cannot be zero"};
        }

        if (count[dim] == 0) {
            return hdf5::HDF5Error{hdf5::HDF5ErrorCode::ZeroParameter, "Count cannot be zero"};
        }

        if (stride[dim] == 0) {
            return hdf5::HDF5Error{hdf5::HDF5ErrorCode::ZeroParameter, "Stride cannot be zero"};
        }

        if (block[dim] == 0) {
            return hdf5::HDF5Error{hdf5::HDF5ErrorCode::ZeroParameter, "Block cannot be zero"};
        }

        if (start[dim] >= dataset_dims[dim]) {
            return hdf5::HDF5Error{hdf5::HDF5ErrorCode::CoordinateOutOfBounds, "Start coordinate exceeds dataset bounds"};
        }

        if (stride[dim] < block[dim]) {
            return hdf5::HDF5Error{hdf5::HDF5ErrorCode::BlockOverlap, "Hyperslab blocks overlap: stride < block"};
        }

        uint64_t last_block_start = start[dim] + (count[dim] - 1) * stride[dim];
        uint64_t last_element = last_block_start + block[dim] - 1;

        if (last_element >= dataset_dims[dim]) {
            return hdf5::HDF5Error{hdf5::HDF5ErrorCode::SelectionOutOfBounds, "Hyperslab selection exceeds dataset bounds"};
        }
    }

    return cstd::nullopt;
}
