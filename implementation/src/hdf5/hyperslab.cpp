#include "hyperslab.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>

HyperslabIterator::HyperslabIterator(
    const coord_t& start,
    const coord_t& count,
    const coord_t& stride,
    const coord_t& block,
    const coord_t& dataset_dims
) :
    start_(start),
    count_(count),
    dataset_dims_(dataset_dims),
    current_coord_(start),
    at_end_(false)
{

    size_t n_dims = dataset_dims.size();

    coord_t norm_stride = stride.empty() ? coord_t(n_dims, 1) : stride;
    coord_t norm_block  = block.empty() ? coord_t(n_dims, 1) : block;

    ValidateParams(start, count, norm_stride, norm_block, dataset_dims);

    stride_ = std::move(norm_stride);
    block_ = std::move(norm_block);

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

uint64_t HyperslabIterator::GetLinearIndex() const {
    if (at_end_) {
        throw std::runtime_error("Iterator is at end");
    }

    const coord_t& coords = current_coord_;
    const coord_t& dims = dataset_dims_;

    uint64_t linear_index = 0;
    uint64_t multiplier = 1;

    // last dimension changes fastest
    for (int dim = static_cast<int>(dims.size()) - 1; dim >= 0; --dim) {
        if (coords[dim] >= dims[dim]) {
            throw std::invalid_argument("Coordinate exceeds dimension bounds");
        }

        linear_index += coords[dim] * multiplier;
        multiplier *= dims[dim];
    }

    return linear_index;
}

uint64_t HyperslabIterator::GetTotalElements() const {
    if (count_.empty()) {
        return 0;
    }

    uint64_t total_elements = 1;

    for (size_t dim = 0; dim < count_.size(); ++dim) {
        if (total_elements > std::numeric_limits<uint64_t>::max() / (count_[dim] * block_[dim])) {
            throw std::overflow_error("Hyperslab selection too large");
        }

        total_elements *= count_[dim] * block_[dim];
    }

    return total_elements;
}

void HyperslabIterator::Reset() {
    at_end_ = false;

    std::ranges::fill(block_index_, 0);
    std::ranges::fill(count_index_, 0);

    current_coord_ = start_;
}

void HyperslabIterator::ValidateParams(
    const coord_t& start,
    const coord_t& count,
    const coord_t& stride,
    const coord_t& block,
    const coord_t& dataset_dims
) {
    const size_t n_dims = dataset_dims.size();

    if (n_dims == 0) {
        throw std::invalid_argument("Dataset must have at least one dimension");
    }

    if (start.size() != n_dims) {
        throw std::invalid_argument("Start must have same dimensionality as dataset");
    }

    if (count.size() != n_dims) {
        throw std::invalid_argument("Count must have same dimensionality as dataset");
    }

    if (stride.size() != n_dims) {
        throw std::invalid_argument("Stride must have same dimensionality as dataset");
    }

    if (block.size() != n_dims) {
        throw std::invalid_argument("Block must have same dimensionality as dataset");
    }

    for (size_t dim = 0; dim < n_dims; ++dim) {
        if (dataset_dims[dim] == 0) {
            throw std::invalid_argument("Dataset dimension cannot be zero");
        }

        if (count[dim] == 0) {
            throw std::invalid_argument("Count cannot be zero");
        }

        if (stride[dim] == 0) {
            throw std::invalid_argument("Stride cannot be zero");
        }

        if (block[dim] == 0) {
            throw std::invalid_argument("Block cannot be zero");
        }

        // Check if the last element in the selection is within bounds
        uint64_t last_block_start = start[dim] + (count[dim] - 1) * stride[dim];
        uint64_t last_element = last_block_start + block[dim] - 1;

        if (last_element >= dataset_dims[dim]) {
            throw std::invalid_argument("Hyperslab selection exceeds dataset bounds");
        }
    }
}

void ValidateHyperslabParameters(
    const HyperslabIterator::coord_t& start,
    const HyperslabIterator::coord_t& count,
    const HyperslabIterator::coord_t& stride,
    const HyperslabIterator::coord_t& block,
    const HyperslabIterator::coord_t& dataset_dims
) {
    if (start.empty()) {
        throw std::invalid_argument("Start coordinates cannot be empty");
    }

    if (count.empty()) {
        throw std::invalid_argument("Count cannot be empty");
    }

    const size_t n_dims = dataset_dims.size();

    if (start.size() != n_dims) {
        throw std::invalid_argument("Start must have same dimensionality as dataset");
    }

    if (count.size() != n_dims) {
        throw std::invalid_argument("Count must have same dimensionality as dataset");
    }

    if (!stride.empty() && stride.size() != n_dims) {
        throw std::invalid_argument("Stride must be empty or have same dimensionality as dataset");
    }

    if (!block.empty() && block.size() != n_dims) {
        throw std::invalid_argument("Block must be empty or have same dimensionality as dataset");
    }

    for (size_t dim = 0; dim < n_dims; ++dim) {
        if (dataset_dims[dim] == 0) {
            throw std::invalid_argument("Dataset dimension cannot be zero");
        }

        if (count[dim] == 0) {
            throw std::invalid_argument("Count cannot be zero");
        }

        if (stride[dim] == 0) {
            throw std::invalid_argument("Stride cannot be zero");
        }

        if (block[dim] == 0) {
            throw std::invalid_argument("Block cannot be zero");
        }

        if (start[dim] >= dataset_dims[dim]) {
            throw std::invalid_argument("Start coordinate exceeds dataset dimension");
        }

        // Check if the last element in the selection is within bounds
        uint64_t last_block_start = start[dim] + (count[dim] - 1) * stride[dim];
        uint64_t last_element = last_block_start + block[dim] - 1;

        if (last_element >= dataset_dims[dim]) {
            throw std::invalid_argument("Hyperslab selection exceeds dataset bounds");
        }
    }
}
