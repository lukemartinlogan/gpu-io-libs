#pragma once

#include <cstdint>

#include "types.h"

/**
 * @brief Iterator for traversing selected elements in a hyperslab selection.
 * 
 * This class implements HDF5's hyperslab selection pattern, iterating through
 * all selected coordinates in row-major (C) order where the last dimension
 * changes fastest.
 * 
 * The mathematical relationship for hyperslab selection is:
 * For dimension i, selected elements are at positions: start[i] + k*stride[i] + j
 * where k ranges from 0 to count[i]-1 and j ranges from 0 to block[i]-1.
 */
class HyperslabIterator {
public:
    using coord_t = hdf5::dim_vector<uint64_t>;

    /**
     * @brief Creates a hyperslab iterator.
     *
     * @param start Starting coordinate for each dimension
     * @param count Number of blocks to select in each dimension
     * @param stride Step size between blocks in each dimension (defaults to 1 if empty)
     * @param block Size of each block in each dimension (defaults to 1 if empty)
     * @param dataset_dims Dimensions of the dataset
     *
     * @return expected containing the iterator or an error if parameters are invalid
     */
    static hdf5::expected<HyperslabIterator> New(
            const coord_t& start,
            const coord_t& count,
            const coord_t& stride,
            const coord_t& block,
            const coord_t& dataset_dims
    );

    /**
     * @brief Move to the next selected element.
     * @return true if advanced successfully, false if at end
     */
    bool Advance();

    /**
     * @brief Check if iterator has reached the end.
     * @return true if at end, false otherwise
     */
    [[nodiscard]] bool IsAtEnd() const {
        return at_end_;
    }

    /**
     * @brief Get the current multi-dimensional coordinate.
     * @return Current coordinate vector
     */
    [[nodiscard]] const coord_t& GetCurrentCoordinate() const {
        return current_coord_;
    }

    /**
     * @brief Get the linear index of the current coordinate.
     * @return Linear index in row-major order
     */
    [[nodiscard]] hdf5::expected<uint64_t> GetLinearIndex() const;

    /**
     * @brief Get the total number of elements in the hyperslab selection.
     * @return Total element count
     */
    [[nodiscard]] hdf5::expected<uint64_t> GetTotalElements() const;

    /**
     * @brief Reset iterator to the beginning of the selection.
     */
    void Reset();

private:
    HyperslabIterator(
        const coord_t& start,
        const coord_t& count,
        const coord_t& stride,
        const coord_t& block,
        const coord_t& dataset_dims
    );

    static cstd::optional<hdf5::HDF5Error> ValidateParams(
        const coord_t& start,
        const coord_t& count,
        const coord_t& stride,
        const coord_t& block,
        const coord_t& dataset_dims
    );

    coord_t start_;
    coord_t count_;
    coord_t stride_;
    coord_t block_;
    coord_t dataset_dims_;

    coord_t current_coord_;
    coord_t block_index_;  // curr pos within each block
    coord_t count_index_;  // curr block index within each count

    bool at_end_;
};