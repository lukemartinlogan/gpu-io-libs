#pragma once

#include <cstdint>
#include <vector>

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
    /**
     * @brief Constructs a hyperslab iterator.
     *
     * @param start Starting coordinate for each dimension
     * @param count Number of blocks to select in each dimension
     * @param stride Step size between blocks in each dimension (defaults to 1 if empty)
     * @param block Size of each block in each dimension (defaults to 1 if empty)
     * @param dataset_dims Dimensions of the dataset
     *
     * @throws std::invalid_argument if parameters are invalid
     */
    HyperslabIterator(
            const std::vector<uint64_t>& start,
            const std::vector<uint64_t>& count,
            const std::vector<uint64_t>& stride,
            const std::vector<uint64_t>& block,
            const std::vector<uint64_t>& dataset_dims
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
    [[nodiscard]] const std::vector<uint64_t>& GetCurrentCoordinate() const {
        return current_coord_;
    }

    /**
     * @brief Get the linear index of the current coordinate.
     * @return Linear index in row-major order
     */
    [[nodiscard]] uint64_t GetLinearIndex() const;

    [[nodiscard]] bool IsEmpty() const;

    /**
     * @brief Get the total number of elements in the hyperslab selection.
     * @return Total element count
     */
    [[nodiscard]] uint64_t GetTotalElements() const;

    /**
     * @brief Reset iterator to the beginning of the selection.
     */
    void Reset();

private:
    static void ValidateParams(
        const std::vector<uint64_t>& start,
        const std::vector<uint64_t>& count,
        const std::vector<uint64_t>& stride,
        const std::vector<uint64_t>& block,
        const std::vector<uint64_t>& dataset_dims
    );

private:
    std::vector<uint64_t> start_;
    std::vector<uint64_t> count_;
    std::vector<uint64_t> stride_;
    std::vector<uint64_t> block_;
    std::vector<uint64_t> dataset_dims_;

    std::vector<uint64_t> current_coord_;
    std::vector<uint64_t> block_index_;  // curr pos within each block
    std::vector<uint64_t> count_index_;  // curr block index within each count

    bool at_end_;
};