#include <numeric>
#include <stdexcept>

#include "object_header.h"
#include "datatype.h"
#include "../serialization/buffer.h"
#include "../util/string.h"

__device__ __host__
size_t DataspaceMessage::TotalElements() const {
    return std::accumulate(
        dimensions.begin(), dimensions.end(),
        1,
        [](size_t acc, const DimensionInfo& info) {
            return acc * info.size;
        }
    );
}

__device__ __host__
size_t DataspaceMessage::MaxElements() const {
    return std::accumulate(
        dimensions.begin(), dimensions.end(),
        1,
        [](size_t acc, const DimensionInfo& info) {
            return acc * info.max_size;
        }
    );
}

__device__ __host__
DataspaceMessage::DataspaceMessage(const hdf5::dim_vector<DimensionInfo>& dimensions, bool max_dim_present, bool perm_indices_present) {
    ASSERT(dimensions.size() <= 255, "DataspaceMessage cannot have more than 255 dimensions");

    this->dimensions = dimensions;

    bitset_.set(0, max_dim_present);
    bitset_.set(1, perm_indices_present);
}

__device__ __host__
uint16_t ObjectHeaderMessage::MessageType() const {
    auto index = cstd::visit([]<typename T>(const T&) { return T::kType; }, message);

    ASSERT(index == message.index(), "mismatch between variant index and message type");

    return index;
}