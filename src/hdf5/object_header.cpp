#include <numeric>
#include <stdexcept>

#include "object_header.h"
#include "datatype.h"
#include "../serialization/buffer.h"
#include "../util/string.h"

__device__
size_t DataspaceMessage::TotalElements() const {
    size_t total = 1;
    for (const auto& dim : dimensions) {
        total *= dim.size;
    }
    return total;
}

__device__
size_t DataspaceMessage::MaxElements() const {
    size_t total = 1;
    for (const auto& dim : dimensions) {
        total *= dim.max_size;
    }
    return total;
}

__device__
DataspaceMessage::DataspaceMessage(const hdf5::dim_vector<DimensionInfo>& dimensions, bool max_dim_present, bool perm_indices_present) {
    ASSERT(dimensions.size() <= 255, "DataspaceMessage cannot have more than 255 dimensions");

    this->dimensions = dimensions;

    bitset_.set(0, max_dim_present);
    bitset_.set(1, perm_indices_present);
}

__device__
uint16_t ObjectHeaderMessage::MessageType() const {
    auto index = cstd::visit([]<typename T>(const T&) { return T::kType; }, message);

    ASSERT(index == message.index(), "mismatch between variant index and message type");

    return index;
}