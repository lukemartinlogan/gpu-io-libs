// gpu_container_helpers.cc — Plain CXX translation unit.
//
// All GroupMetadata / DatasetMetadata construction and PutGroup / PutDataset
// calls live here so they are never seen by clang-18's NVPTX device codegen.
// See gpu_container_helpers.h for the rationale.

#include "gpu_container_helpers.h"
#include "kvhdf5/group.h"
#include "kvhdf5/dataset.h"
#include "kvhdf5/datatype.h"

namespace kvhdf5::test {

bool HostPutGroup(ContainerT* container, GroupId gid, AllocT& alloc) {
    GroupMetadata meta(gid, alloc);
    return container->PutGroup(gid, meta);
}

bool HostPutDataset(ContainerT* container, DatasetId did, AllocT& alloc) {
    // Build a minimal 1-D shape: dims={100}, chunk_dims={10}
    DatasetShape shape;
    shape.ndims_        = 1;
    shape.dims[0]       = 100;
    shape.chunk_dims[0] = 10;
    for (uint8_t i = 1; i < MAX_DATASET_DIMS; ++i) {
        shape.dims[i]       = 0;
        shape.chunk_dims[i] = 0;
    }

    DatatypeRef dtype{PrimitiveType{PrimitiveType::Kind::Float32}};
    DatasetMetadata meta{did, dtype, shape, alloc};
    return container->PutDataset(did, meta);
}

} // namespace kvhdf5::test
