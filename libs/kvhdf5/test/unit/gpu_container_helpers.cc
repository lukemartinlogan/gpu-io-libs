// gpu_container_helpers.cc — Plain CXX translation unit.
//
// All GroupMetadata / DatasetMetadata construction and PutGroup / PutDataset
// calls live here so they are never seen by clang-18's NVPTX device codegen.
// See gpu_container_helpers.h for the rationale.

#include "gpu_container_helpers.h"
#include "kvhdf5/group.h"
#include "kvhdf5/dataset.h"
#include "kvhdf5/datatype.h"
#include "kvhdf5/hdf5_dataset.h"
#include "kvhdf5/hdf5_datatype.h"
#include "kvhdf5/dataspace.h"
#include "kvhdf5/ref.h"

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

DatasetId HostCreateDataset(ContainerT* container, AllocT& alloc,
                             cstd::span<const uint64_t> dims,
                             cstd::span<const uint64_t> chunk_dims,
                             PrimitiveType::Kind kind) {
    KVHDF5_ASSERT(dims.size() == chunk_dims.size(),
                  "dims and chunk_dims must have same rank");
    KVHDF5_ASSERT(dims.size() <= MAX_DATASET_DIMS,
                  "rank exceeds MAX_DATASET_DIMS");

    DatasetShape shape;
    shape.ndims_ = static_cast<uint8_t>(dims.size());
    for (uint8_t i = 0; i < shape.ndims_; ++i) {
        shape.dims[i]       = dims[i];
        shape.chunk_dims[i] = chunk_dims[i];
    }
    for (uint8_t i = shape.ndims_; i < MAX_DATASET_DIMS; ++i) {
        shape.dims[i]       = 0;
        shape.chunk_dims[i] = 0;
    }

    DatasetId did(container->AllocateId());
    DatatypeRef dtype{PrimitiveType{kind}};
    DatasetMetadata meta{did, dtype, shape, alloc};
    bool ok = container->PutDataset(did, meta);
    KVHDF5_ASSERT(ok, "PutDataset failed");
    return did;
}

GroupId HostAddChildGroup(ContainerT* container, GroupId parent_gid,
                          const char* name, AllocT& alloc) {
    auto parent_r = container->GetGroup(parent_gid);
    KVHDF5_ASSERT(parent_r.has_value(), "HostAddChildGroup: parent not found");
    auto parent_meta = parent_r.value();

    GroupId child_id(container->AllocateId());
    GroupMetadata child_meta(child_id, alloc);
    bool ok = container->PutGroup(child_id, child_meta);
    KVHDF5_ASSERT(ok, "PutGroup(child) failed");

    parent_meta.children.push_back(
        GroupEntry::NewGroup(child_id, gpu_string_view(name)));
    ok = container->PutGroup(parent_gid, parent_meta);
    KVHDF5_ASSERT(ok, "PutGroup(parent) failed");
    return child_id;
}

bool HostFindChildGroup(ContainerT* container, GroupId parent_gid,
                        const char* name, GroupId* out) {
    auto meta_r = container->GetGroup(parent_gid);
    if (!meta_r.has_value()) return false;
    auto& meta = meta_r.value();
    gpu_string_view want(name);
    for (size_t i = 0; i < meta.children.size(); ++i) {
        if (meta.children[i].kind == ChildKind::Group &&
            meta.children[i].name == want) {
            *out = GroupId(meta.children[i].object_id);
            return true;
        }
    }
    return false;
}

bool HostFindChildDataset(ContainerT* container, GroupId parent_gid,
                          const char* name, DatasetId* out) {
    auto meta_r = container->GetGroup(parent_gid);
    if (!meta_r.has_value()) return false;
    auto& meta = meta_r.value();
    gpu_string_view want(name);
    for (size_t i = 0; i < meta.children.size(); ++i) {
        if (meta.children[i].kind == ChildKind::Dataset &&
            meta.children[i].name == want) {
            *out = DatasetId(meta.children[i].object_id);
            return true;
        }
    }
    return false;
}

bool HostReadFloat32Dataset(ContainerT* container, DatasetId did,
                            float* buf, size_t num_elems) {
    Dataset<GpuCteBlobStore> ds(did, Ref<ContainerT>(*container));
    uint64_t dims[1] = { static_cast<uint64_t>(num_elems) };
    auto sp_r = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    if (!sp_r.has_value()) return false;
    auto& sp = sp_r.value();
    auto r = ds.Read(Datatype::Float32(), sp, sp, buf);
    return r.has_value();
}

bool HostReadInt32Dataset(ContainerT* container, DatasetId did,
                          int32_t* buf, size_t num_elems) {
    Dataset<GpuCteBlobStore> ds(did, Ref<ContainerT>(*container));
    uint64_t dims[1] = { static_cast<uint64_t>(num_elems) };
    auto sp_r = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    if (!sp_r.has_value()) return false;
    auto& sp = sp_r.value();
    auto r = ds.Read(Datatype::Int32(), sp, sp, buf);
    return r.has_value();
}

bool HostGetInt32Attribute(ContainerT* container, GroupId gid,
                           const char* name, int32_t* out) {
    auto meta_r = container->GetGroup(gid);
    if (!meta_r.has_value()) return false;
    auto& meta = meta_r.value();
    gpu_string_view want(name);
    for (size_t i = 0; i < meta.attributes.size(); ++i) {
        if (meta.attributes[i].name == want) {
            auto& val = meta.attributes[i].value;
            if (val.size() < sizeof(int32_t)) return false;
            int32_t v;
            auto* dst = reinterpret_cast<byte_t*>(&v);
            for (size_t j = 0; j < sizeof(int32_t); ++j) dst[j] = val[j];
            *out = v;
            return true;
        }
    }
    return false;
}

bool HostAddInt32Attribute(ContainerT* container, GroupId gid,
                           const char* name, int32_t value, AllocT& /*alloc*/) {
    auto meta_r = container->GetGroup(gid);
    if (!meta_r.has_value()) return false;
    auto meta = meta_r.value();

    DatatypeRef dtype{PrimitiveType{PrimitiveType::Kind::Int32}};
    cstd::span<const byte_t> val_span(
        reinterpret_cast<const byte_t*>(&value), sizeof(int32_t));
    Attribute attr(gpu_string_view(name), dtype, val_span);

    bool found = false;
    for (size_t i = 0; i < meta.attributes.size(); ++i) {
        if (meta.attributes[i].name == gpu_string_view(name)) {
            meta.attributes[i] = attr;
            found = true;
            break;
        }
    }
    if (!found) meta.attributes.push_back(attr);

    return container->PutGroup(gid, meta);
}

} // namespace kvhdf5::test
