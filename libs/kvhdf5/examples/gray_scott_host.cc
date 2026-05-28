// Host-only helpers for the Gray-Scott example. See gray_scott_host.h.

#include "gray_scott_host.h"

#include "kvhdf5/group.h"
#include "kvhdf5/dataset.h"
#include "kvhdf5/datatype.h"
#include "kvhdf5/dataspace.h"
#include "kvhdf5/hdf5_dataset.h"
#include "kvhdf5/hdf5_datatype.h"
#include "kvhdf5/ref.h"

#include <algorithm>
#include <cstdio>

namespace gs {

namespace {

kvhdf5::DatasetId CreateGridDataset(ContainerT* c, AllocT& alloc, unsigned n) {
    // Single-chunk 2-D float32 dataset. chunk_dims = dims so each Read/Write
    // touches exactly one chunk — minimizes per-call stack scratch.
    kvhdf5::DatasetShape shape;
    shape.ndims_        = 2;
    shape.dims[0]       = n;
    shape.dims[1]       = n;
    shape.chunk_dims[0] = n;
    shape.chunk_dims[1] = n;
    for (uint8_t i = 2; i < kvhdf5::MAX_DATASET_DIMS; ++i) {
        shape.dims[i]       = 0;
        shape.chunk_dims[i] = 0;
    }

    kvhdf5::DatasetId   did(c->AllocateId());
    kvhdf5::DatatypeRef dt{kvhdf5::PrimitiveType{kvhdf5::PrimitiveType::Kind::Float32}};
    kvhdf5::DatasetMetadata meta{did, dt, shape, alloc};
    KVHDF5_ASSERT(c->PutDataset(did, meta), "PutDataset failed");
    return did;
}

void LinkDataset(ContainerT* c, kvhdf5::GroupId gid,
                 kvhdf5::DatasetId did, const char* name) {
    auto meta_r = c->GetGroup(gid);
    KVHDF5_ASSERT(meta_r.has_value(), "LinkDataset: parent group missing");
    auto meta = meta_r.value();
    meta.children.push_back(
        kvhdf5::GroupEntry::NewDataset(did, kvhdf5::gpu_string_view(name)));
    KVHDF5_ASSERT(c->PutGroup(gid, meta), "LinkDataset: PutGroup failed");
}

kvhdf5::GroupId AddChildGroup(ContainerT* c, AllocT& alloc,
                               kvhdf5::GroupId parent, const char* name) {
    kvhdf5::GroupId child(c->AllocateId());
    {
        kvhdf5::GroupMetadata m(child, alloc);
        KVHDF5_ASSERT(c->PutGroup(child, m), "AddChildGroup: PutGroup(child) failed");
    }
    auto pmeta_r = c->GetGroup(parent);
    KVHDF5_ASSERT(pmeta_r.has_value(), "AddChildGroup: parent missing");
    auto pmeta = pmeta_r.value();
    pmeta.children.push_back(
        kvhdf5::GroupEntry::NewGroup(child, kvhdf5::gpu_string_view(name)));
    KVHDF5_ASSERT(c->PutGroup(parent, pmeta),
                  "AddChildGroup: PutGroup(parent) failed");
    return child;
}

} // namespace

GrayScottIds HostBuildScene(ContainerT* c, AllocT& alloc, unsigned n,
                             const std::vector<int>& snap_steps) {
    GrayScottIds ids;

    kvhdf5::GroupId root = c->RootGroup();

    ids.sim_gid  = AddChildGroup(c, alloc, root, "sim");
    ids.snap_gid = AddChildGroup(c, alloc, root, "snapshots");

    // Ping-pong grids in /sim
    ids.u_a = CreateGridDataset(c, alloc, n);
    ids.u_b = CreateGridDataset(c, alloc, n);
    ids.v_a = CreateGridDataset(c, alloc, n);
    ids.v_b = CreateGridDataset(c, alloc, n);
    LinkDataset(c, ids.sim_gid, ids.u_a, "u_a");
    LinkDataset(c, ids.sim_gid, ids.u_b, "u_b");
    LinkDataset(c, ids.sim_gid, ids.v_a, "v_a");
    LinkDataset(c, ids.sim_gid, ids.v_b, "v_b");

    // Pre-create every snapshot dataset on the host (option (a)). The kernel
    // never creates dataset metadata; it just writes chunk data into these.
    ids.snap_steps = snap_steps;
    ids.snap_u_ids.reserve(snap_steps.size());
    ids.snap_v_ids.reserve(snap_steps.size());
    char name_buf[64];
    for (size_t i = 0; i < snap_steps.size(); ++i) {
        kvhdf5::DatasetId u_id = CreateGridDataset(c, alloc, n);
        kvhdf5::DatasetId v_id = CreateGridDataset(c, alloc, n);
        ids.snap_u_ids.push_back(u_id);
        ids.snap_v_ids.push_back(v_id);

        std::snprintf(name_buf, sizeof(name_buf), "u_step_%d", snap_steps[i]);
        LinkDataset(c, ids.snap_gid, u_id, name_buf);
        std::snprintf(name_buf, sizeof(name_buf), "v_step_%d", snap_steps[i]);
        LinkDataset(c, ids.snap_gid, v_id, name_buf);
    }

    return ids;
}

bool HostSeedInitialConditions(ContainerT* c,
                                kvhdf5::DatasetId u_did,
                                kvhdf5::DatasetId v_did,
                                unsigned n) {
    std::vector<float> u(static_cast<size_t>(n) * n, 1.0f);
    std::vector<float> v(static_cast<size_t>(n) * n, 0.0f);

    int seed_half = std::max<int>(1, static_cast<int>(n) / 12);
    int cx = static_cast<int>(n) / 2;
    int cy = static_cast<int>(n) / 2;
    for (int dy = -seed_half; dy <= seed_half; ++dy) {
        for (int dx = -seed_half; dx <= seed_half; ++dx) {
            int x = cx + dx;
            int y = cy + dy;
            if (x < 0 || y < 0 || x >= static_cast<int>(n) ||
                y >= static_cast<int>(n)) continue;
            u[static_cast<size_t>(y) * n + x] = 0.5f;
            v[static_cast<size_t>(y) * n + x] = 0.25f;
        }
    }

    kvhdf5::Dataset<kvhdf5::GpuCteBlobStore> ds_u(
        u_did, kvhdf5::Ref<ContainerT>(*c));
    kvhdf5::Dataset<kvhdf5::GpuCteBlobStore> ds_v(
        v_did, kvhdf5::Ref<ContainerT>(*c));

    uint64_t dims[2] = {n, n};
    auto sp_r = kvhdf5::Dataspace::CreateSimple(
        cstd::span<const uint64_t>(dims, 2));
    if (!sp_r.has_value()) return false;
    auto& sp = sp_r.value();

    if (!ds_u.Write(kvhdf5::Datatype::Float32(), sp, sp, u.data()).has_value())
        return false;
    if (!ds_v.Write(kvhdf5::Datatype::Float32(), sp, sp, v.data()).has_value())
        return false;
    return true;
}

bool HostReadGrid(ContainerT* c, kvhdf5::DatasetId did, float* out,
                  unsigned n) {
    kvhdf5::Dataset<kvhdf5::GpuCteBlobStore> ds(
        did, kvhdf5::Ref<ContainerT>(*c));
    uint64_t dims[2] = {n, n};
    auto sp_r = kvhdf5::Dataspace::CreateSimple(
        cstd::span<const uint64_t>(dims, 2));
    if (!sp_r.has_value()) return false;
    auto& sp = sp_r.value();
    auto r = ds.Read(kvhdf5::Datatype::Float32(), sp, sp, out);
    return r.has_value();
}

bool HostTakeSnapshot(ContainerT* c,
                      kvhdf5::DatasetId u_curr, kvhdf5::DatasetId v_curr,
                      kvhdf5::DatasetId snap_u, kvhdf5::DatasetId snap_v,
                      unsigned n) {
    std::vector<float> u(static_cast<size_t>(n) * n);
    std::vector<float> v(static_cast<size_t>(n) * n);
    if (!HostReadGrid(c, u_curr, u.data(), n)) return false;
    if (!HostReadGrid(c, v_curr, v.data(), n)) return false;

    kvhdf5::Dataset<kvhdf5::GpuCteBlobStore> ds_u(
        snap_u, kvhdf5::Ref<ContainerT>(*c));
    kvhdf5::Dataset<kvhdf5::GpuCteBlobStore> ds_v(
        snap_v, kvhdf5::Ref<ContainerT>(*c));
    uint64_t dims[2] = {n, n};
    auto sp_r = kvhdf5::Dataspace::CreateSimple(
        cstd::span<const uint64_t>(dims, 2));
    if (!sp_r.has_value()) return false;
    auto& sp = sp_r.value();

    if (!ds_u.Write(kvhdf5::Datatype::Float32(), sp, sp, u.data()).has_value())
        return false;
    if (!ds_v.Write(kvhdf5::Datatype::Float32(), sp, sp, v.data()).has_value())
        return false;
    return true;
}

} // namespace gs
