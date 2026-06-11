#pragma once

// Host-only helpers for the Gray-Scott example.
//
// All DatasetMetadata / GroupMetadata construction lives in this plain CXX
// translation unit so it is never visible to clang-18's NVPTX device codegen
// pass. See test/unit/gpu_container_helpers.h for the long-form rationale —
// the short version is that hshm::priv::vector<T> members on those types
// emit NVPTX IR that ptxas 12.x rejects.

#include "kvhdf5/container.h"
#include "kvhdf5/gpu_cte_blob_store.h"

#include <vector>

namespace gs {

using ContainerT = kvhdf5::Container<kvhdf5::GpuCteBlobStore>;
using AllocT     = kvhdf5::AllocatorImpl;

// IDs the kernel needs in order to run a step or take a snapshot. The ping-
// pong layout uses two pairs (u_a/u_b, v_a/v_b) and the host alternates which
// is "curr" each step. Snapshot datasets are *fully created on the host*
// up front (PutDataset of an empty grid metadata blob with the same shape /
// dtype as u/v); the snapshot kernel only writes chunk data into them.
//
// We previously tried option (b) — kernel-side metadata creation by cloning
// a template DatasetMetadata — and hit the iowarp-core "kernel-allocated
// ShmPtr in PutBlobTask handler" issue tracked by the [alloc_buffer] test
// in test_gpu_kernel.cu. Once that upstream fix lands we can revisit.
struct GrayScottIds {
    kvhdf5::GroupId  sim_gid;
    kvhdf5::GroupId  snap_gid;
    kvhdf5::DatasetId u_a;
    kvhdf5::DatasetId u_b;
    kvhdf5::DatasetId v_a;
    kvhdf5::DatasetId v_b;
    std::vector<kvhdf5::DatasetId> snap_u_ids;
    std::vector<kvhdf5::DatasetId> snap_v_ids;
    std::vector<int>               snap_steps;
};

// Build /sim and /snapshots groups under root, create u_a/u_b/v_a/v_b and the
// two template datasets, pre-allocate snapshot DatasetIds (one per entry of
// snap_steps), and link everything into the appropriate groups.
GrayScottIds HostBuildScene(ContainerT* container, AllocT& alloc,
                             unsigned grid_n,
                             const std::vector<int>& snap_steps);

// Seed initial conditions: u=1.0 everywhere, v=0.0 everywhere, with a small
// central square of u=0.5, v=0.25 to break symmetry. Writes go into u_did
// and v_did (typically the "_a" pair, which acts as iteration 0).
bool HostSeedInitialConditions(ContainerT* container,
                                kvhdf5::DatasetId u_did,
                                kvhdf5::DatasetId v_did,
                                unsigned grid_n);

// Read a full grid back to host (SelectAll). Used for the final readback +
// snapshot readback when printing the ASCII heatmap.
bool HostReadGrid(ContainerT* container, kvhdf5::DatasetId did,
                  float* out, unsigned grid_n);

// Host-side snapshot: read u_curr / v_curr through the host Dataset API and
// write the same bytes to snap_u / snap_v. Used in place of a kernel-side
// snapshot to avoid the intermittent kernel-side metadata-roundtrip edge
// case noted in the example comments.
bool HostTakeSnapshot(ContainerT* container,
                      kvhdf5::DatasetId u_curr, kvhdf5::DatasetId v_curr,
                      kvhdf5::DatasetId snap_u, kvhdf5::DatasetId snap_v,
                      unsigned grid_n);

} // namespace gs
