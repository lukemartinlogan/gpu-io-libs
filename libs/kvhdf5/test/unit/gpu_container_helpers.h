#pragma once

// ---------------------------------------------------------------------------
// gpu_container_helpers.h — Host-only helpers for gpu_container_test.cu.
//
// All functions here are compiled in a plain CXX translation unit
// (gpu_container_helpers.cc) so that GroupMetadata / DatasetMetadata
// construction and PutGroup / PutDataset never appear in a .cu file.
//
// Why this matters:
//   GroupMetadata and DatasetMetadata hold hshm::priv::vector<T> members.
//   Their constructors and Serialize methods are annotated CROSS_FUN
//   (__host__ __device__).  When clang-18 compiles a .cu file that calls
//   these functions (even in host-only #if !HSHM_IS_GPU blocks), it
//   device-instantiates them.  The resulting NVPTX IR contains
//   llvm.nvvm.isspacep.shared with a typed pointer (GroupEntry* / Attribute*)
//   instead of i8*, which ptxas 12.x rejects:
//     "Call parameter type does not match function signature /
//      Broken function found, compilation aborted!"
//
//   By defining all GroupMetadata / DatasetMetadata usage in a plain .cc file,
//   clang-18's device codegen never sees those types and the ptxas bug is
//   avoided entirely.
// ---------------------------------------------------------------------------

#include "kvhdf5/container.h"
#include "kvhdf5/gpu_cte_blob_store.h"

namespace kvhdf5::test {

using ContainerT = Container<GpuCteBlobStore>;
using AllocT     = AllocatorImpl;

/// Put an empty GroupMetadata for \p gid into \p container on the host.
/// Returns true on success.
bool HostPutGroup(ContainerT* container, GroupId gid, AllocT& alloc);

/// Put an empty DatasetMetadata for \p did into \p container on the host.
/// Returns true on success.
bool HostPutDataset(ContainerT* container, DatasetId did, AllocT& alloc);

/// Allocate a dataset id and put a DatasetMetadata with the given shape and
/// primitive datatype kind. Used by gpu_dataset_test.cu so tests can size
/// datasets to fit small device-side chunk buffers.
DatasetId HostCreateDataset(ContainerT* container, AllocT& alloc,
                             cstd::span<const uint64_t> dims,
                             cstd::span<const uint64_t> chunk_dims,
                             PrimitiveType::Kind kind);

/// Host-side equivalent of Group::CreateGroup: allocates a child group,
/// stores its (empty) GroupMetadata, and pushes a GroupEntry into \p parent_gid.
/// Returns the new child GroupId.
GroupId HostAddChildGroup(ContainerT* container, GroupId parent_gid,
                          const char* name, AllocT& alloc);

/// Host-side equivalent of Group::SetAttribute for an int32 value. Reads the
/// group meta, pushes/replaces an Attribute on \p gid, writes back.
bool HostAddInt32Attribute(ContainerT* container, GroupId gid,
                           const char* name, int32_t value, AllocT& alloc);

/// Look up a child group of \p parent_gid by name. Returns true on success
/// and writes the resolved GroupId to \p out.
bool HostFindChildGroup(ContainerT* container, GroupId parent_gid,
                        const char* name, GroupId* out);

/// Look up a child dataset of \p parent_gid by name. Returns true on success
/// and writes the resolved DatasetId to \p out.
bool HostFindChildDataset(ContainerT* container, GroupId parent_gid,
                          const char* name, DatasetId* out);

/// Read a SelectAll dataset of float32 elements into \p buf. \p num_elems must
/// match the dataset's element count.
bool HostReadFloat32Dataset(ContainerT* container, DatasetId did,
                            float* buf, size_t num_elems);

/// Read a SelectAll dataset of int32 elements into \p buf. \p num_elems must
/// match the dataset's element count.
bool HostReadInt32Dataset(ContainerT* container, DatasetId did,
                          int32_t* buf, size_t num_elems);

/// Host-side equivalent of Group::GetAttribute for an int32 value. Returns
/// true on success and writes the value to \p out.
bool HostGetInt32Attribute(ContainerT* container, GroupId gid,
                           const char* name, int32_t* out);

} // namespace kvhdf5::test
