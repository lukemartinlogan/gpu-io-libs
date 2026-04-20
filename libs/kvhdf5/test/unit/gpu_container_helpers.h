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

} // namespace kvhdf5::test
