#include "kvhdf5/container.h"
#include "kvhdf5/gpu_cte_blob_store.h"
#include "kvhdf5/group.h"
#include "kvhdf5/allocator.h"
#include <catch2/catch_test_macros.hpp>

__global__ void kernel_empty() { }

__global__ void kernel_touch_container(kvhdf5::Container<kvhdf5::GpuCteBlobStore>* c) {
    (void)c;
}

__global__ void kernel_make_group_meta(kvhdf5::AllocatorImpl* alloc) {
    kvhdf5::GroupMetadata meta(kvhdf5::GroupId(uint64_t{1}), *alloc);
    (void)meta;
}

__global__ void kernel_put_group(
    kvhdf5::Container<kvhdf5::GpuCteBlobStore>* c,
    kvhdf5::AllocatorImpl* alloc)
{
    kvhdf5::GroupId gid(uint64_t{1});
    kvhdf5::GroupMetadata meta(gid, *alloc);
    c->PutGroup(gid, meta);
}

TEST_CASE("clang18 repro step 5", "[clang18_step5]") { REQUIRE(true); }
