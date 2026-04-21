#pragma once

#include "../defines.h"
#include <cuda/std/span>
#include <cuda/std/cstring>
#include <cuda/std/cstdint>
#include <cuda/std/climits>
#include <cuda/__memory/check_address.h>

namespace algorithms {

// Non-null, non-overflowing pointer-range validation. A reimplementation of
// cuda::__is_valid_address_range minus the device-only shared-memory
// address-space probe that triggers a clang-18 / cicc NVPTX codegen bug.
//
// The bug: under CUDA compilation with clang-18 and -DCCCL_ENABLE_ASSERTIONS,
// CCCL's cuda::__is_valid_address_range runs an __isShared(ptr) check inside
// an NV_IF_TARGET(NV_IS_DEVICE, ...) branch. That intrinsic lowers to
// llvm.nvvm.isspacep.shared, whose LLVM signature requires an i8* operand.
// When the input pointer has a non-i8* type (e.g., a struct T* loaded from
// a stack-resident container field in addrspace(5)), the NVPTX backend emits
// the call with the typed pointer instead of bitcasting to i8* first. The IR
// verifier rejects it and aborts compilation with "Broken function found".
// Every entry point that routes such a pointer into a cstd::memcpy (which
// gates its assertions on __is_valid_address_range) is affected. We preserve
// the other three validity checks (non-zero size, no pointer-overflow,
// non-null) and drop only the shared-memory probe.
CROSS_FUN inline bool __safe_valid_address_range(const void* p, size_t n) {
    if (n == 0) return true;  // zero-length ranges are a no-op, not an error
    const auto limit = cuda::std::uintptr_t{UINTMAX_MAX} - static_cast<cuda::std::uintptr_t>(n);
    if (reinterpret_cast<cuda::std::uintptr_t>(p) > limit) return false;
    return p != nullptr;
}

// Copy between typed spans. Mirrors the structure of CCCL's cstd::memcpy
// debug-assertion chain: source range valid, destination range valid,
// source and destination do not overlap, then the raw ::memcpy. The one
// check that would normally be here but is omitted — address-space
// validation via __isShared — is what provokes the clang-18 codegen bug
// described above on __safe_valid_address_range. Empty copies short-
// circuit before any validation.
template<typename T>
CROSS_FUN void copy(cstd::span<const T> src, cstd::span<T> dst) {
    KVHDF5_ASSERT(dst.size() >= src.size(), "Destination span too small for copy");
    const size_t n = src.size() * sizeof(T);
    if (n == 0) return;
    KVHDF5_ASSERT(__safe_valid_address_range(src.data(), n),  "copy: source range is invalid");
    KVHDF5_ASSERT(__safe_valid_address_range(dst.data(), n),  "copy: destination range is invalid");
    KVHDF5_ASSERT(!::cuda::__are_ptrs_overlapping(src.data(), dst.data(), n),
                  "copy: source and destination overlap");
    ::memcpy(dst.data(), src.data(), n);
}

// Overload for non-const source (delegates to const version)
template<typename T>
CROSS_FUN void copy(cstd::span<T> src, cstd::span<T> dst) {
    copy(cstd::span<const T>(src.data(), src.size()), dst);
}

} // namespace algorithms
