#pragma once

#include "gpu_context.h"

namespace iowarp::gpu_posix {
__device__ int open(const char* filename, int flags, int mode, GpuContext& ctx);

__device__ uint32_t pwrite(int fd, const void* buffer, size_t size, size_t offset, GpuContext& ctx);

__device__ uint32_t pread(int fd, void* buffer, size_t size, size_t offset, GpuContext& ctx);

__device__ int close(int fd, GpuContext& ctx);
} // namespace iowarp::gpu_posix
