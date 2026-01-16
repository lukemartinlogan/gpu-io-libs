#pragma once

#include "../hdf5/cstd.h"
#include "../hdf5/gpu_allocator.h"
#include "shm_queue.h"
#include "../hdf5/types.h"
#include "hermes_shm/memory/backend/array_backend.h"

#include <cuda_runtime.h>

namespace iowarp {

struct FileEntry {
  int fd;
  cstd::atomic<offset_t> eof_offset;
  bool is_open;

  __device__ __host__
  FileEntry() : fd(-1), eof_offset(0), is_open(false) {}
};

// GPU execution context for I/O operations
// Simplified to track only one file at a time
struct GpuContext {
  shm_queue* queue_;
  FileEntry file_entry_;

  __device__ __host__
  GpuContext() : queue_(nullptr) {}

  __device__ __host__
  [[nodiscard]] offset_t GetEOF(int fd) const {
    if (file_entry_.fd != fd || !file_entry_.is_open) return 0;
    return file_entry_.eof_offset.load(cstd::memory_order_acquire);
  }

  __device__ __host__
  void UpdateEOF(int fd, size_t new_eof) {
    if (file_entry_.fd != fd || !file_entry_.is_open) return;

    offset_t current_eof = file_entry_.eof_offset.load(cstd::memory_order_acquire);
    while (new_eof > current_eof) {
      if (file_entry_.eof_offset.compare_exchange_weak(
          current_eof, new_eof,
          cstd::memory_order_release,
          cstd::memory_order_acquire
        )
      ) {
        break;
      }
    }
  }
};

} // namespace iowarp
