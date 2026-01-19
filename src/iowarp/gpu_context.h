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
  hdf5::HdfAllocator* allocator_;
  cstd::span<char> staging_buffer_;
  FileEntry file_entry_;

  __device__ __host__
  GpuContext() : queue_(nullptr), allocator_(nullptr), staging_buffer_() {}

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

class GpuContextBuilder {
public:
  static constexpr size_t kDefaultStagingSize = 64 * 1024;  // 64KB
  static constexpr size_t kDefaultHeapSize = 1024 * 1024;   // 1MB

  GpuContextBuilder() = default;
  ~GpuContextBuilder() { Destroy(); }

  // owns CUDA resources, so shouldn't be moved nor copied
  GpuContextBuilder(const GpuContextBuilder&) = delete;
  GpuContextBuilder& operator=(const GpuContextBuilder&) = delete;
  GpuContextBuilder(GpuContextBuilder&&) = delete;
  GpuContextBuilder& operator=(GpuContextBuilder&&) = delete;

  bool Build() {
    return Build(kDefaultStagingSize, kDefaultHeapSize);
  }

  bool Build(size_t staging_size, size_t heap_size) {
    if (built_) return false;

    if (cudaHostAlloc(&h_queue_, sizeof(shm_queue), cudaHostAllocMapped) != cudaSuccess) {
      return false;
    }
    cudaHostGetDevicePointer(&d_queue_, h_queue_, 0);
    new (h_queue_) shm_queue();

    if (cudaHostAlloc(&h_ctx_, sizeof(GpuContext), cudaHostAllocMapped) != cudaSuccess) {
      Destroy();
      return false;
    }

    cudaHostGetDevicePointer(&d_ctx_, h_ctx_, 0);
    new (h_ctx_) GpuContext();
    h_ctx_->queue_ = d_queue_;

    staging_size_ = staging_size;
    if (cudaHostAlloc(&h_staging_, staging_size_, cudaHostAllocMapped) != cudaSuccess) {
      Destroy();
      return false;
    }

    cudaHostGetDevicePointer(&d_staging_, h_staging_, 0);
    h_ctx_->staging_buffer_ = cstd::span<char>(d_staging_, staging_size_);

    size_t alloc_size = heap_size + 3 * hshm::ipc::kBackendHeaderSize;
    if (cudaHostAlloc(&h_alloc_mem_, alloc_size, cudaHostAllocMapped) != cudaSuccess) {
      Destroy();
      return false;
    }
    cudaHostGetDevicePointer(&d_alloc_mem_, h_alloc_mem_, 0);

    // Zero-initialize BEFORE shm_init (which sets up allocator headers)
    memset(h_alloc_mem_, 0, alloc_size);

    // Initialize backend with device pointer so all internal pointers
    // are in device address space (accessible from both CPU and GPU)
    if (!backend_.shm_init(hshm::ipc::MemoryBackendId::GetRoot(), alloc_size, d_alloc_mem_)) {
      Destroy();
      return false;
    }

    h_allocator_ = backend_.MakeAlloc<hdf5::HdfAllocator>();
    // Since backend was initialized with d_alloc_mem_, h_allocator_ is already
    // a device pointer - just use it directly
    d_allocator_ = h_allocator_;
    h_ctx_->allocator_ = d_allocator_;

    printf("[GpuContextBuilder] h_alloc_mem=%p, d_alloc_mem=%p\n", h_alloc_mem_, d_alloc_mem_);
    printf("[GpuContextBuilder] h_allocator=%p, d_allocator=%p\n", h_allocator_, d_allocator_);
    printf("[GpuContextBuilder] backend.data_=%p\n", backend_.data_);

    // Ensure GPU can see all allocator initialization writes
    cudaDeviceSynchronize();

    built_ = true;
    return true;
  }

  void Destroy() {
    if (h_alloc_mem_) { cudaFreeHost(h_alloc_mem_); h_alloc_mem_ = nullptr; }
    if (h_staging_) { cudaFreeHost(h_staging_); h_staging_ = nullptr; }
    if (h_ctx_) { h_ctx_->~GpuContext(); cudaFreeHost(h_ctx_); h_ctx_ = nullptr; }
    if (h_queue_) { h_queue_->~shm_queue(); cudaFreeHost(h_queue_); h_queue_ = nullptr; }
    built_ = false;
  }

  [[nodiscard]] GpuContext* DeviceContext() const { return d_ctx_; }

  [[nodiscard]] GpuContext* HostContext() const { return h_ctx_; }

  [[nodiscard]] shm_queue* HostQueue() const { return h_queue_; }

  [[nodiscard]] bool IsBuilt() const { return built_; }

  [[nodiscard]] hdf5::HdfAllocator* DeviceAllocator() const { return d_allocator_; }

  [[nodiscard]] char* DeviceAllocMem() const { return d_alloc_mem_; }

  // Get the offset from alloc_mem to data_ (where allocator is placed)
  [[nodiscard]] size_t DataOffset() const { return 3 * hshm::ipc::kBackendHeaderSize; }

private:
  bool built_ = false;

  // queue
  shm_queue* h_queue_ = nullptr;
  shm_queue* d_queue_ = nullptr;

  // context
  GpuContext* h_ctx_ = nullptr;
  GpuContext* d_ctx_ = nullptr;

  // staging buffer
  char* h_staging_ = nullptr;
  char* d_staging_ = nullptr;
  size_t staging_size_ = 0;

  // allocator
  char* h_alloc_mem_ = nullptr;
  char* d_alloc_mem_ = nullptr;
  hshm::ipc::ArrayBackend backend_;
  hdf5::HdfAllocator* h_allocator_ = nullptr;
  hdf5::HdfAllocator* d_allocator_ = nullptr;
};

} // namespace iowarp
