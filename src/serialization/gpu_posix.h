#pragma once

#include "serialization.h"
#include "../hdf5/types.h"
#include "../iowarp/gpu_context.h"
#include "../iowarp/gpu_posix.h"

// GPU POSIX-based serializer/deserializer that wraps the IOWarp GPU POSIX interface.
// This provides a serde-compatible interface for reading/writing files through
// the GPU-CPU cooperative I/O system.
//
// Uses a CPU-accessible staging buffer for all I/O operations because:
// - The CPU polling thread calls pread/pwrite which need CPU-accessible memory
// - Deserialization often uses GPU stack variables as destinations
// - GPU stack memory is not accessible from CPU
//
// Usage:
//   GpuPosixReaderWriter io(file_link->fd, file_link->ctx);
//   io.SetPosition(offset);
//   auto data = serde::Read<Chunk>(io);
//
// Thread Safety:
//   Each GpuPosixReaderWriter instance is NOT thread-safe (position is shared state).
//   For multi-threaded GPU code, create a separate instance per thread.
class GpuPosixReaderWriter {
public:
  // GpuContext must remain valid for the lifetime of this object
  // ctx must have a valid staging_buffer_ set
  __device__ __host__
  GpuPosixReaderWriter(int fd, iowarp::GpuContext* ctx, offset_t position = 0)
    : fd_(fd), ctx_(ctx), position_(position) {}

  __device__
  void WriteBuffer(cstd::span<const byte_t> data) {
    ASSERT(!ctx_->staging_buffer_.empty(), "staging buffer required for GPU I/O");

    size_t remaining = data.size();
    size_t offset = 0;

    while (remaining > 0) {
      size_t chunk_size = remaining > ctx_->staging_buffer_.size()
                        ? ctx_->staging_buffer_.size()
                        : remaining;

      // Create spans for the chunk
      auto src_chunk = data.subspan(offset, chunk_size);
      auto staging = cstd::span<byte_t>(
        reinterpret_cast<byte_t*>(ctx_->staging_buffer_.data()),
        chunk_size
      );

      // Copy from source to staging buffer (GPU-side copy)
      cstd::_copy(src_chunk.begin(), src_chunk.end(), staging.begin());

      // Write from staging buffer to file (CPU does the actual write)
      ssize_t written = iowarp::gpu_posix::pwrite(
        fd_,
        ctx_->staging_buffer_.data(),
        chunk_size,
        position_,
        *ctx_
      );

      ASSERT(written == static_cast<ssize_t>(chunk_size), "failed to write all bytes");

      position_ += chunk_size;
      offset += chunk_size;
      remaining -= chunk_size;
    }
  }

  __device__
  void ReadBuffer(cstd::span<byte_t> out) {
    ASSERT(!ctx_->staging_buffer_.empty(), "staging buffer required for GPU I/O");

    size_t remaining = out.size();
    size_t offset = 0;

    while (remaining > 0) {
      size_t chunk_size = remaining > ctx_->staging_buffer_.size()
                        ? ctx_->staging_buffer_.size()
                        : remaining;

      // Read from file into staging buffer (CPU does the actual read)
      ssize_t read_bytes = iowarp::gpu_posix::pread(
        fd_,
        ctx_->staging_buffer_.data(),
        chunk_size,
        position_,
        *ctx_
      );

      if (read_bytes != static_cast<ssize_t>(chunk_size)) {
        printf("[GpuPosixRW] Read failed: expected %zu bytes, got %zd\n", chunk_size, read_bytes);
        ASSERT(false, "failed to read all bytes");
      }

      // Create spans for the chunk
      auto staging = cstd::span<byte_t>(
        reinterpret_cast<byte_t*>(ctx_->staging_buffer_.data()),
        chunk_size
      );
      auto dest_chunk = out.subspan(offset, chunk_size);

      // Copy from staging buffer to destination (GPU-side copy)
      cstd::_copy(staging.begin(), staging.end(), dest_chunk.begin());

      position_ += chunk_size;
      offset += chunk_size;
      remaining -= chunk_size;
    }
  }

  // Get the current file position
  __device__ __host__
  [[nodiscard]] offset_t GetPosition() const {
    return position_;
  }

  // Set the current file position for the next read/write operation
  __device__ __host__
  void SetPosition(offset_t offset) {
    position_ = offset;
  }

  // Get the allocator from the GPU context for dynamic allocations
  __device__ __host__
  [[nodiscard]] hdf5::HdfAllocator* GetAllocator() const {
    return ctx_->allocator_;
  }

private:
  int fd_;
  iowarp::GpuContext* ctx_;
  offset_t position_;
};

static_assert(serde::Serializer<GpuPosixReaderWriter> && serde::Deserializer<GpuPosixReaderWriter>);
static_assert(iowarp::ProvidesAllocator<GpuPosixReaderWriter>);
