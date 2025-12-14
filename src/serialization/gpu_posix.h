#pragma once

#include "serialization.h"
#include "../hdf5/types.h"
#include "../iowarp/gpu_context.h"
#include "../iowarp/gpu_posix.h"

// GPU POSIX-based serializer/deserializer that wraps the IOWarp GPU POSIX interface.
// This provides a serde-compatible interface for reading/writing files through
// the GPU-CPU cooperative I/O system.
//
// Unlike StdioReaderWriter which uses FILE* with automatic position tracking,
// this uses pread/pwrite which don't modify the file position, so we maintain
// an internal position cursor that advances with each read/write operation.
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
  __device__ __host__
  GpuPosixReaderWriter(int fd, iowarp::GpuContext* ctx, offset_t position = 0)
    : fd_(fd), ctx_(ctx), position_(position) {}

  __device__
  void WriteBuffer(cstd::span<const byte_t> data) {
    ssize_t written = iowarp::gpu_posix::pwrite(
      fd_,
      data.data(),
      data.size(),
      position_,
      *ctx_
    );

    ASSERT(written == static_cast<ssize_t>(data.size()), "failed to write all bytes");
    position_ += data.size();
  }

  __device__
  void ReadBuffer(cstd::span<byte_t> out) {
    ssize_t read_bytes = iowarp::gpu_posix::pread(
      fd_,
      out.data(),
      out.size(),
      position_,
      *ctx_
    );

    ASSERT(read_bytes == static_cast<ssize_t>(out.size()), "failed to read all bytes");
    position_ += out.size();
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

private:
  int fd_;
  iowarp::GpuContext* ctx_;
  offset_t position_;
};

static_assert(serde::Serializer<GpuPosixReaderWriter> && serde::Deserializer<GpuPosixReaderWriter>);
