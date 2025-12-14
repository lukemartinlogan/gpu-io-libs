#include "gpu_posix.h"
#include <cuda_runtime.h>

namespace iowarp::gpu_posix {

__device__ int open(const char* filename, int flags, int mode, GpuContext& ctx) {
  // TODO: ideally we probably shouldn't have a const_cast here; needs refactor
  ctx.queue_->post(IoMessage::Open(const_cast<char*>(filename), flags, mode));

  while (ctx.queue_->size() > 0) {
    __nanosleep(500);
  }

  auto& msg = ctx.queue_->get();
  int fd = msg.fd;

  ctx.file_entry_.fd = fd;
  ctx.file_entry_.is_open = true;
  ctx.file_entry_.eof_offset.store(0, cstd::memory_order_release);

  return fd;
}

__device__ ssize_t pwrite(int fd, const void* buffer, size_t size, size_t offset, GpuContext& ctx) {
  ctx.queue_->post(IoMessage::Write(fd, const_cast<char*>(static_cast<const char*>(buffer)), size, offset));

  while (ctx.queue_->size() > 0) {
    __nanosleep(500);
  }

  auto& msg = ctx.queue_->get();
  ssize_t result = msg.result_;

  if (result > 0) {
    size_t new_eof = offset + result;
    ctx.UpdateEOF(fd, new_eof);
  }

  return result;
}

__device__ ssize_t pread(int fd, void* buffer, size_t size, size_t offset, GpuContext& ctx) {
  ctx.queue_->post(IoMessage::Read(fd, static_cast<char*>(buffer), size, offset));

  while (ctx.queue_->size() > 0) {
    __nanosleep(500);
  }

  auto& msg = ctx.queue_->get();
  return msg.result_;
}

__device__ int close(int fd, GpuContext& ctx) {
  ctx.queue_->post(IoMessage::Close(fd));

  while (ctx.queue_->size() > 0) {
    __nanosleep(500);
  }

  ctx.file_entry_.fd = -1;
  ctx.file_entry_.is_open = false;
  ctx.file_entry_.eof_offset.store(0, cstd::memory_order_release);

  auto& msg = ctx.queue_->get();
  return msg.result_;
}

} // namespace iowarp::gpu_posix
