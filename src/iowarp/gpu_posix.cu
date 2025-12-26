#include "gpu_posix.h"
#include <cuda_runtime.h>

namespace iowarp::gpu_posix {

__device__ int open(const char* filename, int flags, int mode, GpuContext& ctx) {
  // TODO: ideally we probably shouldn't have a const_cast here; needs refactor
  printf("[GPU POSIX] open() called for file: %s\n", filename);

  printf("[GPU POSIX] Posting open message to queue\n");
  ctx.queue_->post(IoMessage::Open(const_cast<char*>(filename), flags, mode));
  printf("[GPU POSIX] Message posted, queue size: %llu\n", ctx.queue_->size());

  printf("[GPU POSIX] Waiting for CPU to process...\n");
  uint64_t wait_count = 0;
  while (ctx.queue_->size() > 0) {
    __nanosleep(500);
    wait_count++;
    if (wait_count % 1000000 == 0) {
      printf("[GPU POSIX] Still waiting... (count: %llu, queue size: %llu)\n", wait_count, ctx.queue_->size());
    }
  }
  printf("[GPU POSIX] CPU finished processing, queue cleared\n");

  auto& msg = ctx.queue_->get();
  int fd = msg.fd;
  printf("[GPU POSIX] open() returned fd: %d\n", fd);

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
  printf("[GPU POSIX] pread() called: fd=%d, size=%zu, offset=%zu\n", fd, size, offset);
  ctx.queue_->post(IoMessage::Read(fd, static_cast<char*>(buffer), size, offset));

  while (ctx.queue_->size() > 0) {
    __nanosleep(500);
  }

  auto& msg = ctx.queue_->get();
  printf("[GPU POSIX] pread() returned: %zd bytes\n", msg.result_);
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
