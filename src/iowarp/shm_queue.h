#pragma once

#include "../hdf5/cstd.h"
#include "io_message.h"

namespace iowarp {

class shm_queue {
private:
  cstd::atomic<uint64_t> count_; // 0 = empty, 1 = has message
  cstd::optional<IoMessage> message_;

public:
  __device__ __host__
  shm_queue() : count_(0) {}

  __device__ __host__
  [[nodiscard]] uint64_t size() const {
    return count_.load(cstd::memory_order_acquire);
  }

  __device__ __host__
  [[nodiscard]] cstd::optional<IoMessage>& get() {
    return message_;
  }

  __device__ __host__
  [[nodiscard]] const cstd::optional<IoMessage>& get() const {
    return message_;
  }

  __device__ __host__
  void post(const IoMessage& msg) {
    message_ = msg;
    count_.store(1, cstd::memory_order_release);
  }

  __device__ __host__
  void clear() {
    message_.reset();
    count_.store(0, cstd::memory_order_release);
  }
};

} // namespace iowarp
