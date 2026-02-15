#pragma once

#include "../hdf5/cstd.h"
#include "io_message.h"

namespace iowarp {

class shm_queue {
private:
  cstd::atomic<uint64_t> count_; // 0 = empty, 1 = has message
  IoMessage message_;  // Always present, valid when count_ > 0

public:
  __device__ __host__
  shm_queue() : count_(0), message_() {}

  __device__ __host__
  [[nodiscard]] uint64_t size() const {
    return count_.load(cstd::memory_order_seq_cst);
  }

  __device__ __host__
  [[nodiscard]] IoMessage& get() {
    return message_;
  }

  __device__ __host__
  [[nodiscard]] const IoMessage& get() const {
    return message_;
  }

  __device__ __host__
  void post(const IoMessage& msg) {
    message_ = msg;
#ifdef __CUDA_ARCH__
    __threadfence_system();  // Ensure message is visible to CPU
#endif
    count_.store(1, cstd::memory_order_seq_cst);
  }

  __device__ __host__
  void clear() {
    count_.store(0, cstd::memory_order_seq_cst);
  }
};

} // namespace iowarp
