#pragma once

#include "gpu_context.h"
#include <thread>

namespace iowarp {

class PollingThreadManager {
private:
  std::jthread thread_;
  shm_queue* queue_;
  GpuContext* ctx_;

  void Poll(std::stop_token stop_token);

public:
  PollingThreadManager(shm_queue* queue, GpuContext* ctx);

  ~PollingThreadManager() = default;

  void Stop();

  bool IsRunning() const { return thread_.joinable(); }
};

} // namespace iowarp
