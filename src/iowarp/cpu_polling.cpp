#include "cpu_polling.h"
#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>

namespace iowarp {

PollingThreadManager::PollingThreadManager(shm_queue* queue, GpuContext* ctx)
  : queue_(queue)
  , ctx_(ctx)
  , thread_([this](std::stop_token st) { Poll(st); })
{
  std::cout << "[CPU] Starting polling thread\n";
}

void PollingThreadManager::Stop() {
  if (thread_.joinable()) {
    std::cout << "[CPU] Requesting stop\n";
    thread_.request_stop();

    if (queue_->size() == 0) {
      queue_->post(IoMessage::Shutdown());
    }
  }
}

void PollingThreadManager::Poll(std::stop_token stop_token) {
  std::cout << "[CPU] Polling thread started\n";

  while (!stop_token.stop_requested()) {
    while (queue_->size() == 0 && !stop_token.stop_requested()) {
      std::this_thread::yield();
    }

    if (stop_token.stop_requested()) break;

    std::cout << "[CPU] Got message, size=" << queue_->size() << "\n";

    auto& msg = queue_->get().value();

    switch (msg.type_) {
      case IoType::kOpen: {
        std::cout << "[CPU] Opening file: " << msg.filename
                  << " (flags=" << msg.flags << ", mode=" << msg.mode << ")\n";

        int fd = ::open(msg.filename, msg.flags, msg.mode);
        msg.fd = fd;
        msg.result_ = fd >= 0 ? 0 : -1;
        msg.errno_ = fd >= 0 ? 0 : errno;

        std::cout << "[CPU] Open result: fd=" << fd << "\n";
        break;
      }

      case IoType::kWrite: {
        std::cout << "[CPU] Writing " << msg.size << " bytes at offset "
                  << msg.offset << " to fd=" << msg.fd << "\n";

        ssize_t written = ::pwrite(msg.fd, msg.buffer, msg.size, msg.offset);
        msg.result_ = written;
        msg.errno_ = written >= 0 ? 0 : errno;

        std::cout << "[CPU] Write result: " << written << " bytes\n";
        break;
      }

      case IoType::kRead: {
        std::cout << "[CPU] Reading " << msg.size << " bytes at offset "
                  << msg.offset << " from fd=" << msg.fd << "\n";

        ssize_t read_bytes = ::pread(msg.fd, msg.buffer, msg.size, msg.offset);
        msg.result_ = read_bytes;
        msg.errno_ = read_bytes >= 0 ? 0 : errno;

        std::cout << "[CPU] Read result: " << read_bytes << " bytes\n";
        break;
      }

      case IoType::kClose: {
        std::cout << "[CPU] Closing fd=" << msg.fd << "\n";

        int result = ::close(msg.fd);
        msg.result_ = result;
        msg.errno_ = result >= 0 ? 0 : errno;

        std::cout << "[CPU] Close result: " << result << "\n";
        break;
      }

      case IoType::kShutdown: {
        std::cout << "[CPU] Shutdown requested\n";
        queue_->clear();
        return;
      }

      default:
        std::cout << "[CPU] Unknown message type: " << static_cast<int>(msg.type_) << "\n";
        break;
    }

    queue_->clear();
  }

  std::cout << "[CPU] Polling thread exiting\n";
}

} // namespace iowarp
