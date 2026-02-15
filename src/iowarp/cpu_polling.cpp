#include "cpu_polling.h"
#include <iostream>
#include <cerrno>

#ifdef _WIN32
  #include <io.h>
  #include <fcntl.h>
#else
  #include <fcntl.h>
  #include <unistd.h>
#endif

namespace iowarp {

PollingThreadManager::PollingThreadManager(shm_queue* queue, GpuContext* ctx)
  : queue_(queue)
  , ctx_(ctx)
  , thread_([this](std::stop_token st) { Poll(st); })
{
}

void PollingThreadManager::Stop() {
  if (thread_.joinable()) {
    thread_.request_stop();

    if (queue_->size() == 0) {
      queue_->post(IoMessage::Shutdown());
    }
  }
}

void PollingThreadManager::Poll(std::stop_token stop_token) {
  while (!stop_token.stop_requested()) {
    uint64_t qsize = queue_->size();
    if (qsize == 0) {
      std::this_thread::yield();
      continue;
    }

    auto& msg = queue_->get();

    switch (msg.type_) {
      case IoType::kOpen: {
        int fd;
#ifdef _WIN32
        fd = _open(msg.filename, msg.flags, msg.mode);
#else
        fd = ::open(msg.filename, msg.flags, msg.mode);
#endif
        msg.fd = fd;
        msg.result_ = fd >= 0 ? 0 : -1;
        msg.errno_ = fd >= 0 ? 0 : errno;
        break;
      }

      case IoType::kWrite: {
        ssize_t written;
#ifdef _WIN32
        _lseeki64(msg.fd, msg.offset, SEEK_SET);
        written = _write(msg.fd, msg.buffer, static_cast<unsigned int>(msg.size));
#else
        written = ::pwrite(msg.fd, msg.buffer, msg.size, msg.offset);
#endif
        msg.result_ = written;
        msg.errno_ = written >= 0 ? 0 : errno;
        break;
      }

      case IoType::kRead: {
        ssize_t read_bytes;
#ifdef _WIN32
        _lseeki64(msg.fd, msg.offset, SEEK_SET);
        read_bytes = _read(msg.fd, msg.buffer, static_cast<unsigned int>(msg.size));
#else
        read_bytes = ::pread(msg.fd, msg.buffer, msg.size, msg.offset);
#endif
        msg.result_ = read_bytes;
        msg.errno_ = read_bytes >= 0 ? 0 : errno;
        break;
      }

      case IoType::kClose: {
        int result;
#ifdef _WIN32
        result = _close(msg.fd);
#else
        result = ::close(msg.fd);
#endif
        msg.result_ = result;
        msg.errno_ = result >= 0 ? 0 : errno;
        break;
      }

      case IoType::kShutdown: {
        queue_->clear();
        return;
      }

      default:
        break;
    }

    queue_->clear();
  }
}

} // namespace iowarp
