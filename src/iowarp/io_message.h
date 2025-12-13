#pragma once

#include "../hdf5/cstd.h"

namespace iowarp {

enum class IoType : uint8_t {
  kOpen,
  kWrite,
  kRead,
  kClose,
  kShutdown
};

struct IoMessage {
  IoType type_;

  // params
  char* filename;     // kOpen
  char* buffer;       // kRead/kWrite
  size_t size;        // kRead/kWrite (number of bytes)
  size_t offset;      // kRead/kWrite (file offset)
  int flags;          // kOpen (O_RDONLY, O_WRONLY, etc.)
  int mode;           // kOpen (permissions)

  // outputs
  int fd;             // output for kOpen, input for kRead/kWrite/kClose
  int32_t result_;    // return value from syscall (bytes read/written, or -1)
  int errno_;         // error code if result_ < 0
};

} // namespace iowarp