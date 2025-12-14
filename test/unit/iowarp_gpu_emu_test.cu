#include "../../src/iowarp/gpu_posix.h"
#include "../../src/iowarp/cpu_polling.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

#ifdef _WIN32
  #include <io.h>
  #include <fcntl.h>
  // Windows flags
  #define OPEN_FLAGS (_O_RDWR | _O_CREAT | _O_TRUNC | _O_BINARY)
  #define OPEN_MODE (_S_IREAD | _S_IWRITE)
#else
  #include <fcntl.h>
  #define OPEN_FLAGS (O_RDWR | O_CREAT | O_TRUNC)
  #define OPEN_MODE (0644)
#endif

using namespace iowarp;

// Simple GPU kernel to test I/O operations
__global__ void TestIoWarpKernel(GpuContext* ctx, char* filename, char* write_buffer, char* read_buffer) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;  // Only thread 0 does I/O

  printf("[GPU] Starting I/O test\n");

  // Open file
  int fd = gpu_posix::open(filename, OPEN_FLAGS, OPEN_MODE, *ctx);
  printf("[GPU] Opened file, fd=%d\n", fd);

  if (fd < 0) {
    printf("[GPU] Failed to open file!\n");
    return;
  }

  // Write some data
  const char* test_data = "Hello from GPU!";
  const size_t data_len = 16;  // Length of "Hello from GPU!"

  // Manual copy (memcpy isn't device-callable)
  for (size_t i = 0; i < data_len; i++) {
    write_buffer[i] = test_data[i];
  }

  ssize_t written = gpu_posix::pwrite(fd, write_buffer, data_len, 0, *ctx);
  printf("[GPU] Wrote %zd bytes\n", written);

  // Read it back
  ssize_t read_bytes = gpu_posix::pread(fd, read_buffer, data_len, 0, *ctx);
  printf("[GPU] Read %zd bytes\n", read_bytes);

  // Verify (manual comparison instead of memcmp)
  bool match = (read_bytes == written);
  if (match) {
    for (size_t i = 0; i < data_len; i++) {
      if (write_buffer[i] != read_buffer[i]) {
        match = false;
        break;
      }
    }
  }

  if (match) {
    printf("[GPU] ✓ Data verified successfully!\n");
  } else {
    printf("[GPU] ✗ Data verification failed!\n");
  }

  // Close file
  int close_result = gpu_posix::close(fd, *ctx);
  printf("[GPU] Closed file, result=%d\n", close_result);

  printf("[GPU] Test complete\n");
}

int main() {
  printf("=== IOWarp GPU Emulation Test ===\n\n");
  fflush(stdout);

  // Allocate shared queue using pinned host memory
  shm_queue* h_queue;
  shm_queue* d_queue;
  cudaError_t err = cudaHostAlloc(&h_queue, sizeof(shm_queue), cudaHostAllocMapped);
  if (err != cudaSuccess) {
    printf("[MAIN] cudaHostAlloc failed for queue: %s\n", cudaGetErrorString(err));
    return 1;
  }
  cudaHostGetDevicePointer(&d_queue, h_queue, 0);
  new (h_queue) shm_queue();  // Placement new to initialize

  printf("[MAIN] Allocated queue: host=%p, device=%p\n", (void*)h_queue, (void*)d_queue);
  fflush(stdout);

  // Allocate context
  GpuContext* h_ctx;
  GpuContext* d_ctx;
  err = cudaHostAlloc(&h_ctx, sizeof(GpuContext), cudaHostAllocMapped);
  if (err != cudaSuccess) {
    printf("[MAIN] cudaHostAlloc failed for context: %s\n", cudaGetErrorString(err));
    return 1;
  }
  cudaHostGetDevicePointer(&d_ctx, h_ctx, 0);
  new (h_ctx) GpuContext();
  h_ctx->queue_ = d_queue;

  printf("[MAIN] Allocated context: host=%p, device=%p\n", (void*)h_ctx, (void*)d_ctx);
  fflush(stdout);

  // Allocate filename and buffers using pinned host memory (mapped for GPU access)
  char* filename;
  char* write_buffer;
  char* read_buffer;
  cudaHostAlloc(&filename, 256, cudaHostAllocMapped);
  cudaHostAlloc(&write_buffer, 1024, cudaHostAllocMapped);
  cudaHostAlloc(&read_buffer, 1024, cudaHostAllocMapped);

  strcpy(filename, "test_iowarp_gpu.dat");
  printf("[MAIN] Test file: %s\n\n", filename);
  fflush(stdout);

  // Start CPU polling thread (RAII - automatically stops in destructor)
  {
    PollingThreadManager polling(h_queue, h_ctx);

    // Launch GPU kernel
    printf("[MAIN] Launching GPU kernel...\n");
    fflush(stdout);
    TestIoWarpKernel<<<1, 1>>>(d_ctx, filename, write_buffer, read_buffer);

    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("[MAIN] Kernel launch error: %s\n", cudaGetErrorString(err));
      fflush(stdout);
      return 1;
    }

    // Wait for kernel completion
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      printf("[MAIN] CUDA sync error: %s\n", cudaGetErrorString(err));
      fflush(stdout);
      return 1;
    }

    printf("[MAIN] Kernel completed\n");
    fflush(stdout);

    // PollingThreadManager destructor will stop the polling thread
  }

  // Cleanup
  cudaFreeHost(filename);
  cudaFreeHost(write_buffer);
  cudaFreeHost(read_buffer);
  cudaFreeHost(h_queue);
  cudaFreeHost(h_ctx);

  printf("\n=== Test Complete ===\n");

  return 0;
}
