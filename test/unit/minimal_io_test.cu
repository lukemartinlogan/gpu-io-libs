#include "../../src/iowarp/gpu_posix.h"
#include "../../src/iowarp/cpu_polling.h"
#include <cuda_runtime.h>
#include <cstdio>

using namespace iowarp;

__global__ void io_test_kernel(const char* filename, GpuContext* ctx, int* error_code) {
    printf("[GPU] Kernel started, attempting to open file\\n");

    int fd = iowarp::gpu_posix::open(filename, 0, 0, *ctx);

    if (fd < 0) {
        printf("[GPU] Failed to open file, fd=%d\\n", fd);
        *error_code = 1;
        return;
    }

    printf("[GPU] File opened successfully, fd=%d\\n", fd);

    char buffer[64];
    ssize_t bytes_read = iowarp::gpu_posix::pread(fd, buffer, 32, 0, *ctx);

    printf("[GPU] Read %zd bytes\\n", bytes_read);

    iowarp::gpu_posix::close(fd, *ctx);

    printf("[GPU] File closed, kernel finished\\n");
    *error_code = 0;
}

int main() {
    printf("=== Minimal I/O Test ===\\n\\n");

    const char* test_filename = "gpu_test.h5";

    shm_queue* h_queue;
    shm_queue* d_queue;
    cudaError_t err = cudaHostAlloc(&h_queue, sizeof(shm_queue), cudaHostAllocMapped);
    if (err != cudaSuccess) {
        printf("[MAIN] cudaHostAlloc failed: %s\\n", cudaGetErrorString(err));
        return 1;
    }
    cudaHostGetDevicePointer(&d_queue, h_queue, 0);
    new (h_queue) shm_queue();

    GpuContext* h_ctx;
    GpuContext* d_ctx;
    err = cudaHostAlloc(&h_ctx, sizeof(GpuContext), cudaHostAllocMapped);
    if (err != cudaSuccess) {
        printf("[MAIN] cudaHostAlloc failed: %s\\n", cudaGetErrorString(err));
        return 1;
    }
    cudaHostGetDevicePointer(&d_ctx, h_ctx, 0);
    new (h_ctx) GpuContext();
    h_ctx->queue_ = d_queue;

    char* d_filename;
    int* d_error_code;
    cudaHostAlloc(&d_filename, 256, cudaHostAllocMapped);
    cudaHostAlloc(&d_error_code, sizeof(int), cudaHostAllocMapped);
    strcpy(d_filename, test_filename);
    *d_error_code = 0;

    printf("[MAIN] Memory allocated\\n");

    {
        PollingThreadManager polling(h_queue, h_ctx);

        printf("[MAIN] Launching kernel...\\n");
        io_test_kernel<<<1, 1>>>(d_filename, d_ctx, d_error_code);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("[MAIN] Kernel launch error: %s\\n", cudaGetErrorString(err));
            return 1;
        }

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("[MAIN] CUDA sync error: %s\\n", cudaGetErrorString(err));
            return 1;
        }

        printf("[MAIN] Kernel completed\\n");
    }

    int error_code = *d_error_code;
    printf("[MAIN] Error code: %d\\n", error_code);

    cudaFreeHost(d_filename);
    cudaFreeHost(d_error_code);
    cudaFreeHost(h_queue);
    cudaFreeHost(h_ctx);

    printf("\\n=== Test %s ===\\n", error_code == 0 ? "PASSED" : "FAILED");

    return error_code;
}
