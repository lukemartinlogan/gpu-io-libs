#include "../../src/iowarp/gpu_posix.h"
#include "../../src/iowarp/cpu_polling.h"
#include <cuda_runtime.h>
#include <cstdio>

using namespace iowarp;

__global__ void minimal_kernel(GpuContext* ctx, int* output) {
    printf("[GPU] Kernel started!\\n");
    *output = 42;
    printf("[GPU] Kernel finished!\\n");
}

int main() {
    printf("=== Minimal GPU Test ===\\n\\n");

    // Allocate shared queue
    shm_queue* h_queue;
    shm_queue* d_queue;
    cudaError_t err = cudaHostAlloc(&h_queue, sizeof(shm_queue), cudaHostAllocMapped);
    if (err != cudaSuccess) {
        printf("[MAIN] cudaHostAlloc failed: %s\\n", cudaGetErrorString(err));
        return 1;
    }
    cudaHostGetDevicePointer(&d_queue, h_queue, 0);
    new (h_queue) shm_queue();

    // Allocate context
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

    // Allocate output
    int* d_output;
    cudaHostAlloc(&d_output, sizeof(int), cudaHostAllocMapped);
    *d_output = 0;

    printf("[MAIN] Memory allocated\\n");

    // Start CPU polling thread
    {
        PollingThreadManager polling(h_queue, h_ctx);

        printf("[MAIN] Launching kernel...\\n");
        minimal_kernel<<<1, 1>>>(d_ctx, d_output);

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

    int output_value = *d_output;
    printf("[MAIN] Output: %d\\n", output_value);

    // Cleanup
    cudaFreeHost(d_output);
    cudaFreeHost(h_queue);
    cudaFreeHost(h_ctx);

    printf("\\n=== Test %s ===\\n", output_value == 42 ? "PASSED" : "FAILED");

    return output_value == 42 ? 0 : 1;
}
