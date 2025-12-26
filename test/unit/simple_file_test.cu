#include "../../src/hdf5/file.h"
#include "../../src/iowarp/gpu_posix.h"
#include "../../src/iowarp/cpu_polling.h"
#include <cuda_runtime.h>
#include <cstdio>

using namespace iowarp;

__global__ void file_test_kernel(const char* filename, GpuContext* ctx, int* error_code) {
    printf("[GPU] Kernel started\n");

    printf("[GPU] About to call File::New for: %s\n", filename);
    auto file_result = File::New(filename, ctx);

    if (!file_result) {
        printf("[GPU] Failed to open file\n");
        *error_code = 1;
        return;
    }

    printf("[GPU] File opened successfully\n");
    *error_code = 0;
}

int main() {
    printf("=== Simple File Test ===\n\n");

    size_t stackSize, heapSize;
    cudaDeviceGetLimit(&stackSize, cudaLimitStackSize);
    printf("[MAIN] Current stack size: %zu bytes\n", stackSize);

    cudaDeviceSetLimit(cudaLimitStackSize, 16384);
    cudaDeviceGetLimit(&stackSize, cudaLimitStackSize);
    printf("[MAIN] New stack size: %zu bytes\n", stackSize);

    cudaDeviceGetLimit(&heapSize, cudaLimitMallocHeapSize);
    printf("[MAIN] Current heap size: %zu bytes\n", heapSize);

    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 256 * 1024 * 1024);
    cudaDeviceGetLimit(&heapSize, cudaLimitMallocHeapSize);
    printf("[MAIN] New heap size: %zu bytes\n\n", heapSize);

    const char* test_filename = "gpu_test.h5";

    shm_queue* h_queue;
    shm_queue* d_queue;
    cudaError_t err = cudaHostAlloc(&h_queue, sizeof(shm_queue), cudaHostAllocMapped);
    if (err != cudaSuccess) {
        printf("[MAIN] cudaHostAlloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaHostGetDevicePointer(&d_queue, h_queue, 0);
    new (h_queue) shm_queue();

    printf("[MAIN] Allocated queue\n");

    GpuContext* h_ctx;
    GpuContext* d_ctx;
    err = cudaHostAlloc(&h_ctx, sizeof(GpuContext), cudaHostAllocMapped);
    if (err != cudaSuccess) {
        printf("[MAIN] cudaHostAlloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaHostGetDevicePointer(&d_ctx, h_ctx, 0);
    new (h_ctx) GpuContext();
    h_ctx->queue_ = d_queue;

    printf("[MAIN] Allocated context\n");

    char* d_filename;
    int* d_error_code;
    cudaHostAlloc(&d_filename, 256, cudaHostAllocMapped);
    cudaHostAlloc(&d_error_code, sizeof(int), cudaHostAllocMapped);
    strcpy(d_filename, test_filename);
    *d_error_code = 0;

    printf("[MAIN] Memory allocated\n\n");

    {
        PollingThreadManager polling(h_queue, h_ctx);

        printf("[MAIN] Launching kernel...\n");
        fflush(stdout);

        file_test_kernel<<<1, 1>>>(d_filename, d_ctx, d_error_code);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("[MAIN] Kernel launch error: %s\n", cudaGetErrorString(err));
            fflush(stdout);
            return 1;
        }

        printf("[MAIN] Kernel launched successfully, waiting for completion...\n");
        fflush(stdout);

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("[MAIN] CUDA sync error: %s\n", cudaGetErrorString(err));
            fflush(stdout);
            return 1;
        }

        printf("[MAIN] Kernel completed\n");
        fflush(stdout);
    }

    int error_code = *d_error_code;
    printf("[MAIN] Error code: %d\n", error_code);

    cudaFreeHost(d_filename);
    cudaFreeHost(d_error_code);
    cudaFreeHost(h_queue);
    cudaFreeHost(h_ctx);

    printf("\n=== Test %s ===\n", error_code == 0 ? "PASSED" : "FAILED");

    return error_code;
}
