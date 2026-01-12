#include "../../src/hdf5/file.h"
#include "../../src/hdf5/dataset.h"
#include "../../src/hdf5/datatype.h"
#include "../../src/iowarp/gpu_posix.h"
#include "../../src/iowarp/cpu_polling.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

using namespace iowarp;

__global__ void read_write_hdf5_kernel(const char* filename, GpuContext* ctx, int* read_output, int* write_output, int* error_code) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    *error_code = 0;
    printf("[GPU] Starting kernel\\n");
    printf("[GPU] Opening file: %s\\n", filename);

    auto file_result = File::New(filename, ctx);
    if (!file_result) {
        printf("[GPU] Failed to open file\\n");
        *error_code = 1;
        return;
    }

    printf("[GPU] File opened successfully\\n");
    File file = cstd::move(*file_result);
    auto root = file.RootGroup();

    // === PART 1: Read existing dataset ===
    printf("[GPU] Opening dataset 'integers'\\n");
    auto dataset_result = root.OpenDataset("integers");
    if (!dataset_result) {
        printf("[GPU] Failed to open dataset 'integers'\\n");
        *error_code = 2;
        return;
    }

    printf("[GPU] Dataset 'integers' opened\\n");
    Dataset read_dataset = cstd::move(*dataset_result);

    printf("[GPU] Reading 10 integers from 'integers' dataset\\n");
    for (int i = 0; i < 10; ++i) {
        auto elem_result = read_dataset.Get<int>(i);
        if (!elem_result) {
            printf("[GPU] Failed to read element %d\\n", i);
            *error_code = 3;
            return;
        }
        read_output[i] = *elem_result;
    }

    printf("[GPU] Successfully read data: [");
    for (int i = 0; i < 10; ++i) {
        printf("%d", read_output[i]);
        if (i < 9) printf(", ");
    }
    printf("]\\n");

    // === PART 2: Create new dataset and write to it ===
    printf("[GPU] Creating new dataset 'gpu_written'\\n");

    auto write_dataset_result = root.CreateDataset("gpu_written", {10}, DatatypeMessage::i32_t());
    if (!write_dataset_result) {
        printf("[GPU] Failed to create dataset 'gpu_written'\\n");
        *error_code = 4;
        return;
    }

    printf("[GPU] Dataset 'gpu_written' created\\n");
    Dataset write_dataset = cstd::move(*write_dataset_result);

    // Write data: multiply each value by 2
    printf("[GPU] Writing data to 'gpu_written'\\n");
    int write_data[10];
    for (int i = 0; i < 10; ++i) {
        write_data[i] = i * 2;
    }

    auto write_result = write_dataset.Write(cstd::span<const int>(write_data, 10), 0);
    if (!write_result) {
        printf("[GPU] Failed to write data\\n");
        *error_code = 5;
        return;
    }

    printf("[GPU] Data written successfully\\n");

    // === PART 3: Read back what we wrote to verify ===
    printf("[GPU] Reading back from 'gpu_written' to verify\\n");
    for (int i = 0; i < 10; ++i) {
        auto elem_result = write_dataset.Get<int>(i);
        if (!elem_result) {
            printf("[GPU] Failed to read back element %d\\n", i);
            *error_code = 6;
            return;
        }
        write_output[i] = *elem_result;
    }

    printf("[GPU] Read back data: [");
    for (int i = 0; i < 10; ++i) {
        printf("%d", write_output[i]);
        if (i < 9) printf(", ");
    }
    printf("]\\n");

    printf("[GPU] All operations completed successfully!\\n");
}

int main() {
    printf("=== GPU HDF5 File Read/Write Test ===\\n");
    printf("Testing with: data/gpu_test.h5\\n\\n");
    fflush(stdout);

    // Set large stack size for deep HDF5 call chains with cuda::std templates
    cudaError_t stack_err = cudaDeviceSetLimit(cudaLimitStackSize, 64 * 1024);  // 64KB stack
    if (stack_err != cudaSuccess) {
        printf("[MAIN] Failed to set stack size: %s\\n", cudaGetErrorString(stack_err));
    }
    stack_err = cudaDeviceSetLimit(cudaLimitMallocHeapSize, 256 * 1024 * 1024);  // 256MB heap
    if (stack_err != cudaSuccess) {
        printf("[MAIN] Failed to set heap size: %s\\n", cudaGetErrorString(stack_err));
    }

    const char* test_filename = "data/gpu_test.h5";

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

    printf("[MAIN] Allocated queue\\n");
    fflush(stdout);

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

    printf("[MAIN] Allocated context\\n");
    fflush(stdout);

    // Allocate filename and output buffers
    char* d_filename;
    int* d_read_output;
    int* d_write_output;
    int* d_error_code;
    cudaHostAlloc(&d_filename, 256, cudaHostAllocMapped);
    cudaHostAlloc(&d_read_output, 10 * sizeof(int), cudaHostAllocMapped);
    cudaHostAlloc(&d_write_output, 10 * sizeof(int), cudaHostAllocMapped);
    cudaHostAlloc(&d_error_code, sizeof(int), cudaHostAllocMapped);

    strcpy(d_filename, test_filename);
    *d_error_code = 0;

    printf("[MAIN] Prepared memory\\n\\n");
    fflush(stdout);

    // Start CPU polling thread
    {
        PollingThreadManager polling(h_queue, h_ctx);

        printf("[MAIN] Launching GPU kernel...\\n");
        fflush(stdout);

        read_write_hdf5_kernel<<<1, 1>>>(d_filename, d_ctx, d_read_output, d_write_output, d_error_code);

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

        printf("[MAIN] Kernel completed\\n\\n");
        fflush(stdout);
    }

    // Check results
    if (*d_error_code != 0) {
        printf("[MAIN] GPU kernel reported error code: %d\\n", *d_error_code);
        return 1;
    }

    // Verify read data
    printf("[MAIN] Verifying READ data from 'integers': [");
    for (int i = 0; i < 10; ++i) {
        printf("%d", d_read_output[i]);
        if (i < 9) printf(", ");
    }
    printf("]\\n");

    bool read_correct = true;
    for (int i = 0; i < 10; ++i) {
        if (d_read_output[i] != i) {
            printf("[MAIN] READ mismatch at index %d: expected %d, got %d\\n", i, i, d_read_output[i]);
            read_correct = false;
        }
    }

    if (read_correct) {
        printf("[MAIN] ✓ Read data correct!\\n");
    }

    // Verify write data
    printf("[MAIN] Verifying WRITE data from 'gpu_written': [");
    for (int i = 0; i < 10; ++i) {
        printf("%d", d_write_output[i]);
        if (i < 9) printf(", ");
    }
    printf("]\\n");

    bool write_correct = true;
    for (int i = 0; i < 10; ++i) {
        int expected = i * 2;
        if (d_write_output[i] != expected) {
            printf("[MAIN] WRITE mismatch at index %d: expected %d, got %d\\n", i, expected, d_write_output[i]);
            write_correct = false;
        }
    }

    if (write_correct) {
        printf("[MAIN] ✓ Write/read-back data correct!\\n");
    }

    bool all_correct = read_correct && write_correct;

    // Cleanup
    cudaFreeHost(d_filename);
    cudaFreeHost(d_read_output);
    cudaFreeHost(d_write_output);
    cudaFreeHost(d_error_code);
    cudaFreeHost(h_queue);
    cudaFreeHost(h_ctx);

    printf("\\n=== Test %s ===\\n", all_correct ? "PASSED" : "FAILED");

    return all_correct ? 0 : 1;
}
