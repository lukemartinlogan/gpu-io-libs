#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#include "hdf5/file.h"
#include "hdf5/group.h"
#include "hdf5/dataset.h"
#include "iowarp/gpu_context.h"
#include "iowarp/gpu_posix.h"
#include "iowarp/cpu_polling.h"

__device__ bool g_test_passed = true;
__device__ int g_tests_run = 0;
__device__ int g_tests_failed = 0;

#define GPU_ASSERT(cond, msg) do { \
    if (!(cond)) { \
        printf("ASSERTION FAILED: %s\n  at %s:%d\n", msg, __FILE__, __LINE__); \
        g_test_passed = false; \
        atomicAdd(&g_tests_failed, 1); \
    } \
    atomicAdd(&g_tests_run, 1); \
} while(0)

#define CHECK_EXPECTED(expr, msg) do { \
    auto&& result = (expr); \
    if (!result) { \
        printf("ERROR: %s failed - %s\n", msg, result.error().description); \
        g_test_passed = false; \
        atomicAdd(&g_tests_failed, 1); \
        return; \
    } \
} while(0)

__device__
void test_read_basic_file(iowarp::GpuContext* ctx, const char* filename) {
    printf("\n=== Test: Read Basic File ===\n");
    printf("[TEST] Context ptr: %p, staging_buffer: %p, filename: %s\n", ctx, ctx->staging_buffer_, filename);

    auto file_result = File::New(filename, ctx);
    CHECK_EXPECTED(file_result, "Open test_basic.h5");
    File file = cstd::move(*file_result);

    // Get root group
    Group root = file.RootGroup();

    // Open 1D dataset
    printf("  Opening data_1d dataset...\n");
    auto ds_1d_result = root.OpenDataset("data_1d");
    CHECK_EXPECTED(ds_1d_result, "Open data_1d");
    Dataset ds_1d = cstd::move(*ds_1d_result);

    // Read some values
    auto val0 = ds_1d.Get<double>(0);
    CHECK_EXPECTED(val0, "Read data_1d[0]");
    GPU_ASSERT(*val0 == 0.0, "data_1d[0] should be 0.0");

    auto val50 = ds_1d.Get<double>(50);
    CHECK_EXPECTED(val50, "Read data_1d[50]");
    GPU_ASSERT(*val50 == 50.0, "data_1d[50] should be 50.0");

    printf("  ✓ 1D dataset read correctly\n");

    // Open 2D dataset
    printf("  Opening data_2d dataset...\n");
    auto ds_2d_result = root.OpenDataset("data_2d");
    CHECK_EXPECTED(ds_2d_result, "Open data_2d");
    Dataset ds_2d = cstd::move(*ds_2d_result);

    // Read value from 2D dataset (row-major: index = row * cols + col)
    auto val_2d = ds_2d.Get<int32_t>(5 * 30 + 10);  // row 5, col 10
    CHECK_EXPECTED(val_2d, "Read data_2d[5,10]");
    GPU_ASSERT(*val_2d == 160, "data_2d[5,10] should be 160");

    printf("  ✓ 2D dataset read correctly\n");

    // Open small dataset
    printf("  Opening small dataset...\n");
    auto small_result = root.OpenDataset("small");
    CHECK_EXPECTED(small_result, "Open small");
    Dataset small = cstd::move(*small_result);

    for (int i = 0; i < 5; i++) {
        auto val = small.Get<int32_t>(i);
        CHECK_EXPECTED(val, "Read small dataset");
        GPU_ASSERT(*val == i + 1, "small dataset value incorrect");
    }

    printf("  ✓ Small dataset read correctly\n");
    printf("✓ test_read_basic_file passed\n");
}

__device__
void test_read_groups(iowarp::GpuContext* ctx) {
    printf("\n=== Test: Read Groups ===\n");

    auto file_result = File::New("../../data/test_groups.h5", ctx);
    CHECK_EXPECTED(file_result, "Open test_groups.h5");
    File file = cstd::move(*file_result);

    Group root = file.RootGroup();

    // Open group1
    printf("  Opening group1...\n");
    auto g1_result = root.OpenGroup("group1");
    CHECK_EXPECTED(g1_result, "Open group1");
    Group g1 = cstd::move(*g1_result);

    // Read dataset from group1
    auto ds_a_result = g1.OpenDataset("dataset_a");
    CHECK_EXPECTED(ds_a_result, "Open dataset_a");
    Dataset ds_a = cstd::move(*ds_a_result);

    auto val = ds_a.Get<float>(10);
    CHECK_EXPECTED(val, "Read dataset_a[10]");
    GPU_ASSERT(*val == 10.0f, "dataset_a[10] should be 10.0");

    printf("  ✓ group1/dataset_a read correctly\n");

    // Open group2
    printf("  Opening group2...\n");
    auto g2_result = root.OpenGroup("group2");
    CHECK_EXPECTED(g2_result, "Open group2");
    Group g2 = cstd::move(*g2_result);

    // Read from group2 datasets
    {
        auto ds_result = g2.OpenDataset("array_0");
        CHECK_EXPECTED(ds_result, "Open array_0");
        Dataset ds = cstd::move(*ds_result);
        auto first_val = ds.Get<int32_t>(0);
        CHECK_EXPECTED(first_val, "Read array value");
        GPU_ASSERT(*first_val == 0, "array value incorrect");
    }
    {
        auto ds_result = g2.OpenDataset("array_1");
        CHECK_EXPECTED(ds_result, "Open array_1");
        Dataset ds = cstd::move(*ds_result);
        auto first_val = ds.Get<int32_t>(0);
        CHECK_EXPECTED(first_val, "Read array value");
        GPU_ASSERT(*first_val == 1, "array value incorrect");
    }
    {
        auto ds_result = g2.OpenDataset("array_2");
        CHECK_EXPECTED(ds_result, "Open array_2");
        Dataset ds = cstd::move(*ds_result);
        auto first_val = ds.Get<int32_t>(0);
        CHECK_EXPECTED(first_val, "Read array value");
        GPU_ASSERT(*first_val == 2, "array value incorrect");
    }

    printf("  ✓ group2 datasets read correctly\n");

    // Open nested group
    printf("  Opening nested group...\n");
    auto nested_result = g1.OpenGroup("nested");
    CHECK_EXPECTED(nested_result, "Open nested group");
    Group nested = cstd::move(*nested_result);

    auto deep_result = nested.OpenDataset("deep_data");
    CHECK_EXPECTED(deep_result, "Open deep_data");
    Dataset deep = cstd::move(*deep_result);

    auto deep_val = deep.Get<double>(0);
    CHECK_EXPECTED(deep_val, "Read deep_data[0]");
    GPU_ASSERT(*deep_val == 1.0, "deep_data should contain 1.0");

    printf("  ✓ Nested group read correctly\n");
    printf("✓ test_read_groups passed\n");
}

__device__
void test_read_datatypes(iowarp::GpuContext* ctx) {
    printf("\n=== Test: Read Various Datatypes ===\n");

    auto file_result = File::New("../../data/test_datatypes.h5", ctx);
    CHECK_EXPECTED(file_result, "Open test_datatypes.h5");
    File file = cstd::move(*file_result);

    Group root = file.RootGroup();

    // Test int32
    printf("  Testing int32...\n");
    auto int32_result = root.OpenDataset("int32");
    CHECK_EXPECTED(int32_result, "Open int32");
    Dataset int32_ds = cstd::move(*int32_result);

    auto int32_val = int32_ds.Get<int32_t>(0);
    CHECK_EXPECTED(int32_val, "Read int32[0]");
    GPU_ASSERT(*int32_val == 10000, "int32[0] should be 10000");

    // Test float32
    printf("  Testing float32...\n");
    auto f32_result = root.OpenDataset("float32");
    CHECK_EXPECTED(f32_result, "Open float32");
    Dataset f32_ds = cstd::move(*f32_result);

    auto f32_val = f32_ds.Get<float>(1);
    CHECK_EXPECTED(f32_val, "Read float32[1]");
    GPU_ASSERT(*f32_val == 2.5f, "float32[1] should be 2.5");

    // Test float64
    printf("  Testing float64...\n");
    auto f64_result = root.OpenDataset("float64");
    CHECK_EXPECTED(f64_result, "Open float64");
    Dataset f64_ds = cstd::move(*f64_result);

    auto f64_val = f64_ds.Get<double>(0);
    CHECK_EXPECTED(f64_val, "Read float64[0]");
    GPU_ASSERT(*f64_val > 1.234 && *f64_val < 1.235, "float64[0] should be ~1.23456789");

    printf("  ✓ Datatype reading works\n");
    printf("✓ test_read_datatypes passed\n");
}

__device__
void test_write_operations(iowarp::GpuContext* ctx) {
    printf("\n=== Test: Write Operations ===\n");

    auto file_result = File::New("../../data/test_write_target.h5", ctx);
    CHECK_EXPECTED(file_result, "Open test_write_target.h5");
    File file = cstd::move(*file_result);

    Group root = file.RootGroup();

    // Open dataset for writing
    printf("  Opening write_test dataset...\n");
    auto ds_result = root.OpenDataset("write_test");
    CHECK_EXPECTED(ds_result, "Open write_test");
    Dataset ds = cstd::move(*ds_result);

    // Write some values
    printf("  Writing test data...\n");
    float test_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    auto write_result = ds.Write(cstd::span<const float>(test_data, 5), 10);
    CHECK_EXPECTED(write_result, "Write test data");

    // Read back to verify
    printf("  Reading back written data...\n");
    auto val = ds.Get<float>(11);
    CHECK_EXPECTED(val, "Read written value");
    GPU_ASSERT(*val == 2.0f, "Written value should be 2.0");

    printf("  ✓ Write operations work\n");
    printf("✓ test_write_operations passed\n");
}

__device__
void test_create_file(iowarp::GpuContext* ctx) {
    printf("\n=== Test: Create New File ===\n");

    // Create a new file
    printf("  Creating new file...\n");
    auto file_result = File::New("../../data/test_created.h5", ctx);
    CHECK_EXPECTED(file_result, "Create test_created.h5");
    File file = cstd::move(*file_result);

    Group root = file.RootGroup();

    // Create a new group
    printf("  Creating new group...\n");
    auto new_group_result = root.CreateGroup("test_group");
    CHECK_EXPECTED(new_group_result, "Create test_group");
    Group new_group = cstd::move(*new_group_result);

    // Create a dataset in the new group
    printf("  Creating dataset in group...\n");
    hdf5::dim_vector<len_t> dims;
    dims.push_back(10);

    auto ds_result = new_group.CreateDataset("my_dataset", dims, DatatypeMessage::f32_t());
    CHECK_EXPECTED(ds_result, "Create my_dataset");
    Dataset ds = cstd::move(*ds_result);

    // Write to the new dataset
    printf("  Writing to new dataset...\n");
    float data[] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f};
    auto write_result = ds.Write(cstd::span<const float>(data, 5), 0);
    CHECK_EXPECTED(write_result, "Write to new dataset");

    // Read back
    auto val = ds.Get<float>(2);
    CHECK_EXPECTED(val, "Read from new dataset");
    GPU_ASSERT(*val == 30.0f, "New dataset value should be 30.0");

    printf("  ✓ File creation works\n");
    printf("✓ test_create_file passed\n");
}

__global__
void run_tests_kernel(iowarp::GpuContext* ctx, char* test_basic_path) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("[KERNEL] Entry\n");
        printf("[KERNEL] ctx=%p\n", ctx);
        printf("[KERNEL] queue=%p\n", ctx->queue_);
        printf("[KERNEL] staging=%p\n", ctx->staging_buffer_);
        printf("[KERNEL] Init complete\n");

        printf("========================================\n");
        printf("HDF5 Comprehensive Test Suite (GPU)\n");
        printf("========================================\n");

        test_read_basic_file(ctx, test_basic_path);
        // test_read_groups(ctx);
        // test_read_datatypes(ctx);
        // test_write_operations(ctx);
        // test_create_file(ctx);

        printf("\n========================================\n");
        if (g_test_passed) {
            printf("✓ ALL TESTS PASSED\n");
        } else {
            printf("✗ SOME TESTS FAILED\n");
        }
        printf("Tests run: %d, Failed: %d\n", g_tests_run, g_tests_failed);
        printf("========================================\n");
    }
}

int main() {
    printf("Initializing CUDA...\n");

    // Set large stack size for deep HDF5 call chains with cuda::std templates
    cudaError_t stack_err = cudaDeviceSetLimit(cudaLimitStackSize, 64 * 1024);  // 64KB stack
    if (stack_err != cudaSuccess) {
        printf("[MAIN] Failed to set stack size: %s\n", cudaGetErrorString(stack_err));
    }
    stack_err = cudaDeviceSetLimit(cudaLimitMallocHeapSize, 256 * 1024 * 1024);  // 256MB heap
    if (stack_err != cudaSuccess) {
        printf("[MAIN] Failed to set heap size: %s\n", cudaGetErrorString(stack_err));
    }

    // Allocate shared queue
    using namespace iowarp;
    shm_queue* h_queue;
    shm_queue* d_queue;
    cudaError_t err = cudaHostAlloc(&h_queue, sizeof(shm_queue), cudaHostAllocMapped);
    if (err != cudaSuccess) {
        printf("[MAIN] cudaHostAlloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaHostGetDevicePointer(&d_queue, h_queue, 0);
    new (h_queue) shm_queue();

    // Allocate context
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

    // Allocate staging buffer for GPU-CPU I/O (64KB, CPU-accessible)
    char* h_staging;
    char* d_staging;
    err = cudaHostAlloc(&h_staging, 64 * 1024, cudaHostAllocMapped);
    if (err != cudaSuccess) {
        printf("[MAIN] cudaHostAlloc for staging buffer failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaHostGetDevicePointer(&d_staging, h_staging, 0);
    h_ctx->staging_buffer_ = d_staging;
    printf("[MAIN] Staging buffer allocated\n");

    printf("Running GPU tests...\n\n");

    // Allocate filenames in mapped memory (CPU-accessible)
    char* h_test_basic_path;
    char* d_test_basic_path;
    err = cudaHostAlloc(&h_test_basic_path, 256, cudaHostAllocMapped);
    if (err != cudaSuccess) {
        printf("[MAIN] cudaHostAlloc for filename failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaHostGetDevicePointer(&d_test_basic_path, h_test_basic_path, 0);
    strcpy(h_test_basic_path, "/IdeaProjects/gpu-io-libs/data/test_basic.h5");

    // Run tests in a scope with polling thread
    {
        PollingThreadManager polling(h_queue, h_ctx);

        run_tests_kernel<<<1, 1>>>(d_ctx, d_test_basic_path);

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
            return 1;
        }
    }

    cudaFreeHost(h_test_basic_path);

    // Get results
    bool passed;
    int tests_run, tests_failed;
    cudaMemcpyFromSymbol(&passed, g_test_passed, sizeof(bool));
    cudaMemcpyFromSymbol(&tests_run, g_tests_run, sizeof(int));
    cudaMemcpyFromSymbol(&tests_failed, g_tests_failed, sizeof(int));

    // Cleanup
    cudaFreeHost(h_staging);
    h_ctx->~GpuContext();
    h_queue->~shm_queue();
    cudaFreeHost(h_queue);
    cudaFreeHost(h_ctx);

    return passed ? 0 : 1;
}
