#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#include "hdf5/file.h"
#include "hdf5/group.h"
#include "hdf5/dataset.h"
#include "iowarp/gpu_context.h"
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
    printf("[TEST] Context ptr: %p, filename: %s\n", ctx, filename);

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

// Results structure for allocator test (store in mapped memory for CPU readback)
struct AllocTestResults {
    bool raw_mem_test_pass;
    size_t this_val;
    size_t data_start_val;
    size_t region_size_val;
    void* backend_ptr;
    size_t alloc_offset;
    bool alloc_is_null;
    void* alloc_ptr;
    bool alloc_write_test_pass;
    // Raw memory dump for inspection (BEFORE and AFTER allocation)
    size_t allocator_raw_before[16];
    size_t allocator_raw_after[16];
    // Additional debugging for heap internals
    size_t big_heap_offset;      // Current heap offset before allocation
    size_t big_heap_max_offset;  // Max offset
    size_t heap_alloc_result;    // Direct heap allocation result
    size_t allocator_data_off;   // GetAllocatorDataOff()
    size_t allocator_data_size;  // GetAllocatorDataSize()
};

// Minimal test to isolate allocator behavior on GPU
__global__
void test_allocator_kernel(hdf5::HdfAllocator* allocator, char* raw_mem, size_t data_offset, AllocTestResults* results) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("\n=== Direct Allocator Test (GPU) ===\n");

        // Test 1: Raw memory access (bypass allocator completely)
        char* data_region = raw_mem + data_offset;
        char orig_val = data_region[0];
        data_region[0] = 0x42;
        char read_back = data_region[0];
        results->raw_mem_test_pass = (read_back == 0x42);
        data_region[0] = orig_val;  // restore
        printf("[Test 1] Raw memory access: %s\n", results->raw_mem_test_pass ? "PASS" : "FAIL");

        // Test 2: Read allocator fields
        results->this_val = allocator->this_;
        results->data_start_val = allocator->data_start_;
        results->region_size_val = allocator->region_size_;
        printf("[Test 2] this_=%lu, data_start_=%lu, region_size_=%lu\n",
               (unsigned long)results->this_val,
               (unsigned long)results->data_start_val,
               (unsigned long)results->region_size_val);

        // Dump raw allocator memory BEFORE allocation
        size_t* alloc_as_size_t = reinterpret_cast<size_t*>(allocator);
        for (int i = 0; i < 16; i++) {
            results->allocator_raw_before[i] = alloc_as_size_t[i];
        }

        // Test 3: GetBackendData
        char* backend = allocator->GetBackendData();
        results->backend_ptr = backend;
        printf("[Test 3] GetBackendData()=%p\n", backend);

        // Test 3.5: Check heap internals (at known offsets from memory dump)
        // offset 104 = big_heap_.heap_, offset 112 = big_heap_.max_offset_
        results->big_heap_offset = alloc_as_size_t[13];      // offset 104 / 8 = 13
        results->big_heap_max_offset = alloc_as_size_t[14];  // offset 112 / 8 = 14
        results->allocator_data_off = allocator->GetAllocatorDataOff();
        results->allocator_data_size = allocator->GetAllocatorDataSize();
        printf("[Test 3.5] big_heap: offset=%lu, max=%lu\n",
               (unsigned long)results->big_heap_offset,
               (unsigned long)results->big_heap_max_offset);
        printf("[Test 3.5] allocator_data_off=%lu, allocator_data_size=%lu\n",
               (unsigned long)results->allocator_data_off,
               (unsigned long)results->allocator_data_size);

        // Dump more allocator memory to see free list state (size_t at various offsets)
        printf("[Test 3.6] Allocator memory dump (words 16-31):\n");
        for (int i = 16; i < 32; i++) {
            printf("  [%2d] offset %3d: %20lu (0x%016lx)\n",
                   i, i*8, (unsigned long)alloc_as_size_t[i], (unsigned long)alloc_as_size_t[i]);
        }

        // Test 4: Try allocation
        printf("[Test 4] Calling AllocateOffset(64)...\n");
        printf("[Test 4] results ptr=%p, allocator ptr=%p\n", results, allocator);

        auto offset = allocator->AllocateOffset(64);

        // Check for aliasing: is results inside allocator's region?
        char* results_as_char = reinterpret_cast<char*>(results);
        char* alloc_start = reinterpret_cast<char*>(allocator);
        char* alloc_end = alloc_start + 1048576;  // region_size
        printf("[Test 4] results=%p, allocator range=[%p, %p)\n",
               results_as_char, alloc_start, alloc_end);
        if (results_as_char >= alloc_start && results_as_char < alloc_end) {
            printf("[Test 4] WARNING: results struct is INSIDE allocator region!\n");
        }

        // Use volatile to prevent any optimization of the local
        volatile size_t local_offset = offset.off_.load();
        volatile bool local_is_null = offset.IsNull();

        printf("[Test 4] offset.off_ raw value=%lu\n", (unsigned long)offset.off_.load());
        printf("[Test 4] local_offset=%lu, local_is_null=%d\n",
               (unsigned long)local_offset, (int)local_is_null);

        // Now store to results
        results->alloc_offset = local_offset;
        results->alloc_is_null = local_is_null;

        printf("[Test 4] results->alloc_offset=%lu (after store)\n",
               (unsigned long)results->alloc_offset);

        if (!local_is_null && local_offset != 0) {
            char* alloc_ptr = backend + local_offset;
            results->alloc_ptr = alloc_ptr;
            printf("[Test 4] Allocated at offset %lu, ptr=%p\n",
                   (unsigned long)local_offset, alloc_ptr);

            // Only write if offset is at or beyond data_start (not within allocator header)
            if (local_offset >= results->data_start_val) {
                alloc_ptr[0] = 'X';
                results->alloc_write_test_pass = (alloc_ptr[0] == 'X');
                printf("[Test 4] Write test: %s\n", results->alloc_write_test_pass ? "PASS" : "FAIL");
            } else {
                printf("[Test 4] DANGER: offset %lu is within allocator (data_start=%lu)! Not writing.\n",
                       (unsigned long)local_offset, (unsigned long)results->data_start_val);
                results->alloc_write_test_pass = false;
            }
        } else {
            results->alloc_ptr = nullptr;
            results->alloc_write_test_pass = false;
            printf("[Test 4] Allocation FAILED (local_offset=%lu, local_is_null=%d)\n",
                   (unsigned long)local_offset, (int)local_is_null);
            printf("[Test 4] Re-reading offset.off_=%lu\n", (unsigned long)offset.off_.load());
        }

        // Dump raw allocator memory AFTER allocation
        for (int i = 0; i < 16; i++) {
            results->allocator_raw_after[i] = alloc_as_size_t[i];
        }

        printf("=== End Direct Allocator Test ===\n\n");
    }
}

__global__
void run_tests_kernel(iowarp::GpuContext* ctx, char* test_basic_path) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("[KERNEL] Entry\n");
        printf("[KERNEL] ctx=%p\n", ctx);
        printf("[KERNEL] queue=%p\n", ctx->queue_);
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
    cudaError_t err = cudaDeviceSetLimit(cudaLimitStackSize, 64 * 1024);  // 64KB stack
    if (err != cudaSuccess) {
        printf("[MAIN] Failed to set stack size: %s\n", cudaGetErrorString(err));
    }
    err = cudaDeviceSetLimit(cudaLimitMallocHeapSize, 256 * 1024 * 1024);  // 256MB heap
    if (err != cudaSuccess) {
        printf("[MAIN] Failed to set heap size: %s\n", cudaGetErrorString(err));
    }

    // Build GPU context with all required resources
    iowarp::GpuContextBuilder ctx_builder;
    if (!ctx_builder.Build()) {
        printf("[MAIN] Failed to build GPU context\n");
        return 1;
    }
    printf("[MAIN] GPU context built successfully\n");

    // Allocate filename in mapped memory
    char* h_test_basic_path;
    char* d_test_basic_path;
    err = cudaHostAlloc(&h_test_basic_path, 256, cudaHostAllocMapped);
    if (err != cudaSuccess) {
        printf("[MAIN] cudaHostAlloc for filename failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaHostGetDevicePointer(&d_test_basic_path, h_test_basic_path, 0);
    strcpy(h_test_basic_path, "/IdeaProjects/gpu-io-libs/data/test_basic.h5");

    printf("Running GPU tests...\n\n");

    // Allocate results structure in mapped memory
    AllocTestResults* h_results;
    AllocTestResults* d_results;
    err = cudaHostAlloc(&h_results, sizeof(AllocTestResults), cudaHostAllocMapped);
    if (err != cudaSuccess) {
        printf("[MAIN] cudaHostAlloc for results failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaHostGetDevicePointer(&d_results, h_results, 0);
    memset(h_results, 0, sizeof(AllocTestResults));

    // First, run the minimal allocator test to isolate allocator behavior
    printf("[MAIN] Running direct allocator test...\n");
    test_allocator_kernel<<<1, 1>>>(ctx_builder.DeviceAllocator(),
                                    ctx_builder.DeviceAllocMem(),
                                    ctx_builder.DataOffset(),
                                    d_results);
    err = cudaDeviceSynchronize();

    // Print results from CPU side (more reliable than GPU printf for 64-bit values)
    printf("\n[CPU READBACK] Results from GPU:\n");
    printf("  raw_mem_test: %s\n", h_results->raw_mem_test_pass ? "PASS" : "FAIL");
    printf("  this_val: %zu (0x%zx)\n", h_results->this_val, h_results->this_val);
    printf("  data_start_val: %zu\n", h_results->data_start_val);
    printf("  region_size_val: %zu\n", h_results->region_size_val);
    printf("  backend_ptr: %p\n", h_results->backend_ptr);
    printf("  alloc_offset: %zu (0x%zx)\n", h_results->alloc_offset, h_results->alloc_offset);
    printf("  alloc_is_null: %s\n", h_results->alloc_is_null ? "TRUE (NULL)" : "FALSE (valid)");
    printf("  alloc_ptr: %p\n", h_results->alloc_ptr);
    printf("  alloc_write_test: %s\n", h_results->alloc_write_test_pass ? "PASS" : "FAIL");
    printf("  big_heap_offset: %zu\n", h_results->big_heap_offset);
    printf("  big_heap_max_offset: %zu\n", h_results->big_heap_max_offset);
    printf("  allocator_data_off: %zu\n", h_results->allocator_data_off);
    printf("  allocator_data_size: %zu\n", h_results->allocator_data_size);
    printf("\n[CPU READBACK] Raw allocator memory BEFORE allocation:\n");
    for (int i = 0; i < 16; i++) {
        printf("  [%2d] offset %3d: %20zu (0x%016zx)\n", i, i*8, h_results->allocator_raw_before[i], h_results->allocator_raw_before[i]);
    }
    printf("\n[CPU READBACK] Raw allocator memory AFTER allocation:\n");
    for (int i = 0; i < 16; i++) {
        bool changed = (h_results->allocator_raw_before[i] != h_results->allocator_raw_after[i]);
        printf("  [%2d] offset %3d: %20zu (0x%016zx)%s\n", i, i*8,
               h_results->allocator_raw_after[i], h_results->allocator_raw_after[i],
               changed ? " <-- CHANGED" : "");
    }

    if (err != cudaSuccess) {
        fprintf(stderr, "[MAIN] Allocator test CUDA error: %s\n", cudaGetErrorString(err));
        cudaFreeHost(h_results);
        return 1;
    }
    printf("[MAIN] Direct allocator test completed\n\n");

    cudaFreeHost(h_results);

    // Run tests with polling thread
    {
        iowarp::PollingThreadManager polling(ctx_builder.HostQueue(), ctx_builder.HostContext());

        run_tests_kernel<<<1, 1>>>(ctx_builder.DeviceContext(), d_test_basic_path);

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

    // ctx_builder destructor handles cleanup
    return passed ? 0 : 1;
}
