#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <fstream>
#include <filesystem>

#include "common/benchmark_utils.h"

#ifdef HDF5_CPU_BASELINE_ENABLED
#include "common/hdf5_reference.h"
#endif

// Global GPU context (initialized once)
static bench_utils::GpuBenchContext* g_gpu_ctx = nullptr;

// Path to working copy for write tests
static const char* WRITE_TEST_FILE = "../benches/data/bench_write_copy.h5";
static char* g_d_write_filepath = nullptr;
static char* g_h_write_filepath = nullptr;

// Copy benchmark file for write tests
static bool copy_benchmark_file() {
    std::filesystem::copy_file(
        bench_utils::BENCH_DATA_FILE,
        WRITE_TEST_FILE,
        std::filesystem::copy_options::overwrite_existing
    );
    return std::filesystem::exists(WRITE_TEST_FILE);
}

// ============================================================================
// GPU Benchmark Kernels
// ============================================================================

// Kernel using Dataset::Read() - bulk read for contiguous data
template<typename T>
__global__ void sequential_read_kernel(
    iowarp::GpuContext* ctx,
    const char* filepath,
    const char* dataset_name,
    size_t offset,
    size_t count,
    T* output,
    bool* success
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *success = false;

        auto file_result = File::New(filepath, ctx);
        if (!file_result) return;
        File file = cstd::move(*file_result);

        Group root = file.RootGroup();

        auto ds_result = root.OpenDataset(dataset_name);
        if (!ds_result) return;
        Dataset ds = cstd::move(*ds_result);

        // Use bulk Read (single I/O call for all elements)
        cstd::span<byte_t> buffer(
            reinterpret_cast<byte_t*>(output),
            count * sizeof(T)
        );

        auto read_result = ds.Read(buffer, offset, count);
        if (!read_result) return;

        *success = true;
    }
}

// Kernel using Dataset::ReadHyperslab() - tests our GetNextContiguousRun optimization
template<typename T>
__global__ void hyperslab_read_kernel(
    iowarp::GpuContext* ctx,
    const char* filepath,
    const char* dataset_name,
    size_t offset,
    size_t count,
    T* output,
    bool* success
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *success = false;

        auto file_result = File::New(filepath, ctx);
        if (!file_result) return;
        File file = cstd::move(*file_result);

        Group root = file.RootGroup();

        auto ds_result = root.OpenDataset(dataset_name);
        if (!ds_result) return;
        Dataset ds = cstd::move(*ds_result);

        // Use ReadHyperslab (tests our contiguous run optimization)
        hdf5::dim_vector<uint64_t> start_vec(1, offset);
        hdf5::dim_vector<uint64_t> count_vec(1, count);

        cstd::span<byte_t> buffer(
            reinterpret_cast<byte_t*>(output),
            count * sizeof(T)
        );

        auto read_result = ds.ReadHyperslab(buffer, start_vec, count_vec);
        if (!read_result) return;

        *success = true;
    }
}

__global__ void file_open_kernel(
    iowarp::GpuContext* ctx,
    const char* filepath,
    bool* success
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        auto file_result = File::New(filepath, ctx);
        *success = file_result.has_value();
    }
}

__global__ void dataset_open_kernel(
    iowarp::GpuContext* ctx,
    const char* filepath,
    const char* dataset_name,
    bool* success
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *success = false;

        auto file_result = File::New(filepath, ctx);
        if (!file_result) return;
        File file = cstd::move(*file_result);

        Group root = file.RootGroup();
        auto ds_result = root.OpenDataset(dataset_name);
        *success = ds_result.has_value();
    }
}

template<typename T>
__global__ void sequential_write_kernel(
    iowarp::GpuContext* ctx,
    const char* filepath,
    const char* dataset_name,
    size_t offset,
    size_t count,
    const T* input,
    bool* success
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *success = false;

        auto file_result = File::New(filepath, ctx);
        if (!file_result) return;
        File file = cstd::move(*file_result);

        Group root = file.RootGroup();

        auto ds_result = root.OpenDataset(dataset_name);
        if (!ds_result) return;
        Dataset ds = cstd::move(*ds_result);

        cstd::span<const T> data_span(input, count);
        auto write_result = ds.Write(data_span, offset);
        if (!write_result) return;

        *success = true;
    }
}

__global__ void create_group_kernel(
    iowarp::GpuContext* ctx,
    const char* filepath,
    const char* group_name,
    bool* success
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *success = false;

        auto file_result = File::New(filepath, ctx);
        if (!file_result) return;
        File file = cstd::move(*file_result);

        Group root = file.RootGroup();
        auto group_result = root.CreateGroup(group_name);
        *success = group_result.has_value();
    }
}

// ============================================================================
// GPU Benchmarks
// ============================================================================

static void BM_GPU_FileOpen(benchmark::State& state) {
    bool* d_success;
    cudaMalloc(&d_success, sizeof(bool));

    for (auto _ : state) {
        file_open_kernel<<<1, 1>>>(
            g_gpu_ctx->device_ctx(),
            g_gpu_ctx->device_filepath(),
            d_success
        );
        cudaDeviceSynchronize();
    }

    cudaFree(d_success);
}
BENCHMARK(BM_GPU_FileOpen)->Unit(benchmark::kMillisecond);

static void BM_GPU_DatasetOpen(benchmark::State& state) {
    bool* d_success;
    cudaMalloc(&d_success, sizeof(bool));

    char* h_dsname;
    char* d_dsname;
    cudaHostAlloc(&h_dsname, 64, cudaHostAllocMapped);
    cudaHostGetDevicePointer(&d_dsname, h_dsname, 0);
    strcpy(h_dsname, "data_1d_double");

    for (auto _ : state) {
        dataset_open_kernel<<<1, 1>>>(
            g_gpu_ctx->device_ctx(),
            g_gpu_ctx->device_filepath(),
            d_dsname,
            d_success
        );
        cudaDeviceSynchronize();
    }

    cudaFree(d_success);
    cudaFreeHost(h_dsname);
}
BENCHMARK(BM_GPU_DatasetOpen)->Unit(benchmark::kMillisecond);

static void BM_GPU_SequentialRead_Double(benchmark::State& state) {
    const size_t count = state.range(0);

    double* d_output;
    bool* d_success;
    cudaMalloc(&d_output, count * sizeof(double));
    cudaMalloc(&d_success, sizeof(bool));

    char* h_dsname;
    char* d_dsname;
    cudaHostAlloc(&h_dsname, 64, cudaHostAllocMapped);
    cudaHostGetDevicePointer(&d_dsname, h_dsname, 0);
    strcpy(h_dsname, "data_1d_double");

    for (auto _ : state) {
        sequential_read_kernel<double><<<1, 1>>>(
            g_gpu_ctx->device_ctx(),
            g_gpu_ctx->device_filepath(),
            d_dsname,
            0,
            count,
            d_output,
            d_success
        );
        cudaDeviceSynchronize();
    }

    state.SetItemsProcessed(state.iterations() * count);
    state.SetBytesProcessed(state.iterations() * count * sizeof(double));

    cudaFree(d_output);
    cudaFree(d_success);
    cudaFreeHost(h_dsname);
}
BENCHMARK(BM_GPU_SequentialRead_Double)
    ->Arg(100)->Arg(500)->Arg(1000)->Arg(5000)
    ->Unit(benchmark::kMillisecond);

static void BM_GPU_SequentialRead_Int32(benchmark::State& state) {
    const size_t count = state.range(0);

    int32_t* d_output;
    bool* d_success;
    cudaMalloc(&d_output, count * sizeof(int32_t));
    cudaMalloc(&d_success, sizeof(bool));

    char* h_dsname;
    char* d_dsname;
    cudaHostAlloc(&h_dsname, 64, cudaHostAllocMapped);
    cudaHostGetDevicePointer(&d_dsname, h_dsname, 0);
    strcpy(h_dsname, "data_1d_int32");

    for (auto _ : state) {
        sequential_read_kernel<int32_t><<<1, 1>>>(
            g_gpu_ctx->device_ctx(),
            g_gpu_ctx->device_filepath(),
            d_dsname,
            0,
            count,
            d_output,
            d_success
        );
        cudaDeviceSynchronize();
    }

    state.SetItemsProcessed(state.iterations() * count);
    state.SetBytesProcessed(state.iterations() * count * sizeof(int32_t));

    cudaFree(d_output);
    cudaFree(d_success);
    cudaFreeHost(h_dsname);
}
BENCHMARK(BM_GPU_SequentialRead_Int32)
    ->Arg(100)->Arg(500)->Arg(1000)->Arg(5000)
    ->Unit(benchmark::kMillisecond);

// Hyperslab benchmarks - tests our GetNextContiguousRun optimization
static void BM_GPU_HyperslabRead_Double(benchmark::State& state) {
    const size_t count = state.range(0);

    double* d_output;
    bool* d_success;
    cudaMalloc(&d_output, count * sizeof(double));
    cudaMalloc(&d_success, sizeof(bool));

    char* h_dsname;
    char* d_dsname;
    cudaHostAlloc(&h_dsname, 64, cudaHostAllocMapped);
    cudaHostGetDevicePointer(&d_dsname, h_dsname, 0);
    strcpy(h_dsname, "data_1d_double");

    for (auto _ : state) {
        hyperslab_read_kernel<double><<<1, 1>>>(
            g_gpu_ctx->device_ctx(),
            g_gpu_ctx->device_filepath(),
            d_dsname,
            0,
            count,
            d_output,
            d_success
        );
        cudaDeviceSynchronize();
    }

    state.SetItemsProcessed(state.iterations() * count);
    state.SetBytesProcessed(state.iterations() * count * sizeof(double));

    cudaFree(d_output);
    cudaFree(d_success);
    cudaFreeHost(h_dsname);
}
BENCHMARK(BM_GPU_HyperslabRead_Double)
    ->Arg(100)->Arg(500)->Arg(1000)->Arg(5000)
    ->Unit(benchmark::kMillisecond);

static void BM_GPU_HyperslabRead_Int32(benchmark::State& state) {
    const size_t count = state.range(0);

    int32_t* d_output;
    bool* d_success;
    cudaMalloc(&d_output, count * sizeof(int32_t));
    cudaMalloc(&d_success, sizeof(bool));

    char* h_dsname;
    char* d_dsname;
    cudaHostAlloc(&h_dsname, 64, cudaHostAllocMapped);
    cudaHostGetDevicePointer(&d_dsname, h_dsname, 0);
    strcpy(h_dsname, "data_1d_int32");

    for (auto _ : state) {
        hyperslab_read_kernel<int32_t><<<1, 1>>>(
            g_gpu_ctx->device_ctx(),
            g_gpu_ctx->device_filepath(),
            d_dsname,
            0,
            count,
            d_output,
            d_success
        );
        cudaDeviceSynchronize();
    }

    state.SetItemsProcessed(state.iterations() * count);
    state.SetBytesProcessed(state.iterations() * count * sizeof(int32_t));

    cudaFree(d_output);
    cudaFree(d_success);
    cudaFreeHost(h_dsname);
}
BENCHMARK(BM_GPU_HyperslabRead_Int32)
    ->Arg(100)->Arg(500)->Arg(1000)->Arg(5000)
    ->Unit(benchmark::kMillisecond);

static void BM_GPU_SequentialWrite_Double(benchmark::State& state) {
    const size_t count = state.range(0);

    // Prepare input data
    double* h_input;
    double* d_input;
    cudaHostAlloc(&h_input, count * sizeof(double), cudaHostAllocMapped);
    cudaHostGetDevicePointer(&d_input, h_input, 0);
    for (size_t i = 0; i < count; ++i) h_input[i] = static_cast<double>(i) * 1.5;

    bool* d_success;
    cudaMalloc(&d_success, sizeof(bool));

    char* h_dsname;
    char* d_dsname;
    cudaHostAlloc(&h_dsname, 64, cudaHostAllocMapped);
    cudaHostGetDevicePointer(&d_dsname, h_dsname, 0);
    strcpy(h_dsname, "data_1d_double");

    for (auto _ : state) {
        state.PauseTiming();
        copy_benchmark_file();
        state.ResumeTiming();

        sequential_write_kernel<double><<<1, 1>>>(
            g_gpu_ctx->device_ctx(),
            g_d_write_filepath,
            d_dsname,
            0,
            count,
            d_input,
            d_success
        );
        cudaDeviceSynchronize();
    }

    state.SetItemsProcessed(state.iterations() * count);
    state.SetBytesProcessed(state.iterations() * count * sizeof(double));

    cudaFree(d_success);
    cudaFreeHost(h_input);
    cudaFreeHost(h_dsname);
}
BENCHMARK(BM_GPU_SequentialWrite_Double)
    ->Arg(100)->Arg(500)->Arg(1000)
    ->Unit(benchmark::kMillisecond);

// TODO: CreateGroup benchmark disabled - causes segfault, needs investigation
// static void BM_GPU_CreateGroup(benchmark::State& state) { ... }

// ============================================================================
// CPU Benchmarks (HDF5 library baseline)
// ============================================================================

#ifdef HDF5_CPU_BASELINE_ENABLED

static void BM_CPU_FileOpen(benchmark::State& state) {
    std::string filepath = bench_utils::BENCH_DATA_FILE;

    for (auto _ : state) {
        hdf5_ref::File file(filepath);
        benchmark::DoNotOptimize(file.id());
    }
}
BENCHMARK(BM_CPU_FileOpen)->Unit(benchmark::kMillisecond);

static void BM_CPU_DatasetOpen(benchmark::State& state) {
    std::string filepath = bench_utils::BENCH_DATA_FILE;

    for (auto _ : state) {
        hdf5_ref::File file(filepath);
        hdf5_ref::Dataset dataset(file.id(), "data_1d_double");
        benchmark::DoNotOptimize(dataset.id());
    }
}
BENCHMARK(BM_CPU_DatasetOpen)->Unit(benchmark::kMillisecond);

static void BM_CPU_SequentialRead_Double(benchmark::State& state) {
    const size_t count = state.range(0);
    std::vector<double> buffer(count);
    std::string filepath = bench_utils::BENCH_DATA_FILE;

    for (auto _ : state) {
        hdf5_ref::File file(filepath);
        hdf5_ref::Dataset dataset(file.id(), "data_1d_double");
        dataset.read_sequential<double>(buffer.data(), 0, count);
        benchmark::DoNotOptimize(buffer.data());
    }

    state.SetItemsProcessed(state.iterations() * count);
    state.SetBytesProcessed(state.iterations() * count * sizeof(double));
}
BENCHMARK(BM_CPU_SequentialRead_Double)
    ->Arg(100)->Arg(500)->Arg(1000)->Arg(5000)
    ->Unit(benchmark::kMillisecond);

static void BM_CPU_SequentialRead_Int32(benchmark::State& state) {
    const size_t count = state.range(0);
    std::vector<int32_t> buffer(count);
    std::string filepath = bench_utils::BENCH_DATA_FILE;

    for (auto _ : state) {
        hdf5_ref::File file(filepath);
        hdf5_ref::Dataset dataset(file.id(), "data_1d_int32");
        dataset.read_sequential<int32_t>(buffer.data(), 0, count);
        benchmark::DoNotOptimize(buffer.data());
    }

    state.SetItemsProcessed(state.iterations() * count);
    state.SetBytesProcessed(state.iterations() * count * sizeof(int32_t));
}
BENCHMARK(BM_CPU_SequentialRead_Int32)
    ->Arg(100)->Arg(500)->Arg(1000)->Arg(5000)
    ->Unit(benchmark::kMillisecond);

static void BM_CPU_SequentialWrite_Double(benchmark::State& state) {
    const size_t count = state.range(0);
    std::vector<double> buffer(count);
    for (size_t i = 0; i < count; ++i) buffer[i] = static_cast<double>(i) * 1.5;

    for (auto _ : state) {
        state.PauseTiming();
        copy_benchmark_file();
        state.ResumeTiming();

        hdf5_ref::File file(WRITE_TEST_FILE, true);
        hdf5_ref::Dataset dataset(file.id(), "data_1d_double");
        dataset.write_sequential<double>(buffer.data(), 0, count);
    }

    state.SetItemsProcessed(state.iterations() * count);
    state.SetBytesProcessed(state.iterations() * count * sizeof(double));
}
BENCHMARK(BM_CPU_SequentialWrite_Double)
    ->Arg(100)->Arg(500)->Arg(1000)
    ->Unit(benchmark::kMillisecond);

// TODO: CreateGroup benchmark disabled - GPU version causes segfault
// static void BM_CPU_CreateGroup(benchmark::State& state) { ... }

#endif // HDF5_CPU_BASELINE_ENABLED

// ============================================================================
// Verification - Ensure GPU and CPU produce identical results
// ============================================================================

#ifdef HDF5_CPU_BASELINE_ENABLED

template<typename T>
bool verify_results(const char* dataset_name, size_t count) {
    // Allocate GPU output buffer (mapped memory for easy access)
    T* h_gpu_output;
    T* d_gpu_output;
    bool* h_success;
    bool* d_success;
    char* h_dsname;
    char* d_dsname;

    cudaHostAlloc(&h_gpu_output, count * sizeof(T), cudaHostAllocMapped);
    cudaHostGetDevicePointer(&d_gpu_output, h_gpu_output, 0);
    cudaHostAlloc(&h_success, sizeof(bool), cudaHostAllocMapped);
    cudaHostGetDevicePointer(&d_success, h_success, 0);
    cudaHostAlloc(&h_dsname, 64, cudaHostAllocMapped);
    cudaHostGetDevicePointer(&d_dsname, h_dsname, 0);
    strcpy(h_dsname, dataset_name);

    // Read with GPU
    sequential_read_kernel<T><<<1, 1>>>(
        g_gpu_ctx->device_ctx(),
        g_gpu_ctx->device_filepath(),
        d_dsname,
        0, count,
        d_gpu_output,
        d_success
    );
    cudaDeviceSynchronize();

    if (!*h_success) {
        fprintf(stderr, "  FAIL: GPU read failed for %s\n", dataset_name);
        cudaFreeHost(h_gpu_output);
        cudaFreeHost(h_success);
        cudaFreeHost(h_dsname);
        return false;
    }

    // Read with CPU HDF5
    std::vector<T> cpu_output(count);
    {
        hdf5_ref::File file(bench_utils::BENCH_DATA_FILE);
        hdf5_ref::Dataset dataset(file.id(), dataset_name);
        dataset.read_sequential<T>(cpu_output.data(), 0, count);
    }

    // Compare results
    bool match = true;
    for (size_t i = 0; i < count && match; ++i) {
        if (!bench_utils::values_equal(h_gpu_output[i], cpu_output[i])) {
            fprintf(stderr, "  FAIL: Mismatch at index %zu: GPU=%g, CPU=%g\n",
                    i, (double)h_gpu_output[i], (double)cpu_output[i]);
            match = false;
        }
    }

    cudaFreeHost(h_gpu_output);
    cudaFreeHost(h_success);
    cudaFreeHost(h_dsname);

    if (match) {
        printf("  PASS: %s (%zu elements)\n", dataset_name, count);
    }
    return match;
}

template<typename T>
bool verify_write_results(const char* dataset_name, size_t count) {
    // Create test data
    std::vector<T> test_data(count);
    for (size_t i = 0; i < count; ++i) {
        test_data[i] = static_cast<T>(i * 2 + 7);  // Different pattern than original
    }

    // Copy file for GPU write test
    copy_benchmark_file();

    // Allocate GPU buffers
    T* h_input;
    T* d_input;
    bool* h_success;
    bool* d_success;
    char* h_dsname;
    char* d_dsname;

    cudaHostAlloc(&h_input, count * sizeof(T), cudaHostAllocMapped);
    cudaHostGetDevicePointer(&d_input, h_input, 0);
    cudaHostAlloc(&h_success, sizeof(bool), cudaHostAllocMapped);
    cudaHostGetDevicePointer(&d_success, h_success, 0);
    cudaHostAlloc(&h_dsname, 64, cudaHostAllocMapped);
    cudaHostGetDevicePointer(&d_dsname, h_dsname, 0);

    memcpy(h_input, test_data.data(), count * sizeof(T));
    strcpy(h_dsname, dataset_name);

    // Write with GPU
    sequential_write_kernel<T><<<1, 1>>>(
        g_gpu_ctx->device_ctx(),
        g_d_write_filepath,
        d_dsname,
        0, count,
        d_input,
        d_success
    );
    cudaDeviceSynchronize();

    if (!*h_success) {
        fprintf(stderr, "  FAIL: GPU write failed for %s\n", dataset_name);
        cudaFreeHost(h_input);
        cudaFreeHost(h_success);
        cudaFreeHost(h_dsname);
        return false;
    }

    // Read back with CPU HDF5 and verify
    std::vector<T> readback(count);
    {
        hdf5_ref::File file(WRITE_TEST_FILE);
        hdf5_ref::Dataset dataset(file.id(), dataset_name);
        dataset.read_sequential<T>(readback.data(), 0, count);
    }

    bool match = true;
    for (size_t i = 0; i < count && match; ++i) {
        if (!bench_utils::values_equal(readback[i], test_data[i])) {
            fprintf(stderr, "  FAIL: Write mismatch at index %zu: got=%g, expected=%g\n",
                    i, (double)readback[i], (double)test_data[i]);
            match = false;
        }
    }

    cudaFreeHost(h_input);
    cudaFreeHost(h_success);
    cudaFreeHost(h_dsname);

    if (match) {
        printf("  PASS: Write %s (%zu elements)\n", dataset_name, count);
    }
    return match;
}

bool run_verification() {
    printf("Verifying GPU results match CPU HDF5...\n");

    bool all_pass = true;
    all_pass &= verify_results<double>("data_1d_double", 100);
    all_pass &= verify_results<int32_t>("data_1d_int32", 100);

    // Also verify larger reads to ensure bulk read works correctly
    all_pass &= verify_results<double>("data_1d_double", 5000);
    all_pass &= verify_results<int32_t>("data_1d_int32", 5000);

    printf("Verifying GPU writes...\n");
    all_pass &= verify_write_results<double>("data_1d_double", 100);

    if (all_pass) {
        printf("All verification tests passed!\n\n");
    } else {
        fprintf(stderr, "Verification FAILED!\n");
    }
    return all_pass;
}

#endif // HDF5_CPU_BASELINE_ENABLED

// ============================================================================
// Main - Initialize GPU context before benchmarks run
// ============================================================================

int main(int argc, char** argv) {
    // Set CUDA limits
    cudaDeviceSetLimit(cudaLimitStackSize, 64 * 1024);
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 256 * 1024 * 1024);

    // Initialize GPU context
    g_gpu_ctx = new bench_utils::GpuBenchContext();
    if (!g_gpu_ctx->init(bench_utils::BENCH_DATA_FILE)) {
        fprintf(stderr, "Failed to initialize GPU context\n");
        return 1;
    }

    printf("GPU context initialized successfully\n");
    printf("Benchmark data file: %s\n\n", bench_utils::BENCH_DATA_FILE);

    // Initialize write test filepath in mapped memory
    cudaHostAlloc(&g_h_write_filepath, 512, cudaHostAllocMapped);
    cudaHostGetDevicePointer(&g_d_write_filepath, g_h_write_filepath, 0);
    strcpy(g_h_write_filepath, WRITE_TEST_FILE);

#ifdef HDF5_CPU_BASELINE_ENABLED
    // Verify correctness before running benchmarks
    if (!run_verification()) {
        cudaFreeHost(g_h_write_filepath);
        delete g_gpu_ctx;
        return 1;
    }
#endif

    // Run Google Benchmark
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();

    // Cleanup
    cudaFreeHost(g_h_write_filepath);
    delete g_gpu_ctx;
    g_gpu_ctx = nullptr;

    return 0;
}
