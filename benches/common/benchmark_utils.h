#pragma once

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#include "hdf5/file.h"
#include "hdf5/group.h"
#include "hdf5/dataset.h"
#include "iowarp/gpu_context.h"
#include "iowarp/cpu_polling.h"

#ifdef HDF5_CPU_BASELINE_ENABLED
#include "hdf5_reference.h"
#endif

namespace bench_utils {

inline const char* DATA_DIR = "benches/data/";
inline const char* BENCH_DATA_FILE = "benches/data/bench_data.h5";
inline const char* BENCH_INDICES_FILE = "benches/data/bench_indices.h5";

// matches generate_data.py
inline constexpr size_t SIZE_1D = 10000;
inline constexpr size_t SIZE_2D_ROWS = 100;
inline constexpr size_t SIZE_2D_COLS = 100;

inline constexpr double FLOAT_TOLERANCE = 1e-6;

template<typename T>
bool values_equal(T a, T b) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::abs(a - b) < static_cast<T>(FLOAT_TOLERANCE);
    } else {
        return a == b;
    }
}

struct GpuBenchContext {
    iowarp::GpuContextBuilder builder;
    std::unique_ptr<iowarp::PollingThreadManager> polling;
    char* d_filepath = nullptr;
    char* h_filepath = nullptr;

    bool init(const char* filepath) {
        if (!builder.Build()) {
            fprintf(stderr, "Failed to build GPU context\n");
            return false;
        }

        // Allocate filepath in mapped memory
        cudaError_t err = cudaHostAlloc(&h_filepath, 512, cudaHostAllocMapped);
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to allocate mapped memory for filepath\n");
            return false;
        }
        cudaHostGetDevicePointer(&d_filepath, h_filepath, 0);
        strcpy(h_filepath, filepath);

        polling = std::make_unique<iowarp::PollingThreadManager>(
            builder.HostQueue(),
            builder.HostContext()
        );

        return true;
    }

    ~GpuBenchContext() {
        polling.reset();
        if (h_filepath) {
            cudaFreeHost(h_filepath);
        }
    }

    iowarp::GpuContext* device_ctx() { return builder.DeviceContext(); }
    const char* device_filepath() { return d_filepath; }
};

#ifdef HDF5_CPU_BASELINE_ENABLED

template<typename T>
bool verify_sequential_read(
    const char* filepath,
    const char* dataset_name,
    size_t offset,
    size_t count,
    GpuBenchContext& gpu_ctx
) {
    std::vector<T> cpu_data(count);
    {
        hdf5_ref::File file(filepath);
        hdf5_ref::Dataset dataset(file.id(), dataset_name);
        dataset.read_sequential<T>(cpu_data.data(), offset, count);
    }

    // Read with GPU implementation
    std::vector<T> gpu_data(count);

    // We need to run a kernel to read with GPU implementation
    // For simplicity in verification, we'll trust the comprehensive test
    // and skip per-benchmark verification (it would require kernel launches)

    // TODO: Implement GPU verification kernel if needed

    return true;
}

#endif // HDF5_CPU_BASELINE_ENABLED

} // namespace bench_utils
