#include "ram_backend.h"

#include "gs_kernel.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace gs_trad {

namespace {

void CudaCheck(cudaError_t err, const char* where) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", where, cudaGetErrorString(err));
        throw std::runtime_error(where);
    }
}

} // namespace

void RamBackend::Setup(int n_grids, GrayScottParams params) {
    n_grids_ = n_grids;
    params_  = params;
    parity_a_ = true;

    size_t total_bytes = static_cast<size_t>(n_grids_) * kBytesPerGrid;
    CudaCheck(cudaMalloc(&d_u_a_, total_bytes), "cudaMalloc d_u_a_");
    CudaCheck(cudaMalloc(&d_u_b_, total_bytes), "cudaMalloc d_u_b_");
    CudaCheck(cudaMalloc(&d_v_a_, total_bytes), "cudaMalloc d_v_a_");
    CudaCheck(cudaMalloc(&d_v_b_, total_bytes), "cudaMalloc d_v_b_");
    CudaCheck(cudaMallocHost(reinterpret_cast<void**>(&h_u_pinned_),
                             total_bytes), "cudaMallocHost h_u_pinned_");
    CudaCheck(cudaMallocHost(reinterpret_cast<void**>(&h_v_pinned_),
                             total_bytes), "cudaMallocHost h_v_pinned_");
    CudaCheck(cudaStreamCreate(&stream_), "cudaStreamCreate");
}

void RamBackend::SeedInitial() {
    // u=1.0 everywhere, v=0.0 everywhere, with a central seed of u=0.5, v=0.25.
    std::vector<float> u(static_cast<size_t>(n_grids_) * kCellsPerGrid, 1.0f);
    std::vector<float> v(static_cast<size_t>(n_grids_) * kCellsPerGrid, 0.0f);

    int seed_half = std::max<int>(1, static_cast<int>(kN) / 12);
    int cx = static_cast<int>(kN) / 2;
    int cy = static_cast<int>(kN) / 2;
    for (int g = 0; g < n_grids_; ++g) {
        size_t off = static_cast<size_t>(g) * kCellsPerGrid;
        for (int dy = -seed_half; dy <= seed_half; ++dy) {
            for (int dx = -seed_half; dx <= seed_half; ++dx) {
                int x = cx + dx, y = cy + dy;
                if (x < 0 || y < 0 ||
                    x >= static_cast<int>(kN) || y >= static_cast<int>(kN))
                    continue;
                u[off + static_cast<size_t>(y) * kN + x] = 0.5f;
                v[off + static_cast<size_t>(y) * kN + x] = 0.25f;
            }
        }
    }
    size_t total = static_cast<size_t>(n_grids_) * kBytesPerGrid;
    // Seed slot A (the initial "curr").
    CudaCheck(cudaMemcpy(d_u_a_, u.data(), total, cudaMemcpyHostToDevice),
              "seed d_u_a_");
    CudaCheck(cudaMemcpy(d_v_a_, v.data(), total, cudaMemcpyHostToDevice),
              "seed d_v_a_");

    // Also persist the initial state into the map so it's observable through
    // the storage interface, matching what iowarp's HostSeedInitialConditions
    // does (it writes the initial blob through the storage API).
    size_t bytes = total;
    for (int g = 0; g < n_grids_; ++g) {
        size_t off = static_cast<size_t>(g) * kBytesPerGrid;
        BlobKey ku{static_cast<uint32_t>(g), 0, Field::U, Kind::PingPongA};
        BlobKey kv{static_cast<uint32_t>(g), 0, Field::V, Kind::PingPongA};
        auto& bu = store_[ku]; bu.resize(kBytesPerGrid);
        auto& bv = store_[kv]; bv.resize(kBytesPerGrid);
        std::memcpy(bu.data(), reinterpret_cast<const uint8_t*>(u.data()) + off,
                    kBytesPerGrid);
        std::memcpy(bv.data(), reinterpret_cast<const uint8_t*>(v.data()) + off,
                    kBytesPerGrid);
    }
    (void)bytes;
}

void RamBackend::StepAll() {
    float* d_u_curr = parity_a_ ? d_u_a_ : d_u_b_;
    float* d_v_curr = parity_a_ ? d_v_a_ : d_v_b_;
    float* d_u_next = parity_a_ ? d_u_b_ : d_u_a_;
    float* d_v_next = parity_a_ ? d_v_b_ : d_v_a_;

    LaunchGsStep(d_u_curr, d_v_curr, d_u_next, d_v_next,
                 n_grids_, params_, stream_);
    CudaCheck(cudaGetLastError(), "LaunchGsStep");

    PersistNext();
    parity_a_ = !parity_a_;
}

void RamBackend::PersistNext() {
    float* d_u_next = parity_a_ ? d_u_b_ : d_u_a_;
    float* d_v_next = parity_a_ ? d_v_b_ : d_v_a_;
    Kind   next_kind = parity_a_ ? Kind::PingPongB : Kind::PingPongA;

    size_t total = static_cast<size_t>(n_grids_) * kBytesPerGrid;
    CudaCheck(cudaMemcpyAsync(h_u_pinned_, d_u_next, total,
                              cudaMemcpyDeviceToHost, stream_),
              "memcpy u_next");
    CudaCheck(cudaMemcpyAsync(h_v_pinned_, d_v_next, total,
                              cudaMemcpyDeviceToHost, stream_),
              "memcpy v_next");
    CudaCheck(cudaStreamSynchronize(stream_), "stream sync persist");

    for (int g = 0; g < n_grids_; ++g) {
        size_t off_bytes = static_cast<size_t>(g) * kBytesPerGrid;
        BlobKey ku{static_cast<uint32_t>(g), 0, Field::U, next_kind};
        BlobKey kv{static_cast<uint32_t>(g), 0, Field::V, next_kind};
        auto& bu = store_[ku]; bu.resize(kBytesPerGrid);
        auto& bv = store_[kv]; bv.resize(kBytesPerGrid);
        std::memcpy(bu.data(),
                    reinterpret_cast<const uint8_t*>(h_u_pinned_) + off_bytes,
                    kBytesPerGrid);
        std::memcpy(bv.data(),
                    reinterpret_cast<const uint8_t*>(h_v_pinned_) + off_bytes,
                    kBytesPerGrid);
    }
}

void RamBackend::Snapshot(int step_number) {
    // Snapshot reads the current state out of the device "curr" buffers and
    // stores it under Kind::Snapshot keys. Reuses the pinned staging.
    float* d_u_curr = parity_a_ ? d_u_a_ : d_u_b_;
    float* d_v_curr = parity_a_ ? d_v_a_ : d_v_b_;

    size_t total = static_cast<size_t>(n_grids_) * kBytesPerGrid;
    CudaCheck(cudaMemcpyAsync(h_u_pinned_, d_u_curr, total,
                              cudaMemcpyDeviceToHost, stream_),
              "memcpy snapshot u");
    CudaCheck(cudaMemcpyAsync(h_v_pinned_, d_v_curr, total,
                              cudaMemcpyDeviceToHost, stream_),
              "memcpy snapshot v");
    CudaCheck(cudaStreamSynchronize(stream_), "stream sync snapshot");

    for (int g = 0; g < n_grids_; ++g) {
        size_t off_bytes = static_cast<size_t>(g) * kBytesPerGrid;
        BlobKey ku{static_cast<uint32_t>(g),
                   static_cast<uint16_t>(step_number),
                   Field::U, Kind::Snapshot};
        BlobKey kv{static_cast<uint32_t>(g),
                   static_cast<uint16_t>(step_number),
                   Field::V, Kind::Snapshot};
        auto& bu = store_[ku]; bu.resize(kBytesPerGrid);
        auto& bv = store_[kv]; bv.resize(kBytesPerGrid);
        std::memcpy(bu.data(),
                    reinterpret_cast<const uint8_t*>(h_u_pinned_) + off_bytes,
                    kBytesPerGrid);
        std::memcpy(bv.data(),
                    reinterpret_cast<const uint8_t*>(h_v_pinned_) + off_bytes,
                    kBytesPerGrid);
    }
}

void RamBackend::ReadbackGrid(int grid_idx, float* u_out, float* v_out) {
    float* d_u_curr = parity_a_ ? d_u_a_ : d_u_b_;
    float* d_v_curr = parity_a_ ? d_v_a_ : d_v_b_;
    size_t off_bytes = static_cast<size_t>(grid_idx) * kBytesPerGrid;
    CudaCheck(cudaMemcpy(u_out,
                         reinterpret_cast<const uint8_t*>(d_u_curr) + off_bytes,
                         kBytesPerGrid, cudaMemcpyDeviceToHost),
              "readback u");
    CudaCheck(cudaMemcpy(v_out,
                         reinterpret_cast<const uint8_t*>(d_v_curr) + off_bytes,
                         kBytesPerGrid, cudaMemcpyDeviceToHost),
              "readback v");
}

void RamBackend::Teardown() {
    if (stream_)     { cudaStreamDestroy(stream_); stream_ = nullptr; }
    if (h_u_pinned_) { cudaFreeHost(h_u_pinned_); h_u_pinned_ = nullptr; }
    if (h_v_pinned_) { cudaFreeHost(h_v_pinned_); h_v_pinned_ = nullptr; }
    if (d_u_a_)      { cudaFree(d_u_a_); d_u_a_ = nullptr; }
    if (d_u_b_)      { cudaFree(d_u_b_); d_u_b_ = nullptr; }
    if (d_v_a_)      { cudaFree(d_v_a_); d_v_a_ = nullptr; }
    if (d_v_b_)      { cudaFree(d_v_b_); d_v_b_ = nullptr; }
    store_.clear();
}

} // namespace gs_trad
