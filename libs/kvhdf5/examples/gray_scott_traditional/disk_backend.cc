#include "disk_backend.h"

#include "blob_key.h"
#include "gs_kernel.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
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

void EnsureDir(const std::string& dir) {
    if (mkdir(dir.c_str(), 0755) != 0 && errno != EEXIST) {
        std::fprintf(stderr, "mkdir %s: %s\n", dir.c_str(), std::strerror(errno));
        throw std::runtime_error("mkdir failed");
    }
}

// Byte offset of a ping-pong slot within a per-grid working file.
//   [ u_a | u_b | v_a | v_b ]
off_t SlotOffset(Kind kind, Field field) {
    int slot = 0;
    if (kind == Kind::PingPongA && field == Field::U) slot = 0;
    if (kind == Kind::PingPongB && field == Field::U) slot = 1;
    if (kind == Kind::PingPongA && field == Field::V) slot = 2;
    if (kind == Kind::PingPongB && field == Field::V) slot = 3;
    return static_cast<off_t>(slot) * static_cast<off_t>(kBytesPerGrid);
}

void WriteAt(int fd, off_t off, const void* data, size_t bytes,
             const char* where) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    size_t remaining = bytes;
    while (remaining > 0) {
        ssize_t n = pwrite(fd, p, remaining, off);
        if (n < 0) {
            if (errno == EINTR) continue;
            std::fprintf(stderr, "%s: pwrite: %s\n",
                         where, std::strerror(errno));
            throw std::runtime_error(where);
        }
        p         += n;
        off       += n;
        remaining -= static_cast<size_t>(n);
    }
}

} // namespace

void DiskBackend::Setup(int n_grids, GrayScottParams params) {
    n_grids_ = n_grids;
    params_  = params;
    parity_a_ = true;

    EnsureDir(out_dir_);

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

    // Open and pre-size one working file per grid (4 slots).
    work_fds_.assign(n_grids_, -1);
    char name[256];
    for (int g = 0; g < n_grids_; ++g) {
        std::snprintf(name, sizeof(name), "%s/grid_%d_work.bin",
                      out_dir_.c_str(), g);
        int fd = open(name, O_RDWR | O_CREAT | O_TRUNC, 0644);
        if (fd < 0) {
            std::fprintf(stderr, "open %s: %s\n", name, std::strerror(errno));
            throw std::runtime_error("open work file");
        }
        if (ftruncate(fd, static_cast<off_t>(4) * kBytesPerGrid) != 0) {
            std::fprintf(stderr, "ftruncate %s: %s\n", name,
                         std::strerror(errno));
            throw std::runtime_error("ftruncate work file");
        }
        work_fds_[g] = fd;
    }
}

void DiskBackend::SeedInitial() {
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
    CudaCheck(cudaMemcpy(d_u_a_, u.data(), total, cudaMemcpyHostToDevice),
              "seed d_u_a_");
    CudaCheck(cudaMemcpy(d_v_a_, v.data(), total, cudaMemcpyHostToDevice),
              "seed d_v_a_");

    // Persist initial state into the A slots so the on-disk view is consistent
    // with what the iowarp example does at seed time.
    for (int g = 0; g < n_grids_; ++g) {
        size_t off_bytes = static_cast<size_t>(g) * kBytesPerGrid;
        WriteAt(work_fds_[g], SlotOffset(Kind::PingPongA, Field::U),
                reinterpret_cast<const uint8_t*>(u.data()) + off_bytes,
                kBytesPerGrid, "seed u_a");
        WriteAt(work_fds_[g], SlotOffset(Kind::PingPongA, Field::V),
                reinterpret_cast<const uint8_t*>(v.data()) + off_bytes,
                kBytesPerGrid, "seed v_a");
    }
}

void DiskBackend::StepAll() {
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

void DiskBackend::PersistNext() {
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

    off_t u_off = SlotOffset(next_kind, Field::U);
    off_t v_off = SlotOffset(next_kind, Field::V);
    for (int g = 0; g < n_grids_; ++g) {
        size_t off_bytes = static_cast<size_t>(g) * kBytesPerGrid;
        WriteAt(work_fds_[g], u_off,
                reinterpret_cast<const uint8_t*>(h_u_pinned_) + off_bytes,
                kBytesPerGrid, "pwrite u_next");
        WriteAt(work_fds_[g], v_off,
                reinterpret_cast<const uint8_t*>(h_v_pinned_) + off_bytes,
                kBytesPerGrid, "pwrite v_next");
    }
}

void DiskBackend::Snapshot(int step_number) {
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

    char name[256];
    for (int g = 0; g < n_grids_; ++g) {
        size_t off_bytes = static_cast<size_t>(g) * kBytesPerGrid;
        std::snprintf(name, sizeof(name), "%s/snap_step_%d_grid_%d.bin",
                      out_dir_.c_str(), step_number, g);
        int fd = open(name, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd < 0) {
            std::fprintf(stderr, "open %s: %s\n", name, std::strerror(errno));
            throw std::runtime_error("open snapshot file");
        }
        // [ u | v ] in one file per snapshot per grid.
        WriteAt(fd, 0,
                reinterpret_cast<const uint8_t*>(h_u_pinned_) + off_bytes,
                kBytesPerGrid, "snapshot u");
        WriteAt(fd, static_cast<off_t>(kBytesPerGrid),
                reinterpret_cast<const uint8_t*>(h_v_pinned_) + off_bytes,
                kBytesPerGrid, "snapshot v");
        close(fd);
    }
}

void DiskBackend::ReadbackGrid(int grid_idx, float* u_out, float* v_out) {
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

void DiskBackend::Teardown() {
    for (int fd : work_fds_) {
        if (fd >= 0) close(fd);
    }
    work_fds_.clear();
    if (stream_)     { cudaStreamDestroy(stream_); stream_ = nullptr; }
    if (h_u_pinned_) { cudaFreeHost(h_u_pinned_); h_u_pinned_ = nullptr; }
    if (h_v_pinned_) { cudaFreeHost(h_v_pinned_); h_v_pinned_ = nullptr; }
    if (d_u_a_)      { cudaFree(d_u_a_); d_u_a_ = nullptr; }
    if (d_u_b_)      { cudaFree(d_u_b_); d_u_b_ = nullptr; }
    if (d_v_a_)      { cudaFree(d_v_a_); d_v_a_ = nullptr; }
    if (d_v_b_)      { cudaFree(d_v_b_); d_v_b_ = nullptr; }
}

} // namespace gs_trad
