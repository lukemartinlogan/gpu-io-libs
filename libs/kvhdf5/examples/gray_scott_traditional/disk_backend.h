#pragma once

#include "gs_params.h"

#include <cuda_runtime.h>

#include <string>
#include <vector>

namespace gs_trad {

// Disk-backed backend. Layout follows real HPC checkpoint convention:
//
//   <out_dir>/grid_<idx>_work.bin
//       Per-grid working file with 4 ping-pong slots laid out at fixed
//       offsets: [u_a | u_b | v_a | v_b]. Overwritten in place via pwrite
//       every step.
//
//   <out_dir>/snap_step_<n>_grid_<idx>.bin
//       Per-(step, grid) checkpoint files. One file per snapshot per grid,
//       containing [u | v] for that grid. Written once per snapshot via a
//       single pwrite (or O_CREAT+write+close).
class DiskBackend {
public:
    explicit DiskBackend(std::string out_dir) : out_dir_(std::move(out_dir)) {}

    void Setup(int n_grids, GrayScottParams params);
    void SeedInitial();
    void StepAll();
    void Snapshot(int step_number);
    void ReadbackGrid(int grid_idx, float* u_out, float* v_out);
    void Teardown();

private:
    void PersistNext();

    std::string out_dir_;
    int             n_grids_  = 0;
    GrayScottParams params_{};
    bool            parity_a_ = true;

    float* d_u_a_ = nullptr;
    float* d_u_b_ = nullptr;
    float* d_v_a_ = nullptr;
    float* d_v_b_ = nullptr;

    float* h_u_pinned_ = nullptr;
    float* h_v_pinned_ = nullptr;

    cudaStream_t stream_ = nullptr;

    // One open fd per working file (kept open so pwrites avoid open/close
    // syscalls on the hot path).
    std::vector<int> work_fds_;
};

} // namespace gs_trad
