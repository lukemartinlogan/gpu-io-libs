#pragma once

#include "blob_key.h"
#include "gs_params.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace gs_trad {

// In-memory blob store backend.
//
// Per-step path: compute kernel writes to device "next" buffers, host
// cudaMemcpyAsync's both fields into pinned staging, then assigns into the
// std::unordered_map keyed by BlobKey (PingPongA/B slots).
//
// Snapshots reuse the same memcpy + assign path with Kind::Snapshot keys.
class RamBackend {
public:
    void Setup(int n_grids, GrayScottParams params);
    void SeedInitial();
    void StepAll();
    void Snapshot(int step_number);
    void ReadbackGrid(int grid_idx, float* u_out, float* v_out);
    void Teardown();

private:
    void PersistNext();   // memcpy device next-slot to host + store in map

    int               n_grids_     = 0;
    GrayScottParams   params_{};
    bool              parity_a_    = true;  // true: curr=A, next=B; flips each step

    // Flat device buffers of size n_grids_ * kBytesPerGrid each.
    float* d_u_a_ = nullptr;
    float* d_u_b_ = nullptr;
    float* d_v_a_ = nullptr;
    float* d_v_b_ = nullptr;

    // Pinned staging buffers for download (one per field, sized for all grids).
    float* h_u_pinned_ = nullptr;
    float* h_v_pinned_ = nullptr;

    cudaStream_t stream_ = nullptr;

    std::unordered_map<BlobKey, std::vector<uint8_t>, BlobKeyHash> store_;
};

} // namespace gs_trad
