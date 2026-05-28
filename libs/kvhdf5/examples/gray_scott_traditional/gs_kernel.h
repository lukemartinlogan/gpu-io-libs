#pragma once

#include "gs_params.h"

#include <cuda_runtime.h>

namespace gs_trad {

// Launches gs_step_kernel on `n_grids` independent grids. Each grid occupies
// kCellsPerGrid contiguous floats in the flat u/v arrays. blockIdx.x picks the
// grid; a single thread per block does the full step. The kernel is written
// for arbitrary <<<n_grids, T>>> — only the launch shape changes when we want
// more parallelism per grid later.
void LaunchGsStep(const float* d_u_in, const float* d_v_in,
                  float* d_u_out, float* d_v_out,
                  int n_grids, GrayScottParams params,
                  cudaStream_t stream);

} // namespace gs_trad
