#pragma once

#include "gs_params.h"

namespace gs_trad {

// All backends conform to this shape. We instantiate RunSimulation<Backend>
// twice (once for RAM, once for Disk) so there's no virtual dispatch in the
// hot loop, but the interface stays the same.
//
// Lifecycle: Setup -> SeedInitial -> (StepAll | Snapshot)* -> ReadbackGrid* -> Teardown.
struct GrayScottBackendConcept {
    void Setup(int n_grids, GrayScottParams params);
    void SeedInitial();
    void StepAll();
    void Snapshot(int step_number);
    void ReadbackGrid(int grid_idx, float* u_out, float* v_out);
    void Teardown();
};

} // namespace gs_trad
