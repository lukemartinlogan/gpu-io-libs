// Traditional CUDA Gray-Scott driver. Same simulation as
// examples/gray_scott_iowarp/, but storage goes GPU -> DRAM -> {map | disk}
// instead of through the iowarp CTE runtime.
//
// Usage:
//   kvhdf5_gray_scott_traditional <backend> <num_steps> <snap_interval>
//                                 [<n_grids>] [<out_dir>]
//
//   <backend>       ram | disk
//   <out_dir>       only used by disk (default: ./gs_trad_out)
//   <n_grids>       number of independent grids (default: 1)

#include "disk_backend.h"
#include "gs_params.h"
#include "heatmap.h"
#include "ram_backend.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

template <typename Backend>
int RunSimulation(Backend& backend, int n_grids, int num_steps,
                  int snap_interval) {
    gs_trad::GrayScottParams params{0.16f, 0.08f, 0.055f, 0.062f, 1.0f};

    backend.Setup(n_grids, params);
    backend.SeedInitial();

    std::printf("  initial state seeded; running %d steps "
                "(snapshot every %d) on %d grid(s)\n",
                num_steps, snap_interval, n_grids);

    auto t_start = std::chrono::steady_clock::now();
    for (int step = 1; step <= num_steps; ++step) {
        backend.StepAll();
        if (step % snap_interval == 0) {
            backend.Snapshot(step);
            std::printf("  step %5d: snapshot written\n", step);
        } else if (step % 200 == 0) {
            std::printf("  step %5d: ok\n", step);
        }
    }
    auto t_end = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t_end - t_start).count();
    std::printf("\nsimulation complete: %d steps in %.2f s (%.2f ms/step)\n",
                num_steps, secs, secs * 1000.0 / num_steps);

    std::vector<float> u_final(gs_trad::kCellsPerGrid);
    std::vector<float> v_final(gs_trad::kCellsPerGrid);
    backend.ReadbackGrid(0, u_final.data(), v_final.data());
    gs_common::DumpHeatmap(v_final.data(), gs_trad::kN, "Final state (grid 0)");

    backend.Teardown();
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    const char* backend_name   = (argc >= 2) ? argv[1] : "ram";
    int         num_steps      = (argc >= 3) ? std::atoi(argv[2]) : 1500;
    int         snap_interval  = (argc >= 4) ? std::atoi(argv[3]) : 500;
    int         n_grids        = (argc >= 5) ? std::atoi(argv[4]) : 1;
    const char* out_dir        = (argc >= 6) ? argv[5] : "./gs_trad_out";

    if (num_steps     <= 0) num_steps     = 1500;
    if (snap_interval <= 0) snap_interval = 500;
    if (n_grids       <= 0) n_grids       = 1;

    std::printf("Gray-Scott traditional (no iowarp)\n");
    std::printf("  backend:        %s\n", backend_name);
    std::printf("  grid:           %u x %u (float32)\n",
                gs_trad::kN, gs_trad::kN);
    std::printf("  grids:          %d\n", n_grids);
    std::printf("  steps:          %d\n", num_steps);
    std::printf("  snap interval:  %d\n", snap_interval);

    if (std::strcmp(backend_name, "ram") == 0) {
        gs_trad::RamBackend backend;
        return RunSimulation(backend, n_grids, num_steps, snap_interval);
    } else if (std::strcmp(backend_name, "disk") == 0) {
        std::printf("  out dir:        %s\n", out_dir);
        gs_trad::DiskBackend backend(out_dir);
        return RunSimulation(backend, n_grids, num_steps, snap_interval);
    } else {
        std::fprintf(stderr, "unknown backend '%s' (expected ram|disk)\n",
                     backend_name);
        return 2;
    }
}
