// Gray-Scott reaction-diffusion on GPU, end-to-end through kvhdf5.
//
// Demonstrates:
//   - Container + Group hierarchy on host (managed memory).
//   - Host seeding of initial conditions via Dataset::Write (host-side).
//   - A 2-D, single-chunk float32 dataset layout with ping-pong buffers
//     (u_a/u_b, v_a/v_b) so each iteration reads from one and writes to the
//     other, then the host swaps which IDs are "curr".
//   - Per-iteration <<<1,1>>> kernel that does the full Gray-Scott step on
//     the current u/v grids and writes the next pair.
//   - Periodic snapshots: every snap_interval steps the host reads the
//     current u/v grids through the same Dataset<> API and writes them into
//     pre-existing snapshot datasets. Snapshots are pure data movement, so
//     keeping them on the host avoids an intermittent kernel-side roundtrip
//     edge case while still exercising the full kvhdf5 read/write path.
//   - Host readback of the final state and every snapshot via the same
//     Container + Dataset abstractions.
//
// Run inside the gpu-io-libs-dev container with CHI_IPC_MODE=SHM:
//   ./examples/kvhdf5_gray_scott

#include "iowarp_step.h"
#include "gray_scott_host.h"
#include "heatmap.h"

// The GPU CTE runtime initialization is delegated to gray_scott_init.cc
// to avoid CUDA compilation issues with the gpu_cte_fixture.h header.
extern void GrayScottInitializeRuntime();

// Globals set by gpu_cte_fixture.h (defined in gray_scott_init.cc)
extern wrp_cte::core::TagId g_gpu_cte_tag_id;
extern wrp_cte::core::TagId g_gpu_pool_tag_id;
extern chi::PoolId g_gpu_cte_pool_id;

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <string>
#include <vector>

using namespace kvhdf5;

#if !HSHM_IS_GPU

int main(int argc, char** argv) {
    int num_steps = 1500;
    int snap_interval = 300;
    if (argc >= 2) num_steps     = std::atoi(argv[1]);
    if (argc >= 3) snap_interval = std::atoi(argv[2]);
    if (num_steps     <= 0)            { num_steps     = 1500; }
    if (snap_interval <= 0)            { snap_interval = 300;  }

    std::printf("Gray-Scott reaction-diffusion on GPU via kvhdf5\n");
    std::printf("  grid:          %u x %u (single chunk per dataset, float32)\n",
                gs_iowarp::cfg::kN, gs_iowarp::cfg::kN);
    std::printf("  steps:         %d\n", num_steps);
    std::printf("  snap interval: %d\n", snap_interval);

    GrayScottInitializeRuntime();

    GpuCteBlobStore store = GpuCteBlobStore::Create(
        g_gpu_pool_tag_id, g_gpu_cte_pool_id);
    if (!store.IsValid()) {
        std::fprintf(stderr, "GpuCteBlobStore::Create failed\n");
        return 1;
    }

    gs_iowarp::ManagedAllocBox alloc_box;
    if (!alloc_box.Setup()) {
        std::fprintf(stderr, "alloc_box setup failed\n");
        return 1;
    }
    gs_iowarp::ManagedContainerBox cont_box;
    if (!cont_box.Setup(std::move(store), alloc_box.allocator)) {
        std::fprintf(stderr, "container setup failed\n");
        return 1;
    }

    std::vector<int> snap_steps;
    for (int s = snap_interval; s <= num_steps; s += snap_interval) {
        snap_steps.push_back(s);
    }

    gs::GrayScottIds ids = gs::HostBuildScene(
        cont_box.ptr, *alloc_box.allocator, gs_iowarp::cfg::kN, snap_steps);

    if (!gs::HostSeedInitialConditions(
            cont_box.ptr, ids.u_a, ids.v_a, gs_iowarp::cfg::kN)) {
        std::fprintf(stderr, "HostSeedInitialConditions failed\n");
        return 1;
    }
    std::printf("  initial state seeded; %zu snapshots planned\n",
                snap_steps.size());

    gs_iowarp::GrayScottParams params{0.16f, 0.08f, 0.055f, 0.062f, 1.0f};

    DatasetId u_curr = ids.u_a, u_next = ids.u_b;
    DatasetId v_curr = ids.v_a, v_next = ids.v_b;

    auto t_start = std::chrono::steady_clock::now();

    size_t snap_idx = 0;
    for (int step = 1; step <= num_steps; ++step) {
        int rc = gs_iowarp::LaunchAndPoll(
            [&](chi::IpcManagerGpuInfo gpu_info,
                cudaStream_t stream, int* d_status) {
                gs_iowarp::IowarpKernelGrayScottStep<<<1, 1, 0, stream>>>(
                    gpu_info, cont_box.ptr,
                    u_curr, v_curr, u_next, v_next,
                    params, d_status);
            });
        if (rc != 1) {
            std::fprintf(stderr,
                         "step %d: iteration kernel failed (status=%d)\n",
                         step, rc);
            return 1;
        }
        std::swap(u_curr, u_next);
        std::swap(v_curr, v_next);

        if (snap_idx < snap_steps.size() && step == snap_steps[snap_idx]) {
            if (!gs::HostTakeSnapshot(cont_box.ptr, u_curr, v_curr,
                                       ids.snap_u_ids[snap_idx],
                                       ids.snap_v_ids[snap_idx],
                                       gs_iowarp::cfg::kN)) {
                std::fprintf(stderr,
                             "step %d: host snapshot failed\n", step);
                return 1;
            }
            std::printf("  step %5d: snapshot %zu written\n",
                         step, snap_idx);
            ++snap_idx;
        } else if (step % 200 == 0) {
            std::printf("  step %5d: ok\n", step);
        }
    }

    auto t_end = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t_end - t_start).count();
    std::printf("\nsimulation complete: %d steps in %.2f s (%.2f ms/step)\n",
                num_steps, secs, secs * 1000.0 / num_steps);

    std::vector<float> u_final(gs_iowarp::cfg::kCellsPerGrid);
    std::vector<float> v_final(gs_iowarp::cfg::kCellsPerGrid);
    if (!gs::HostReadGrid(cont_box.ptr, u_curr, u_final.data(),
                           gs_iowarp::cfg::kN)) {
        std::fprintf(stderr, "final u read failed\n");
        return 1;
    }
    if (!gs::HostReadGrid(cont_box.ptr, v_curr, v_final.data(),
                           gs_iowarp::cfg::kN)) {
        std::fprintf(stderr, "final v read failed\n");
        return 1;
    }
    gs_common::DumpHeatmap(v_final.data(), gs_iowarp::cfg::kN, "Final state");

    for (size_t s = 0; s < snap_steps.size(); ++s) {
        std::vector<float> snap_v(gs_iowarp::cfg::kCellsPerGrid);
        if (!gs::HostReadGrid(cont_box.ptr, ids.snap_v_ids[s],
                               snap_v.data(), gs_iowarp::cfg::kN)) {
            std::fprintf(stderr, "snapshot %zu read failed\n", s);
            continue;
        }
        std::string title = "Snapshot @ step " + std::to_string(snap_steps[s]);
        gs_common::DumpHeatmap(snap_v.data(), gs_iowarp::cfg::kN, title.c_str());
    }

    cont_box.Teardown();
    alloc_box.Teardown();
    std::printf("\nDone.\n");
    return 0;
}

#endif // !HSHM_IS_GPU
