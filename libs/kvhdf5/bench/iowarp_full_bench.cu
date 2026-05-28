// Full-simulation iowarp benchmark. Each timed iteration runs kFullSteps
// steps of the Gray-Scott kernel + kFullSnapInterval-cadence snapshots, all
// through the kvhdf5 CTE-backed Dataset<> API on a single grid (<<<1,1>>>
// since GpuCteBlobStore is single-warp-only as of writing).
//
// The fixture's SetUp does the entire iowarp ceremony — runtime init,
// GpuCteBlobStore creation, managed-memory container, scene build, seeding —
// so the timed loop is purely the simulation itself. Per-iter cost includes
// LaunchAndPoll's own per-launch overhead (pinned status word + stream
// create/destroy); that matches the example's behavior and is what the
// advisor's "GPU -> RUNTIME" path actually costs end-to-end.

#include <benchmark/benchmark.h>

#include "iowarp_step.h"
#include "gray_scott_host.h"

extern void GrayScottInitializeRuntime();
extern wrp_cte::core::TagId g_gpu_cte_tag_id;
extern wrp_cte::core::TagId g_gpu_pool_tag_id;
extern chi::PoolId g_gpu_cte_pool_id;

#include <optional>
#include <utility>
#include <vector>

#if !HSHM_IS_GPU

namespace {

constexpr gs_iowarp::GrayScottParams kIowarpBenchParams{
    /*Du=*/0.16f, /*Dv=*/0.08f, /*F=*/0.055f, /*k=*/0.062f, /*dt=*/1.0f
};

constexpr int kFullSteps        = 105;
constexpr int kFullSnapInterval = 15;

class GsIowarpFullFixture : public benchmark::Fixture {
public:
    void SetUp(benchmark::State& state) override {
        GrayScottInitializeRuntime();

        auto store = kvhdf5::GpuCteBlobStore::Create(
            g_gpu_pool_tag_id, g_gpu_cte_pool_id);
        if (!store.IsValid()) {
            state.SkipWithError("GpuCteBlobStore::Create failed");
            return;
        }
        if (!alloc_box_.Setup()) {
            state.SkipWithError("alloc_box setup failed");
            return;
        }
        if (!cont_box_.Setup(std::move(store), alloc_box_.allocator)) {
            state.SkipWithError("container setup failed");
            return;
        }

        snap_steps_.clear();
        for (int s = kFullSnapInterval; s <= kFullSteps;
             s += kFullSnapInterval) {
            snap_steps_.push_back(s);
        }

        ids_ = gs::HostBuildScene(cont_box_.ptr, *alloc_box_.allocator,
                                   gs_iowarp::cfg::kN, snap_steps_);

        if (!gs::HostSeedInitialConditions(cont_box_.ptr,
                                            ids_.u_a, ids_.v_a,
                                            gs_iowarp::cfg::kN)) {
            state.SkipWithError("HostSeedInitialConditions failed");
            return;
        }

        u_curr_ = ids_.u_a; u_next_ = ids_.u_b;
        v_curr_ = ids_.v_a; v_next_ = ids_.v_b;
    }

    void TearDown(benchmark::State&) override {
        cont_box_.Teardown();
        alloc_box_.Teardown();
    }

protected:
    gs_iowarp::ManagedAllocBox     alloc_box_;
    gs_iowarp::ManagedContainerBox cont_box_;
    gs::GrayScottIds               ids_;
    std::vector<int>               snap_steps_;
    kvhdf5::DatasetId              u_curr_, u_next_, v_curr_, v_next_;
};

BENCHMARK_DEFINE_F(GsIowarpFullFixture, BM_GrayScott_Full_Iowarp)
(benchmark::State& state) {
    for (auto _ : state) {
        size_t snap_idx = 0;
        for (int step = 1; step <= kFullSteps; ++step) {
            int rc = gs_iowarp::LaunchAndPoll(
                [&](chi::IpcManagerGpuInfo gpu_info,
                    cudaStream_t stream, int* d_status) {
                    gs_iowarp::IowarpKernelGrayScottStep<<<1, 1, 0, stream>>>(
                        gpu_info, cont_box_.ptr,
                        u_curr_, v_curr_, u_next_, v_next_,
                        kIowarpBenchParams, d_status);
                });
            if (rc != 1) {
                state.SkipWithError("step kernel failed");
                return;
            }
            std::swap(u_curr_, u_next_);
            std::swap(v_curr_, v_next_);

            if (snap_idx < snap_steps_.size()
                && step == snap_steps_[snap_idx]) {
                if (!gs::HostTakeSnapshot(cont_box_.ptr, u_curr_, v_curr_,
                                           ids_.snap_u_ids[snap_idx],
                                           ids_.snap_v_ids[snap_idx],
                                           gs_iowarp::cfg::kN)) {
                    state.SkipWithError("snapshot failed");
                    return;
                }
                ++snap_idx;
            }
        }
    }

    // Per iteration: kFullSteps * (u_next + v_next) of per-step persistence,
    // plus snap_steps_.size() * (u_snap + v_snap) of snapshot data.
    size_t bytes_per_iter =
        (static_cast<size_t>(kFullSteps) * 2 +
         snap_steps_.size() * 2) *
        gs_iowarp::cfg::kBytesPerGrid;
    state.SetBytesProcessed(state.iterations() *
                            static_cast<int64_t>(bytes_per_iter));
    state.SetItemsProcessed(state.iterations() * kFullSteps);
}
BENCHMARK_REGISTER_F(GsIowarpFullFixture, BM_GrayScott_Full_Iowarp)
    ->Unit(benchmark::kMillisecond)
    ->MinTime(0.1);

} // namespace

#endif // !HSHM_IS_GPU
