// Google Benchmark sweep for the traditional (non-iowarp) Gray-Scott
// backends. Each timed iteration is one StepAll() — compute kernel +
// per-grid persistence. Snapshots are intentionally NOT exercised in the
// timed loop: the advisor's interest is the per-step GPU -> DRAM -> STORAGE
// vs GPU -> RUNTIME path, so snapshots would only add noise.
//
// The iowarp backend will be added here once <<<blocks, threads>>>
// concurrency is unblocked on the GpuCteBlobStore side. Until then the
// sweep is single-sided.

#include <benchmark/benchmark.h>

#include "disk_backend.h"
#include "gs_params.h"
#include "ram_backend.h"

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>

namespace {

constexpr gs_trad::GrayScottParams kBenchParams{
    /*Du=*/0.16f, /*Dv=*/0.08f, /*F=*/0.055f, /*k=*/0.062f, /*dt=*/1.0f
};

// Per-step storage volume: each step writes u_next and v_next for every
// grid. Used for SetBytesProcessed so the report shows MB/s.
size_t StepBytes(int n_grids) {
    return static_cast<size_t>(n_grids) * 2 * gs_trad::kBytesPerGrid;
}

// ---------------------------------------------------------------------------
// RAM backend
// ---------------------------------------------------------------------------

class GsRamFixture : public benchmark::Fixture {
public:
    void SetUp(benchmark::State& state) override {
        n_grids_ = static_cast<int>(state.range(0));
        backend_.Setup(n_grids_, kBenchParams);
        backend_.SeedInitial();
    }
    void TearDown(benchmark::State&) override {
        backend_.Teardown();
    }
protected:
    gs_trad::RamBackend backend_;
    int n_grids_ = 0;
};

BENCHMARK_DEFINE_F(GsRamFixture, BM_GrayScott_TraditionalRam)
(benchmark::State& state) {
    for (auto _ : state) {
        backend_.StepAll();
    }
    state.SetBytesProcessed(state.iterations() * StepBytes(n_grids_));
    state.SetItemsProcessed(state.iterations() *
                            static_cast<int64_t>(n_grids_));
}
BENCHMARK_REGISTER_F(GsRamFixture, BM_GrayScott_TraditionalRam)
    ->Arg(1)->Arg(2)->Arg(4)->Arg(8)->Arg(16)->Arg(32)
    ->Unit(benchmark::kMicrosecond);

// ---------------------------------------------------------------------------
// Disk backend
// ---------------------------------------------------------------------------

class GsDiskFixture : public benchmark::Fixture {
public:
    void SetUp(benchmark::State& state) override {
        n_grids_ = static_cast<int>(state.range(0));

        char tmpl[] = "/tmp/gs_bench_disk_XXXXXX";
        const char* dir = ::mkdtemp(tmpl);
        if (!dir) {
            state.SkipWithError("mkdtemp failed");
            return;
        }
        out_dir_ = dir;

        backend_ = std::make_unique<gs_trad::DiskBackend>(out_dir_);
        backend_->Setup(n_grids_, kBenchParams);
        backend_->SeedInitial();
    }
    void TearDown(benchmark::State&) override {
        if (backend_) {
            backend_->Teardown();
            backend_.reset();
        }
        if (!out_dir_.empty()) {
            // mkdtemp-generated path, no special chars — safe for shell quoting.
            std::string cmd = "rm -rf '" + out_dir_ + "'";
            (void)std::system(cmd.c_str());
            out_dir_.clear();
        }
    }
protected:
    std::unique_ptr<gs_trad::DiskBackend> backend_;
    std::string out_dir_;
    int n_grids_ = 0;
};

BENCHMARK_DEFINE_F(GsDiskFixture, BM_GrayScott_TraditionalDisk)
(benchmark::State& state) {
    for (auto _ : state) {
        backend_->StepAll();
    }
    state.SetBytesProcessed(state.iterations() * StepBytes(n_grids_));
    state.SetItemsProcessed(state.iterations() *
                            static_cast<int64_t>(n_grids_));
}
BENCHMARK_REGISTER_F(GsDiskFixture, BM_GrayScott_TraditionalDisk)
    ->Arg(1)->Arg(2)->Arg(4)->Arg(8)->Arg(16)->Arg(32)
    ->Unit(benchmark::kMicrosecond);

// ---------------------------------------------------------------------------
// Full-simulation variants — same shape as the iowarp full benchmark for an
// apples-to-apples comparison. n_grids=1, kFullSteps steps per iter with a
// snapshot every kFullSnapInterval steps (matching the iowarp fixture).
// ---------------------------------------------------------------------------

constexpr int kFullSteps        = 105;
constexpr int kFullSnapInterval = 15;

class GsRamFullFixture : public benchmark::Fixture {
public:
    void SetUp(benchmark::State&) override {
        backend_.Setup(1, kBenchParams);
        backend_.SeedInitial();
    }
    void TearDown(benchmark::State&) override {
        backend_.Teardown();
    }
protected:
    gs_trad::RamBackend backend_;
};

BENCHMARK_DEFINE_F(GsRamFullFixture, BM_GrayScott_Full_TraditionalRam)
(benchmark::State& state) {
    for (auto _ : state) {
        for (int step = 1; step <= kFullSteps; ++step) {
            backend_.StepAll();
            if (step % kFullSnapInterval == 0) {
                backend_.Snapshot(step);
            }
        }
    }
    int n_snaps = kFullSteps / kFullSnapInterval;
    size_t bytes_per_iter =
        (static_cast<size_t>(kFullSteps) * 2 +
         static_cast<size_t>(n_snaps) * 2) *
        gs_trad::kBytesPerGrid;
    state.SetBytesProcessed(state.iterations() *
                            static_cast<int64_t>(bytes_per_iter));
    state.SetItemsProcessed(state.iterations() * kFullSteps);
}
BENCHMARK_REGISTER_F(GsRamFullFixture, BM_GrayScott_Full_TraditionalRam)
    ->Unit(benchmark::kMillisecond)
    ->MinTime(0.1);

class GsDiskFullFixture : public benchmark::Fixture {
public:
    void SetUp(benchmark::State& state) override {
        char tmpl[] = "/tmp/gs_bench_disk_full_XXXXXX";
        const char* dir = ::mkdtemp(tmpl);
        if (!dir) {
            state.SkipWithError("mkdtemp failed");
            return;
        }
        out_dir_ = dir;
        backend_ = std::make_unique<gs_trad::DiskBackend>(out_dir_);
        backend_->Setup(1, kBenchParams);
        backend_->SeedInitial();
    }
    void TearDown(benchmark::State&) override {
        if (backend_) {
            backend_->Teardown();
            backend_.reset();
        }
        if (!out_dir_.empty()) {
            std::string cmd = "rm -rf '" + out_dir_ + "'";
            (void)std::system(cmd.c_str());
            out_dir_.clear();
        }
    }
protected:
    std::unique_ptr<gs_trad::DiskBackend> backend_;
    std::string out_dir_;
};

BENCHMARK_DEFINE_F(GsDiskFullFixture, BM_GrayScott_Full_TraditionalDisk)
(benchmark::State& state) {
    for (auto _ : state) {
        for (int step = 1; step <= kFullSteps; ++step) {
            backend_->StepAll();
            if (step % kFullSnapInterval == 0) {
                backend_->Snapshot(step);
            }
        }
    }
    int n_snaps = kFullSteps / kFullSnapInterval;
    size_t bytes_per_iter =
        (static_cast<size_t>(kFullSteps) * 2 +
         static_cast<size_t>(n_snaps) * 2) *
        gs_trad::kBytesPerGrid;
    state.SetBytesProcessed(state.iterations() *
                            static_cast<int64_t>(bytes_per_iter));
    state.SetItemsProcessed(state.iterations() * kFullSteps);
}
BENCHMARK_REGISTER_F(GsDiskFullFixture, BM_GrayScott_Full_TraditionalDisk)
    ->Unit(benchmark::kMillisecond)
    ->MinTime(0.1);

} // namespace
