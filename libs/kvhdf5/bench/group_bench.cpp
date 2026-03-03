#include "bench_fixture.h"
#include "kvhdf5/hdf5.h"
#include "kvhdf5/cte_blob_store.h"
#include <string>

using namespace kvhdf5;

// ============================================================================
// BM_FileCreate
// ============================================================================

class FileCreateFixture : public bench::CteFixture {};

BENCHMARK_DEFINE_F(FileCreateFixture, BM_FileCreate)(benchmark::State& state) {
    for (auto _ : state) {
        ResetAllocator();
        auto file = CreateFile();
        benchmark::DoNotOptimize(file);
        state.PauseTiming();
        DestroyFile(file);
        state.ResumeTiming();
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK_REGISTER_F(FileCreateFixture, BM_FileCreate)
    ->Iterations(5);

// ============================================================================
// BM_CreateGroup / BM_OpenGroup
// ============================================================================

class GroupCreateFixture : public bench::CteFixture {};

BENCHMARK_DEFINE_F(GroupCreateFixture, BM_CreateGroup)(benchmark::State& state) {
    for (auto _ : state) {
        state.PauseTiming();
        ResetAllocator();
        auto file = CreateFile();
        auto root = file.OpenRootGroup();
        state.ResumeTiming();
        auto result = root.CreateGroup("test");
        benchmark::DoNotOptimize(result);
        state.PauseTiming();
        DestroyFile(file);
        state.ResumeTiming();
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK_REGISTER_F(GroupCreateFixture, BM_CreateGroup)
    ->Iterations(5);

class GroupOpenFixture : public bench::CteFixture {};

BENCHMARK_DEFINE_F(GroupOpenFixture, BM_OpenGroup)(benchmark::State& state) {
    auto file = CreateFile();
    auto root = file.OpenRootGroup();
    root.CreateGroup("target");

    for (auto _ : state) {
        ResetAllocator();
        auto result = root.OpenGroup("target");
        benchmark::DoNotOptimize(result);
    }
    DestroyFile(file);
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK_REGISTER_F(GroupOpenFixture, BM_OpenGroup);

// ============================================================================
// BM_CreateSiblingGroups
// ============================================================================

class SiblingGroupsFixture : public bench::CteFixture {};

BENCHMARK_DEFINE_F(SiblingGroupsFixture, BM_CreateSiblingGroups)(benchmark::State& state) {
    int64_t count = state.range(0);

    for (auto _ : state) {
        state.PauseTiming();
        ResetAllocator();
        auto file = CreateFile();
        auto root = file.OpenRootGroup();
        state.ResumeTiming();

        for (int64_t i = 0; i < count; ++i) {
            std::string name = "s_" + std::to_string(i);
            auto result = root.CreateGroup(gpu_string_view(name.c_str()));
            benchmark::DoNotOptimize(result);
        }

        state.PauseTiming();
        DestroyFile(file);
        state.ResumeTiming();
    }
    state.SetItemsProcessed(state.iterations() * count);
}
BENCHMARK_REGISTER_F(SiblingGroupsFixture, BM_CreateSiblingGroups)
    ->Arg(1)->Arg(5)->Arg(10)->Arg(20)
    ->Iterations(5);

// ============================================================================
// BM_CreateGroupDeep / BM_OpenGroupDeep
// ============================================================================

class DeepGroupFixture : public bench::CteFixture {};

BENCHMARK_DEFINE_F(DeepGroupFixture, BM_CreateGroupDeep)(benchmark::State& state) {
    int64_t depth = state.range(0);

    for (auto _ : state) {
        state.PauseTiming();
        ResetAllocator();
        auto file = CreateFile();
        auto root = file.OpenRootGroup();
        state.ResumeTiming();

        auto current = root;
        for (int64_t d = 0; d < depth; ++d) {
            std::string name = "d_" + std::to_string(d);
            auto result = current.CreateGroup(gpu_string_view(name.c_str()));
            current = result.value();
        }
        benchmark::DoNotOptimize(current);

        state.PauseTiming();
        DestroyFile(file);
        state.ResumeTiming();
    }
    state.SetItemsProcessed(state.iterations() * depth);
}
BENCHMARK_REGISTER_F(DeepGroupFixture, BM_CreateGroupDeep)
    ->Arg(1)->Arg(3)->Arg(5)->Arg(10)
    ->Iterations(5);

class OpenDeepGroupFixture : public bench::CteFixture {};

BENCHMARK_DEFINE_F(OpenDeepGroupFixture, BM_OpenGroupDeep)(benchmark::State& state) {
    int64_t depth = state.range(0);

    for (auto _ : state) {
        state.PauseTiming();
        ResetAllocator();
        auto file = CreateFile();
        auto root = file.OpenRootGroup();
        auto current = root;
        for (int64_t d = 0; d < depth; ++d) {
            std::string name = "d_" + std::to_string(d);
            current = current.CreateGroup(gpu_string_view(name.c_str())).value();
        }
        state.ResumeTiming();

        auto g = root;
        for (int64_t d = 0; d < depth; ++d) {
            std::string name = "d_" + std::to_string(d);
            g = g.OpenGroup(gpu_string_view(name.c_str())).value();
        }
        benchmark::DoNotOptimize(g);

        state.PauseTiming();
        DestroyFile(file);
        state.ResumeTiming();
    }
    state.SetItemsProcessed(state.iterations() * depth);
    state.counters["opens/sec"] = benchmark::Counter(
        state.iterations() * depth, benchmark::Counter::kIsRate);
}
BENCHMARK_REGISTER_F(OpenDeepGroupFixture, BM_OpenGroupDeep)
    ->Arg(1)->Arg(3)->Arg(5)->Arg(10)
    ->Iterations(5);

// ============================================================================
// BM_CreateDataset / BM_OpenDataset
// ============================================================================

class DatasetCreateFixture : public bench::CteFixture {};

BENCHMARK_DEFINE_F(DatasetCreateFixture, BM_CreateDataset)(benchmark::State& state) {
    uint64_t dim = 100;
    auto space = Dataspace::CreateSimple(cstd::span<const uint64_t>(&dim, 1)).value();

    for (auto _ : state) {
        state.PauseTiming();
        ResetAllocator();
        auto file = CreateFile();
        auto root = file.OpenRootGroup();
        state.ResumeTiming();
        auto result = root.CreateDataset("ds", Datatype::Float64(), space);
        benchmark::DoNotOptimize(result);
        state.PauseTiming();
        DestroyFile(file);
        state.ResumeTiming();
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK_REGISTER_F(DatasetCreateFixture, BM_CreateDataset)
    ->Iterations(5);

class DatasetOpenFixture : public bench::CteFixture {};

BENCHMARK_DEFINE_F(DatasetOpenFixture, BM_OpenDataset)(benchmark::State& state) {
    auto file = CreateFile();
    auto root = file.OpenRootGroup();

    uint64_t dim = 100;
    auto space = Dataspace::CreateSimple(cstd::span<const uint64_t>(&dim, 1)).value();
    root.CreateDataset("target_ds", Datatype::Float64(), space);

    for (auto _ : state) {
        ResetAllocator();
        auto result = root.OpenDataset("target_ds");
        benchmark::DoNotOptimize(result);
    }
    DestroyFile(file);
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK_REGISTER_F(DatasetOpenFixture, BM_OpenDataset);

// ============================================================================
// BM_GetGroupInfo
// ============================================================================

class GroupInfoFixture : public bench::CteFixture {};

BENCHMARK_DEFINE_F(GroupInfoFixture, BM_GetGroupInfo)(benchmark::State& state) {
    int64_t num_children = state.range(0);

    auto file = CreateFile();
    auto root = file.OpenRootGroup();
    for (int64_t i = 0; i < num_children; ++i) {
        std::string name = "c_" + std::to_string(i);
        root.CreateGroup(gpu_string_view(name.c_str()));
    }

    for (auto _ : state) {
        ResetAllocator();
        auto info = root.GetInfo();
        benchmark::DoNotOptimize(info);
    }
    DestroyFile(file);
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK_REGISTER_F(GroupInfoFixture, BM_GetGroupInfo)
    ->Arg(0)->Arg(5)->Arg(10);
