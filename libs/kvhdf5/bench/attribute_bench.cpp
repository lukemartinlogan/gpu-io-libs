#include "bench_fixture.h"
#include "kvhdf5/hdf5.h"
#include "kvhdf5/cpu_cte_blob_store.h"
#include <string>
#include <vector>

using namespace kvhdf5;

// ============================================================================
// BM_GroupSetAttribute / BM_GroupGetAttribute
// ============================================================================

class GroupAttrFixture : public bench::CteFixture {};

BENCHMARK_DEFINE_F(GroupAttrFixture, BM_GroupSetAttribute)(benchmark::State& state) {
    int64_t value_bytes = state.range(0);
    auto file = CreateFile();
    auto root = file.OpenRootGroup();

    // Build a datatype matching the value size
    // Use Int8 with compound to get exact byte count
    Datatype type = Datatype::Int8();
    if (value_bytes == 4) type = Datatype::Int32();
    else if (value_bytes == 8) type = Datatype::Int64();
    else type = Datatype::CreateCompound(value_bytes);

    std::vector<byte_t> data(value_bytes, byte_t{42});

    for (auto _ : state) {
        state.PauseTiming();
        ResetAllocator();
        state.ResumeTiming();
        root.SetAttribute("attr", type, data.data());
    }
    DestroyFile(file);
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK_REGISTER_F(GroupAttrFixture, BM_GroupSetAttribute)
    ->Arg(4)->Arg(8)->Arg(64)->Arg(128);

BENCHMARK_DEFINE_F(GroupAttrFixture, BM_GroupGetAttribute)(benchmark::State& state) {
    int64_t value_bytes = state.range(0);
    auto file = CreateFile();
    auto root = file.OpenRootGroup();

    Datatype type = Datatype::Int8();
    if (value_bytes == 4) type = Datatype::Int32();
    else if (value_bytes == 8) type = Datatype::Int64();
    else type = Datatype::CreateCompound(value_bytes);

    std::vector<byte_t> data(value_bytes, byte_t{42});
    root.SetAttribute("attr", type, data.data());

    std::vector<byte_t> out(value_bytes);

    for (auto _ : state) {
        state.PauseTiming();
        ResetAllocator();
        state.ResumeTiming();
        root.GetAttribute("attr", type, out.data());
        benchmark::DoNotOptimize(out.data());
    }
    DestroyFile(file);
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK_REGISTER_F(GroupAttrFixture, BM_GroupGetAttribute)
    ->Arg(4)->Arg(8)->Arg(64)->Arg(128);

// ============================================================================
// BM_DatasetSetAttribute / BM_DatasetGetAttribute
// ============================================================================

class DatasetAttrFixture : public bench::CteFixture {};

BENCHMARK_DEFINE_F(DatasetAttrFixture, BM_DatasetSetAttribute)(benchmark::State& state) {
    auto file = CreateFile();
    auto root = file.OpenRootGroup();

    uint64_t dim = 100;
    auto space = Dataspace::CreateSimple(cstd::span<const uint64_t>(&dim, 1)).value();
    auto ds = root.CreateDataset("ds", Datatype::Float64(), space).value();

    int32_t value = 42;
    auto type = Datatype::Int32();

    for (auto _ : state) {
        state.PauseTiming();
        ResetAllocator();
        state.ResumeTiming();
        ds.SetAttribute("attr", type, &value);
    }
    DestroyFile(file);
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK_REGISTER_F(DatasetAttrFixture, BM_DatasetSetAttribute);

BENCHMARK_DEFINE_F(DatasetAttrFixture, BM_DatasetGetAttribute)(benchmark::State& state) {
    auto file = CreateFile();
    auto root = file.OpenRootGroup();

    uint64_t dim = 100;
    auto space = Dataspace::CreateSimple(cstd::span<const uint64_t>(&dim, 1)).value();
    auto ds = root.CreateDataset("ds", Datatype::Float64(), space).value();

    int32_t value = 42;
    auto type = Datatype::Int32();
    ds.SetAttribute("attr", type, &value);

    int32_t out = 0;
    for (auto _ : state) {
        state.PauseTiming();
        ResetAllocator();
        state.ResumeTiming();
        ds.GetAttribute("attr", type, &out);
        benchmark::DoNotOptimize(out);
    }
    DestroyFile(file);
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK_REGISTER_F(DatasetAttrFixture, BM_DatasetGetAttribute);

// ============================================================================
// BM_SetManyAttributes / BM_GetNthAttribute
// ============================================================================

class ManyAttrsFixture : public bench::CteFixture {};

BENCHMARK_DEFINE_F(ManyAttrsFixture, BM_SetManyAttributes)(benchmark::State& state) {
    int64_t count = state.range(0);

    for (auto _ : state) {
        state.PauseTiming();
        ResetAllocator();
        auto file = CreateFile();
        auto root = file.OpenRootGroup();
        auto type = Datatype::Int32();
        state.ResumeTiming();

        for (int64_t i = 0; i < count; ++i) {
            std::string name = "a_" + std::to_string(i);
            int32_t val = static_cast<int32_t>(i);
            root.SetAttribute(gpu_string_view(name.c_str()), type, &val);
        }

        state.PauseTiming();
        DestroyFile(file);
        state.ResumeTiming();
    }
    state.SetItemsProcessed(state.iterations() * count);
}
BENCHMARK_REGISTER_F(ManyAttrsFixture, BM_SetManyAttributes)
    ->Arg(1)->Arg(5)->Arg(10)->Arg(20)
    ->Iterations(5);

BENCHMARK_DEFINE_F(ManyAttrsFixture, BM_GetNthAttribute)(benchmark::State& state) {
    int64_t count = state.range(0);

    auto file = CreateFile();
    auto root = file.OpenRootGroup();
    auto type = Datatype::Int32();

    for (int64_t i = 0; i < count; ++i) {
        std::string name = "a_" + std::to_string(i);
        int32_t val = static_cast<int32_t>(i);
        root.SetAttribute(gpu_string_view(name.c_str()), type, &val);
    }

    // Get the last attribute (worst case linear scan)
    std::string last_name = "a_" + std::to_string(count - 1);
    int32_t out = 0;

    for (auto _ : state) {
        state.PauseTiming();
        ResetAllocator();
        state.ResumeTiming();
        root.GetAttribute(
            gpu_string_view(last_name.c_str()), type, &out);
        benchmark::DoNotOptimize(out);
    }
    DestroyFile(file);
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK_REGISTER_F(ManyAttrsFixture, BM_GetNthAttribute)
    ->Arg(1)->Arg(5)->Arg(10)->Arg(20);

// ============================================================================
// BM_AttributeHandleWrite / BM_AttributeHandleRead
// ============================================================================

class AttrHandleFixture : public bench::CteFixture {};

BENCHMARK_DEFINE_F(AttrHandleFixture, BM_AttributeHandleWrite)(benchmark::State& state) {
    auto file = CreateFile();
    auto root = file.OpenRootGroup();

    // Create initial attribute so the handle has something to update
    int32_t val = 0;
    auto type = Datatype::Int32();
    root.SetAttribute("handle_attr", type, &val);

    auto handle = root.OpenAttribute("handle_attr");

    for (auto _ : state) {
        state.PauseTiming();
        ResetAllocator();
        state.ResumeTiming();
        val++;
        handle.Write(type, &val);
    }
    DestroyFile(file);
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK_REGISTER_F(AttrHandleFixture, BM_AttributeHandleWrite);

BENCHMARK_DEFINE_F(AttrHandleFixture, BM_AttributeHandleRead)(benchmark::State& state) {
    auto file = CreateFile();
    auto root = file.OpenRootGroup();

    int32_t val = 42;
    auto type = Datatype::Int32();
    root.SetAttribute("handle_attr", type, &val);

    auto handle = root.OpenAttribute("handle_attr");
    int32_t out = 0;

    for (auto _ : state) {
        state.PauseTiming();
        ResetAllocator();
        state.ResumeTiming();
        handle.Read(type, &out);
        benchmark::DoNotOptimize(out);
    }
    DestroyFile(file);
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK_REGISTER_F(AttrHandleFixture, BM_AttributeHandleRead);
