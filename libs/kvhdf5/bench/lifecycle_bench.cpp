#include "bench_fixture.h"
#include "kvhdf5/hdf5.h"
#include "kvhdf5/cte_blob_store.h"
#include <string>
#include <vector>

using namespace kvhdf5;

// ============================================================================
// BM_FullLifecycle — create file, group, dataset, write, read, verify
// ============================================================================

class LifecycleFixture : public bench::CteFixture {};

BENCHMARK_DEFINE_F(LifecycleFixture, BM_FullLifecycle)(benchmark::State& state) {
    int64_t elements = state.range(0);
    auto type = Datatype::Float64();

    std::vector<double> write_buf(elements);
    for (int64_t i = 0; i < elements; ++i) write_buf[i] = static_cast<double>(i);
    std::vector<double> read_buf(elements);

    for (auto _ : state) {
        ResetAllocator();
        // Create file + group + dataset
        auto file = CreateFile();
        auto root = file.OpenRootGroup();
        auto grp = root.CreateGroup("data").value();

        uint64_t dim = elements;
        auto space = Dataspace::CreateSimple(cstd::span<const uint64_t>(&dim, 1)).value();
        auto ds = grp.CreateDataset("values", type, space).value();

        // Set attribute
        int32_t version = 1;
        ds.SetAttribute("version", Datatype::Int32(), &version);

        // Write
        auto mem_space = Dataspace::CreateSimple(cstd::span<const uint64_t>(&dim, 1)).value();
        auto file_space = Dataspace::CreateSimple(cstd::span<const uint64_t>(&dim, 1)).value();
        ds.Write(type, mem_space, file_space, write_buf.data());

        // Read back
        ds.Read(type, mem_space, file_space, read_buf.data());
        benchmark::DoNotOptimize(read_buf.data());
        state.PauseTiming();
        DestroyFile(file);
        state.ResumeTiming();
    }
    state.SetBytesProcessed(state.iterations() * elements * sizeof(double) * 2);
}
BENCHMARK_REGISTER_F(LifecycleFixture, BM_FullLifecycle)
    ->Arg(100)->Arg(1000)
    ->Iterations(5);

// ============================================================================
// BM_LargeSequentialWriteRead
// ============================================================================

class LargeSeqFixture : public bench::CteFixture {};

BENCHMARK_DEFINE_F(LargeSeqFixture, BM_LargeSequentialWriteRead)(benchmark::State& state) {
    int64_t total = state.range(0);
    int64_t chunk = state.range(1);
    auto type = Datatype::Float64();

    std::vector<double> data(total);
    for (int64_t i = 0; i < total; ++i) data[i] = static_cast<double>(i);
    std::vector<double> out(total);

    for (auto _ : state) {
        ResetAllocator();
        auto file = CreateFile();
        auto root = file.OpenRootGroup();

        uint64_t dim = total;
        auto space = Dataspace::CreateSimple(cstd::span<const uint64_t>(&dim, 1)).value();
        DatasetCreateProps props;
        props.chunk_dims.push_back(chunk);
        auto ds = root.CreateDataset("ds", type, space, props).value();

        auto mem_space = Dataspace::CreateSimple(cstd::span<const uint64_t>(&dim, 1)).value();
        auto file_space = Dataspace::CreateSimple(cstd::span<const uint64_t>(&dim, 1)).value();

        ds.Write(type, mem_space, file_space, data.data());
        ds.Read(type, mem_space, file_space, out.data());
        benchmark::DoNotOptimize(out.data());
        state.PauseTiming();
        DestroyFile(file);
        state.ResumeTiming();
    }
    int64_t num_chunks = (total + chunk - 1) / chunk;
    state.SetBytesProcessed(state.iterations() * total * sizeof(double) * 2);
    state.counters["chunks/sec"] = benchmark::Counter(
        state.iterations() * num_chunks * 2, benchmark::Counter::kIsRate);
}
BENCHMARK_REGISTER_F(LargeSeqFixture, BM_LargeSequentialWriteRead)
    ->Args({10000, 100})
    ->Iterations(5);

// ============================================================================
// BM_Strided2DHyperslab
// ============================================================================

class Strided2DFixture : public bench::CteFixture {};

BENCHMARK_DEFINE_F(Strided2DFixture, BM_Strided2DHyperslab)(benchmark::State& state) {
    uint64_t rows = state.range(0);
    uint64_t cols = state.range(1);
    auto type = Datatype::Float64();

    auto file = CreateFile();
    auto root = file.OpenRootGroup();

    uint64_t dims[2] = {rows, cols};
    auto space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 2)).value();
    DatasetCreateProps props;
    props.chunk_dims.push_back(10);
    props.chunk_dims.push_back(10);
    auto ds = root.CreateDataset("ds2d", type, space, props).value();

    // Write full dataset
    auto full_space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 2)).value();
    std::vector<double> data(rows * cols);
    for (uint64_t i = 0; i < rows * cols; ++i) data[i] = static_cast<double>(i);
    ds.Write(type, full_space, full_space, data.data());

    // Strided 2D read: every other row, every other column
    auto file_space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 2)).value();
    uint64_t start[2] = {0, 0};
    uint64_t stride_v[2] = {2, 2};
    uint64_t count[2] = {rows / 2, cols / 2};
    uint64_t block[2] = {1, 1};
    file_space.SelectHyperslab(SelectionOp::Set,
        cstd::span<const uint64_t>(start, 2),
        cstd::span<const uint64_t>(stride_v, 2),
        cstd::span<const uint64_t>(count, 2),
        cstd::span<const uint64_t>(block, 2));

    uint64_t mem_dims[2] = {rows / 2, cols / 2};
    auto mem_space = Dataspace::CreateSimple(cstd::span<const uint64_t>(mem_dims, 2)).value();
    uint64_t selected = (rows / 2) * (cols / 2);

    std::vector<double> out(selected);
    for (auto _ : state) {
        ResetAllocator();
        ds.Read(type, mem_space, file_space, out.data());
        benchmark::DoNotOptimize(out.data());
    }
    DestroyFile(file);
    state.SetBytesProcessed(state.iterations() * selected * sizeof(double));
}
BENCHMARK_REGISTER_F(Strided2DFixture, BM_Strided2DHyperslab)
    ->Args({100, 100});

// ============================================================================
// BM_DeepHierarchyTraversal
// ============================================================================

class DeepTraversalFixture : public bench::CteFixture {};

BENCHMARK_DEFINE_F(DeepTraversalFixture, BM_DeepHierarchyTraversal)(benchmark::State& state) {
    int64_t depth = state.range(0);

    auto file = CreateFile();
    auto root = file.OpenRootGroup();

    // Pre-create deep hierarchy
    auto current = root;
    for (int64_t d = 0; d < depth; ++d) {
        std::string name = "level_" + std::to_string(d);
        current = current.CreateGroup(gpu_string_view(name.c_str())).value();
    }

    for (auto _ : state) {
        ResetAllocator();
        auto g = root;
        for (int64_t d = 0; d < depth; ++d) {
            std::string name = "level_" + std::to_string(d);
            g = g.OpenGroup(gpu_string_view(name.c_str())).value();
        }
        benchmark::DoNotOptimize(g);
    }
    DestroyFile(file);
    state.counters["groups/sec"] = benchmark::Counter(
        state.iterations() * depth, benchmark::Counter::kIsRate);
}
BENCHMARK_REGISTER_F(DeepTraversalFixture, BM_DeepHierarchyTraversal)
    ->Arg(5)->Arg(10)->Arg(20);

// ============================================================================
// BM_ManySmallAttributes
// ============================================================================

class ManySmallAttrsFixture : public bench::CteFixture {};

BENCHMARK_DEFINE_F(ManySmallAttrsFixture, BM_ManySmallAttributes)(benchmark::State& state) {
    int64_t count = state.range(0);
    auto type = Datatype::Int32();

    for (auto _ : state) {
        state.PauseTiming();
        ResetAllocator();
        auto file = CreateFile();
        auto root = file.OpenRootGroup();
        state.ResumeTiming();

        for (int64_t i = 0; i < count; ++i) {
            std::string name = "a_" + std::to_string(i);
            int32_t val = static_cast<int32_t>(i);
            root.SetAttribute(gpu_string_view(name.c_str()), type, &val);
        }

        // Read all back
        for (int64_t i = 0; i < count; ++i) {
            std::string name = "a_" + std::to_string(i);
            int32_t out = 0;
            root.GetAttribute(gpu_string_view(name.c_str()), type, &out);
            benchmark::DoNotOptimize(out);
        }
        state.PauseTiming();
        DestroyFile(file);
        state.ResumeTiming();
    }
    state.counters["attrs/sec"] = benchmark::Counter(
        state.iterations() * count * 2, benchmark::Counter::kIsRate);
}
BENCHMARK_REGISTER_F(ManySmallAttrsFixture, BM_ManySmallAttributes)
    ->Arg(10)->Arg(30)->Arg(50)
    ->Iterations(5);

// ============================================================================
// BM_SparseChunkAccess
// Args: {total_chunks, written_percent}
// ============================================================================

class SparseChunkFixture : public bench::CteFixture {};

BENCHMARK_DEFINE_F(SparseChunkFixture, BM_SparseChunkAccess)(benchmark::State& state) {
    int64_t total_chunks = state.range(0);
    int64_t written_pct = state.range(1);
    int64_t chunk_size = 10;
    int64_t total_elements = total_chunks * chunk_size;
    int64_t written_chunks = total_chunks * written_pct / 100;
    auto type = Datatype::Float64();

    auto file = CreateFile();
    auto root = file.OpenRootGroup();

    uint64_t dim = total_elements;
    auto space = Dataspace::CreateSimple(cstd::span<const uint64_t>(&dim, 1)).value();
    DatasetCreateProps props;
    props.chunk_dims.push_back(chunk_size);
    auto ds = root.CreateDataset("ds", type, space, props).value();

    // Write only some chunks (every N-th chunk to achieve target percentage)
    std::vector<double> chunk_data(chunk_size, 1.0);
    int64_t step = (written_chunks > 0) ? total_chunks / written_chunks : total_chunks;
    for (int64_t c = 0; c < total_chunks && c * step < total_chunks; ++c) {
        int64_t chunk_idx = c * step;
        if (chunk_idx >= total_chunks) break;

        // Write chunk at chunk_idx via hyperslab
        uint64_t start = chunk_idx * chunk_size;
        uint64_t stride_v = 1;
        uint64_t count_v = chunk_size;
        uint64_t block_v = 1;

        auto file_space = Dataspace::CreateSimple(cstd::span<const uint64_t>(&dim, 1)).value();
        file_space.SelectHyperslab(SelectionOp::Set,
            cstd::span<const uint64_t>(&start, 1),
            cstd::span<const uint64_t>(&stride_v, 1),
            cstd::span<const uint64_t>(&count_v, 1),
            cstd::span<const uint64_t>(&block_v, 1));

        uint64_t mem_dim = chunk_size;
        auto mem_space = Dataspace::CreateSimple(cstd::span<const uint64_t>(&mem_dim, 1)).value();
        ds.Write(type, mem_space, file_space, chunk_data.data());
    }

    // Read full dataset (including sparse/empty chunks)
    auto full_space = Dataspace::CreateSimple(cstd::span<const uint64_t>(&dim, 1)).value();
    std::vector<double> out(total_elements);

    for (auto _ : state) {
        ResetAllocator();
        ds.Read(type, full_space, full_space, out.data());
        benchmark::DoNotOptimize(out.data());
    }
    DestroyFile(file);
    state.SetBytesProcessed(state.iterations() * total_elements * sizeof(double));
}
BENCHMARK_REGISTER_F(SparseChunkFixture, BM_SparseChunkAccess)
    ->Args({100, 10})
    ->Args({100, 50});
