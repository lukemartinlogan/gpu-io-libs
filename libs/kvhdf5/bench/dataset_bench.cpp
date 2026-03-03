#include "bench_fixture.h"
#include "kvhdf5/hdf5.h"
#include "kvhdf5/cte_blob_store.h"
#include <string>
#include <vector>

using namespace kvhdf5;

// ============================================================================
// Helper: create a 1D dataset with specified total elements and chunk size
// ============================================================================

static Dataset<CteBlobStore> Make1DDataset(
    Group<CteBlobStore>& root, const char* name,
    uint64_t total_elements, uint64_t chunk_size
) {
    uint64_t dim = total_elements;
    auto space = Dataspace::CreateSimple(cstd::span<const uint64_t>(&dim, 1)).value();
    DatasetCreateProps props;
    props.chunk_dims.push_back(chunk_size);
    return root.CreateDataset(
        gpu_string_view(name), Datatype::Float64(), space, props).value();
}

// ============================================================================
// BM_WriteSingleChunk / BM_ReadSingleChunk
// ============================================================================

class SingleChunkFixture : public bench::CteFixture {};

BENCHMARK_DEFINE_F(SingleChunkFixture, BM_WriteSingleChunk)(benchmark::State& state) {
    int64_t elements = state.range(0);
    auto file = CreateFile();
    auto root = file.OpenRootGroup();

    auto ds = Make1DDataset(root, "ds", elements, elements);
    auto type = Datatype::Float64();

    uint64_t dim = elements;
    auto mem_space = Dataspace::CreateSimple(cstd::span<const uint64_t>(&dim, 1)).value();
    auto file_space = Dataspace::CreateSimple(cstd::span<const uint64_t>(&dim, 1)).value();

    std::vector<double> data(elements);
    for (int64_t i = 0; i < elements; ++i) data[i] = static_cast<double>(i);

    for (auto _ : state) {
        ResetAllocator();
        ds.Write(type, mem_space, file_space, data.data());
    }
    DestroyFile(file);
    state.SetBytesProcessed(state.iterations() * elements * sizeof(double));
    state.counters["elements/sec"] = benchmark::Counter(
        state.iterations() * elements, benchmark::Counter::kIsRate);
}
BENCHMARK_REGISTER_F(SingleChunkFixture, BM_WriteSingleChunk)
    ->Arg(100)->Arg(1000)->Arg(4000)->Arg(8000);

BENCHMARK_DEFINE_F(SingleChunkFixture, BM_ReadSingleChunk)(benchmark::State& state) {
    int64_t elements = state.range(0);
    auto file = CreateFile();
    auto root = file.OpenRootGroup();

    auto ds = Make1DDataset(root, "ds", elements, elements);
    auto type = Datatype::Float64();

    uint64_t dim = elements;
    auto mem_space = Dataspace::CreateSimple(cstd::span<const uint64_t>(&dim, 1)).value();
    auto file_space = Dataspace::CreateSimple(cstd::span<const uint64_t>(&dim, 1)).value();

    std::vector<double> data(elements);
    for (int64_t i = 0; i < elements; ++i) data[i] = static_cast<double>(i);
    ds.Write(type, mem_space, file_space, data.data());

    std::vector<double> out(elements);

    for (auto _ : state) {
        ResetAllocator();
        ds.Read(type, mem_space, file_space, out.data());
        benchmark::DoNotOptimize(out.data());
    }
    DestroyFile(file);
    state.SetBytesProcessed(state.iterations() * elements * sizeof(double));
    state.counters["elements/sec"] = benchmark::Counter(
        state.iterations() * elements, benchmark::Counter::kIsRate);
}
BENCHMARK_REGISTER_F(SingleChunkFixture, BM_ReadSingleChunk)
    ->Arg(100)->Arg(1000)->Arg(4000)->Arg(8000);

// ============================================================================
// BM_WriteMultiChunk / BM_ReadMultiChunk
// Uses Args({total, chunk})
// ============================================================================

class MultiChunkFixture : public bench::CteFixture {};

BENCHMARK_DEFINE_F(MultiChunkFixture, BM_WriteMultiChunk)(benchmark::State& state) {
    int64_t total = state.range(0);
    int64_t chunk = state.range(1);

    auto file = CreateFile();
    auto root = file.OpenRootGroup();
    auto ds = Make1DDataset(root, "ds", total, chunk);
    auto type = Datatype::Float64();

    uint64_t dim = total;
    auto mem_space = Dataspace::CreateSimple(cstd::span<const uint64_t>(&dim, 1)).value();
    auto file_space = Dataspace::CreateSimple(cstd::span<const uint64_t>(&dim, 1)).value();

    std::vector<double> data(total);
    for (int64_t i = 0; i < total; ++i) data[i] = static_cast<double>(i);

    for (auto _ : state) {
        ResetAllocator();
        ds.Write(type, mem_space, file_space, data.data());
    }
    DestroyFile(file);
    state.SetBytesProcessed(state.iterations() * total * sizeof(double));
    int64_t num_chunks = (total + chunk - 1) / chunk;
    state.counters["chunks/sec"] = benchmark::Counter(
        state.iterations() * num_chunks, benchmark::Counter::kIsRate);
}
BENCHMARK_REGISTER_F(MultiChunkFixture, BM_WriteMultiChunk)
    ->Args({1000, 100})
    ->Args({10000, 100})
    ->Args({10000, 1000});

BENCHMARK_DEFINE_F(MultiChunkFixture, BM_ReadMultiChunk)(benchmark::State& state) {
    int64_t total = state.range(0);
    int64_t chunk = state.range(1);

    auto file = CreateFile();
    auto root = file.OpenRootGroup();
    auto ds = Make1DDataset(root, "ds", total, chunk);
    auto type = Datatype::Float64();

    uint64_t dim = total;
    auto mem_space = Dataspace::CreateSimple(cstd::span<const uint64_t>(&dim, 1)).value();
    auto file_space = Dataspace::CreateSimple(cstd::span<const uint64_t>(&dim, 1)).value();

    std::vector<double> data(total);
    for (int64_t i = 0; i < total; ++i) data[i] = static_cast<double>(i);
    ds.Write(type, mem_space, file_space, data.data());

    std::vector<double> out(total);

    for (auto _ : state) {
        ResetAllocator();
        ds.Read(type, mem_space, file_space, out.data());
        benchmark::DoNotOptimize(out.data());
    }
    DestroyFile(file);
    state.SetBytesProcessed(state.iterations() * total * sizeof(double));
    int64_t num_chunks = (total + chunk - 1) / chunk;
    state.counters["chunks/sec"] = benchmark::Counter(
        state.iterations() * num_chunks, benchmark::Counter::kIsRate);
}
BENCHMARK_REGISTER_F(MultiChunkFixture, BM_ReadMultiChunk)
    ->Args({1000, 100})
    ->Args({10000, 100})
    ->Args({10000, 1000});

// ============================================================================
// BM_Write2D / BM_Read2D
// Args: {rows, cols, chunk_rows, chunk_cols}
// ============================================================================

class TwoDFixture : public bench::CteFixture {};

BENCHMARK_DEFINE_F(TwoDFixture, BM_Write2D)(benchmark::State& state) {
    uint64_t rows = state.range(0);
    uint64_t cols = state.range(1);
    uint64_t cr = state.range(2);
    uint64_t cc = state.range(3);

    auto file = CreateFile();
    auto root = file.OpenRootGroup();

    uint64_t dims[2] = {rows, cols};
    auto space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 2)).value();
    DatasetCreateProps props;
    props.chunk_dims.push_back(cr);
    props.chunk_dims.push_back(cc);
    auto ds = root.CreateDataset("ds2d", Datatype::Float64(), space, props).value();
    auto type = Datatype::Float64();

    auto mem_space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 2)).value();
    auto file_space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 2)).value();

    std::vector<double> data(rows * cols);
    for (uint64_t i = 0; i < rows * cols; ++i) data[i] = static_cast<double>(i);

    for (auto _ : state) {
        ResetAllocator();
        ds.Write(type, mem_space, file_space, data.data());
    }
    DestroyFile(file);
    state.SetBytesProcessed(state.iterations() * rows * cols * sizeof(double));
}
BENCHMARK_REGISTER_F(TwoDFixture, BM_Write2D)
    ->Args({32, 32, 8, 8})
    ->Args({100, 100, 10, 10});

BENCHMARK_DEFINE_F(TwoDFixture, BM_Read2D)(benchmark::State& state) {
    uint64_t rows = state.range(0);
    uint64_t cols = state.range(1);
    uint64_t cr = state.range(2);
    uint64_t cc = state.range(3);

    auto file = CreateFile();
    auto root = file.OpenRootGroup();

    uint64_t dims[2] = {rows, cols};
    auto space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 2)).value();
    DatasetCreateProps props;
    props.chunk_dims.push_back(cr);
    props.chunk_dims.push_back(cc);
    auto ds = root.CreateDataset("ds2d", Datatype::Float64(), space, props).value();
    auto type = Datatype::Float64();

    auto mem_space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 2)).value();
    auto file_space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 2)).value();

    std::vector<double> data(rows * cols);
    for (uint64_t i = 0; i < rows * cols; ++i) data[i] = static_cast<double>(i);
    ds.Write(type, mem_space, file_space, data.data());

    std::vector<double> out(rows * cols);
    for (auto _ : state) {
        ResetAllocator();
        ds.Read(type, mem_space, file_space, out.data());
        benchmark::DoNotOptimize(out.data());
    }
    DestroyFile(file);
    state.SetBytesProcessed(state.iterations() * rows * cols * sizeof(double));
}
BENCHMARK_REGISTER_F(TwoDFixture, BM_Read2D)
    ->Args({32, 32, 8, 8})
    ->Args({100, 100, 10, 10});

// ============================================================================
// BM_WriteHyperslabContiguous / BM_ReadHyperslabContiguous
// Args: {total, slab_count}
// ============================================================================

class HyperslabContFixture : public bench::CteFixture {};

BENCHMARK_DEFINE_F(HyperslabContFixture, BM_WriteHyperslabContiguous)(benchmark::State& state) {
    int64_t total = state.range(0);
    int64_t slab_count = state.range(1);

    auto file = CreateFile();
    auto root = file.OpenRootGroup();
    auto ds = Make1DDataset(root, "ds", total, total);
    auto type = Datatype::Float64();

    // Contiguous hyperslab: start=0, stride=1, count=slab_count, block=1
    uint64_t dim = total;
    auto file_space = Dataspace::CreateSimple(cstd::span<const uint64_t>(&dim, 1)).value();
    uint64_t start = 0, stride = 1, count = slab_count, block = 1;
    file_space.SelectHyperslab(SelectionOp::Set,
        cstd::span<const uint64_t>(&start, 1),
        cstd::span<const uint64_t>(&stride, 1),
        cstd::span<const uint64_t>(&count, 1),
        cstd::span<const uint64_t>(&block, 1));

    uint64_t mem_dim = slab_count;
    auto mem_space = Dataspace::CreateSimple(cstd::span<const uint64_t>(&mem_dim, 1)).value();

    std::vector<double> data(slab_count);
    for (int64_t i = 0; i < slab_count; ++i) data[i] = static_cast<double>(i);

    for (auto _ : state) {
        ResetAllocator();
        ds.Write(type, mem_space, file_space, data.data());
    }
    DestroyFile(file);
    state.SetBytesProcessed(state.iterations() * slab_count * sizeof(double));
}
BENCHMARK_REGISTER_F(HyperslabContFixture, BM_WriteHyperslabContiguous)
    ->Args({1000, 100})
    ->Args({1000, 500});

BENCHMARK_DEFINE_F(HyperslabContFixture, BM_ReadHyperslabContiguous)(benchmark::State& state) {
    int64_t total = state.range(0);
    int64_t slab_count = state.range(1);

    auto file = CreateFile();
    auto root = file.OpenRootGroup();
    auto ds = Make1DDataset(root, "ds", total, total);
    auto type = Datatype::Float64();

    // Write full dataset first
    uint64_t dim = total;
    auto full_space = Dataspace::CreateSimple(cstd::span<const uint64_t>(&dim, 1)).value();
    std::vector<double> full_data(total);
    for (int64_t i = 0; i < total; ++i) full_data[i] = static_cast<double>(i);
    ds.Write(type, full_space, full_space, full_data.data());

    // Contiguous hyperslab for reading
    auto file_space = Dataspace::CreateSimple(cstd::span<const uint64_t>(&dim, 1)).value();
    uint64_t start = 0, stride = 1, count = slab_count, block = 1;
    file_space.SelectHyperslab(SelectionOp::Set,
        cstd::span<const uint64_t>(&start, 1),
        cstd::span<const uint64_t>(&stride, 1),
        cstd::span<const uint64_t>(&count, 1),
        cstd::span<const uint64_t>(&block, 1));

    uint64_t mem_dim = slab_count;
    auto mem_space = Dataspace::CreateSimple(cstd::span<const uint64_t>(&mem_dim, 1)).value();

    std::vector<double> out(slab_count);
    for (auto _ : state) {
        ResetAllocator();
        ds.Read(type, mem_space, file_space, out.data());
        benchmark::DoNotOptimize(out.data());
    }
    DestroyFile(file);
    state.SetBytesProcessed(state.iterations() * slab_count * sizeof(double));
}
BENCHMARK_REGISTER_F(HyperslabContFixture, BM_ReadHyperslabContiguous)
    ->Args({1000, 100})
    ->Args({1000, 500});

// ============================================================================
// BM_WriteHyperslabStrided / BM_ReadHyperslabStrided
// Args: {total, stride_val, count_val}
// ============================================================================

class HyperslabStridedFixture : public bench::CteFixture {};

BENCHMARK_DEFINE_F(HyperslabStridedFixture, BM_WriteHyperslabStrided)(benchmark::State& state) {
    int64_t total = state.range(0);
    int64_t stride_val = state.range(1);
    int64_t count_val = state.range(2);

    auto file = CreateFile();
    auto root = file.OpenRootGroup();
    auto ds = Make1DDataset(root, "ds", total, total);
    auto type = Datatype::Float64();

    uint64_t dim = total;
    auto file_space = Dataspace::CreateSimple(cstd::span<const uint64_t>(&dim, 1)).value();
    uint64_t start = 0;
    uint64_t stride_u = stride_val;
    uint64_t count_u = count_val;
    uint64_t block = 1;
    file_space.SelectHyperslab(SelectionOp::Set,
        cstd::span<const uint64_t>(&start, 1),
        cstd::span<const uint64_t>(&stride_u, 1),
        cstd::span<const uint64_t>(&count_u, 1),
        cstd::span<const uint64_t>(&block, 1));

    uint64_t mem_dim = count_val;
    auto mem_space = Dataspace::CreateSimple(cstd::span<const uint64_t>(&mem_dim, 1)).value();

    std::vector<double> data(count_val);
    for (int64_t i = 0; i < count_val; ++i) data[i] = static_cast<double>(i);

    for (auto _ : state) {
        ResetAllocator();
        ds.Write(type, mem_space, file_space, data.data());
    }
    DestroyFile(file);
    state.SetBytesProcessed(state.iterations() * count_val * sizeof(double));
}
BENCHMARK_REGISTER_F(HyperslabStridedFixture, BM_WriteHyperslabStrided)
    ->Args({1000, 3, 100})
    ->Args({1000, 10, 50});

BENCHMARK_DEFINE_F(HyperslabStridedFixture, BM_ReadHyperslabStrided)(benchmark::State& state) {
    int64_t total = state.range(0);
    int64_t stride_val = state.range(1);
    int64_t count_val = state.range(2);

    auto file = CreateFile();
    auto root = file.OpenRootGroup();
    auto ds = Make1DDataset(root, "ds", total, total);
    auto type = Datatype::Float64();

    // Write full dataset first
    uint64_t dim = total;
    auto full_space = Dataspace::CreateSimple(cstd::span<const uint64_t>(&dim, 1)).value();
    std::vector<double> full_data(total);
    for (int64_t i = 0; i < total; ++i) full_data[i] = static_cast<double>(i);
    ds.Write(type, full_space, full_space, full_data.data());

    // Strided read
    auto file_space = Dataspace::CreateSimple(cstd::span<const uint64_t>(&dim, 1)).value();
    uint64_t start = 0;
    uint64_t stride_u = stride_val;
    uint64_t count_u = count_val;
    uint64_t block = 1;
    file_space.SelectHyperslab(SelectionOp::Set,
        cstd::span<const uint64_t>(&start, 1),
        cstd::span<const uint64_t>(&stride_u, 1),
        cstd::span<const uint64_t>(&count_u, 1),
        cstd::span<const uint64_t>(&block, 1));

    uint64_t mem_dim = count_val;
    auto mem_space = Dataspace::CreateSimple(cstd::span<const uint64_t>(&mem_dim, 1)).value();

    std::vector<double> out(count_val);
    for (auto _ : state) {
        ResetAllocator();
        ds.Read(type, mem_space, file_space, out.data());
        benchmark::DoNotOptimize(out.data());
    }
    DestroyFile(file);
    state.SetBytesProcessed(state.iterations() * count_val * sizeof(double));
}
BENCHMARK_REGISTER_F(HyperslabStridedFixture, BM_ReadHyperslabStrided)
    ->Args({1000, 3, 100})
    ->Args({1000, 10, 50});

// ============================================================================
// BM_Read2DRowSlice / BM_Read2DColSlice
// Args: {rows, cols, chunk_rows, chunk_cols}
// ============================================================================

class Slice2DFixture : public bench::CteFixture {};

BENCHMARK_DEFINE_F(Slice2DFixture, BM_Read2DRowSlice)(benchmark::State& state) {
    uint64_t rows = state.range(0);
    uint64_t cols = state.range(1);
    uint64_t cr = state.range(2);
    uint64_t cc = state.range(3);

    auto file = CreateFile();
    auto root = file.OpenRootGroup();

    uint64_t dims[2] = {rows, cols};
    auto space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 2)).value();
    DatasetCreateProps props;
    props.chunk_dims.push_back(cr);
    props.chunk_dims.push_back(cc);
    auto ds = root.CreateDataset("ds2d", Datatype::Float64(), space, props).value();
    auto type = Datatype::Float64();

    // Write full dataset
    auto full_space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 2)).value();
    std::vector<double> data(rows * cols);
    for (uint64_t i = 0; i < rows * cols; ++i) data[i] = static_cast<double>(i);
    ds.Write(type, full_space, full_space, data.data());

    // Row slice: row 0, all columns
    auto file_space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 2)).value();
    uint64_t start[2] = {0, 0};
    uint64_t stride[2] = {1, 1};
    uint64_t count[2] = {1, cols};
    uint64_t block_v[2] = {1, 1};
    file_space.SelectHyperslab(SelectionOp::Set,
        cstd::span<const uint64_t>(start, 2),
        cstd::span<const uint64_t>(stride, 2),
        cstd::span<const uint64_t>(count, 2),
        cstd::span<const uint64_t>(block_v, 2));

    uint64_t mem_dims[2] = {1, cols};
    auto mem_space = Dataspace::CreateSimple(cstd::span<const uint64_t>(mem_dims, 2)).value();

    std::vector<double> out(cols);
    for (auto _ : state) {
        ResetAllocator();
        ds.Read(type, mem_space, file_space, out.data());
        benchmark::DoNotOptimize(out.data());
    }
    DestroyFile(file);
    state.SetBytesProcessed(state.iterations() * cols * sizeof(double));
}
BENCHMARK_REGISTER_F(Slice2DFixture, BM_Read2DRowSlice)
    ->Args({100, 100, 10, 10});

BENCHMARK_DEFINE_F(Slice2DFixture, BM_Read2DColSlice)(benchmark::State& state) {
    uint64_t rows = state.range(0);
    uint64_t cols = state.range(1);
    uint64_t cr = state.range(2);
    uint64_t cc = state.range(3);

    auto file = CreateFile();
    auto root = file.OpenRootGroup();

    uint64_t dims[2] = {rows, cols};
    auto space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 2)).value();
    DatasetCreateProps props;
    props.chunk_dims.push_back(cr);
    props.chunk_dims.push_back(cc);
    auto ds = root.CreateDataset("ds2d", Datatype::Float64(), space, props).value();
    auto type = Datatype::Float64();

    // Write full dataset
    auto full_space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 2)).value();
    std::vector<double> data(rows * cols);
    for (uint64_t i = 0; i < rows * cols; ++i) data[i] = static_cast<double>(i);
    ds.Write(type, full_space, full_space, data.data());

    // Column slice: all rows, col 0
    auto file_space = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 2)).value();
    uint64_t start[2] = {0, 0};
    uint64_t stride[2] = {1, 1};
    uint64_t count[2] = {rows, 1};
    uint64_t block_v[2] = {1, 1};
    file_space.SelectHyperslab(SelectionOp::Set,
        cstd::span<const uint64_t>(start, 2),
        cstd::span<const uint64_t>(stride, 2),
        cstd::span<const uint64_t>(count, 2),
        cstd::span<const uint64_t>(block_v, 2));

    uint64_t mem_dims[2] = {rows, 1};
    auto mem_space = Dataspace::CreateSimple(cstd::span<const uint64_t>(mem_dims, 2)).value();

    std::vector<double> out(rows);
    for (auto _ : state) {
        ResetAllocator();
        ds.Read(type, mem_space, file_space, out.data());
        benchmark::DoNotOptimize(out.data());
    }
    DestroyFile(file);
    state.SetBytesProcessed(state.iterations() * rows * sizeof(double));
}
BENCHMARK_REGISTER_F(Slice2DFixture, BM_Read2DColSlice)
    ->Args({100, 100, 10, 10});

// ============================================================================
// BM_SetExtent
// ============================================================================

class SetExtentFixture : public bench::CteFixture {};

BENCHMARK_DEFINE_F(SetExtentFixture, BM_SetExtent)(benchmark::State& state) {
    auto file = CreateFile();
    auto root = file.OpenRootGroup();

    uint64_t dim = 100;
    auto space = Dataspace::CreateSimple(cstd::span<const uint64_t>(&dim, 1)).value();
    auto ds = root.CreateDataset("ds", Datatype::Float64(), space).value();

    uint64_t new_dim = 200;
    for (auto _ : state) {
        ResetAllocator();
        new_dim = (new_dim == 200) ? 300 : 200;
        ds.SetExtent(cstd::span<const uint64_t>(&new_dim, 1));
    }
    DestroyFile(file);
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK_REGISTER_F(SetExtentFixture, BM_SetExtent);

// ============================================================================
// BM_ChunkIter
// ============================================================================

class ChunkIterFixture : public bench::CteFixture {};

BENCHMARK_DEFINE_F(ChunkIterFixture, BM_ChunkIter)(benchmark::State& state) {
    int64_t num_chunks = state.range(0);
    int64_t chunk_size = 10;
    int64_t total = num_chunks * chunk_size;

    auto file = CreateFile();
    auto root = file.OpenRootGroup();
    auto ds = Make1DDataset(root, "ds", total, chunk_size);
    auto type = Datatype::Float64();

    // Write all chunks
    uint64_t dim = total;
    auto space = Dataspace::CreateSimple(cstd::span<const uint64_t>(&dim, 1)).value();
    std::vector<double> data(total);
    for (int64_t i = 0; i < total; ++i) data[i] = static_cast<double>(i);
    ds.Write(type, space, space, data.data());

    auto counter_cb = [](const ChunkKey&, uint64_t, void* ud) -> bool {
        (*static_cast<int64_t*>(ud))++;
        return true;
    };

    for (auto _ : state) {
        ResetAllocator();
        int64_t count = 0;
        ds.ChunkIter(counter_cb, &count);
        benchmark::DoNotOptimize(count);
    }
    DestroyFile(file);
    state.counters["chunks/sec"] = benchmark::Counter(
        state.iterations() * num_chunks, benchmark::Counter::kIsRate);
}
BENCHMARK_REGISTER_F(ChunkIterFixture, BM_ChunkIter)
    ->Arg(4)->Arg(10)->Arg(100);
