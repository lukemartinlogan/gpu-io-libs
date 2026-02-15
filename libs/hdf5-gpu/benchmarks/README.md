# HDF5 GPU Reader Benchmarks

Compares GPU HDF5 implementation against standard HDF5 C library.

## Quick Start

**Generate test data:**
```bash
cd benches
uv run generate_data.py
```
Creates `data/bench_data.h5` (278 KB) with various datasets.

**Build:**
```bash
# From project root
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target hdf5_benchmarks -j$(nproc)
```

**Run:**
```bash
# From build directory
./benches/hdf5_benchmarks

# Or from benches directory
../build/benches/hdf5_benchmarks
```

## Command-Line Options

```bash
# Run specific benchmarks
./benches/hdf5_benchmarks --benchmark_filter='BM_GPU'
./benches/hdf5_benchmarks --benchmark_filter='SequentialRead'

# Multiple runs for statistical significance
./benches/hdf5_benchmarks --benchmark_repetitions=5

# CSV output
./benches/hdf5_benchmarks --benchmark_format=csv > results.csv

# List all available benchmarks
./benches/hdf5_benchmarks --benchmark_list_tests
```
(Paths assume running from `build/` directory)

## Understanding Results

**GPU Benchmarks** measure:
- **Time**: Total wall time (Google Benchmark)
- **kernel_ms**: Pure GPU execution (CUDA Events)
- **overhead_ms**: Kernel launch overhead (Total - Kernel)

**CPU Benchmarks** measure:
- **CPU**: Actual execution time (use this for comparison)
- **Time**: Wall time (includes overhead)

## Benchmark Types

- `BM_GPU_FileOpen` - Open HDF5 file on GPU
- `BM_GPU_DatasetOpen` - Open dataset on GPU
- `BM_GPU_SequentialRead_*` - Bulk read using `Dataset::Read()`
- `BM_GPU_HyperslabRead_*` - Read using hyperslab with contiguous optimization
- `BM_CPU_*` - Standard HDF5 library baseline

## What's Being Measured

**Each iteration includes**:
1. Arena allocator Reset() - clears GPU memory
2. File open - parse HDF5 metadata
3. Dataset open - locate dataset in file
4. Data read - transfer data to GPU memory

**Hot cache**: File is cached in RAM (typical result)
**Cold cache**: Not measurable in Docker (requires host privileges)

## Performance Expectations

GPU is currently **50-75x slower** than CPU for typical operations due to:
- CPU polling mechanism adds ~150μs latency per I/O operation
- HDF5 metadata parsing requires ~150-200 small reads
- File open dominates total time (~30ms GPU vs ~0.4ms CPU)

However, GPU read time is **constant** regardless of element count, demonstrating that bulk I/O optimizations are working correctly.
