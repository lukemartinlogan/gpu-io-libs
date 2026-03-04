# kvhdf5
An HDF5-like API, built over a key-value store.

## Building
This library does not depend on any parts of the existing HDF5 library. Having `libhdf5-dev` is not necessary.

### Requirements
- Cmake 3.28+
- C++20 compatiable compiler

Clone the repository:
```bash
git clone https://github.com/lukemartinlogan/gpu-io-libs
cd gpu-io-libs
```

Enter the `kvhdf5` subdirectory:
```bash
cd libs/kvhdf5
```

Configure CMake for a debug build:
```bash
cmake -B build -S . -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_BUILD_TYPE=Debug
```

Or for a release build:
```bash
cmake -B build -S . -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_BUILD_TYPE=Release
```

Build tests & benchmarks:
```bash
cmake --build build --target kvhdf5_tests kvhdf5_benchmarks -j\$(nproc)
```

### Running Tests
Run all tests:
```bash
./build/test/kvhdf5_tests
```

Run a specific tag:
```bash
./build/test/kvhdf5_tests '[serde]'
```

### Running Benchmarks
Run all benchmarks:
```bash
./build/bench/kvhdf5_benchmarks --benchmark_out=/bench/results.csv --benchmark_out_format=csv
```

Run a specific one by filter:
```bash
./build/bench/kvhdf5_benchmarks --benchmark_filter='BM_RawCtePut' --benchmark_out=/bench/results.csv --benchmark_out_format=csv
```

### Cleaning build
```bash
rm -rf build
```
