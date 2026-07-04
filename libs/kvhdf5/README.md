# kvhdf5
An HDF5-like API, built over a key-value store.

## Building
This library does not depend on any parts of the existing HDF5 library. Having `libhdf5-dev` is not necessary. The core is header-only (`src/`); the CMake project builds the test suites, which fetch `iowarp-core` (the CLIO runtime) and Catch2 via `FetchContent`.

### Requirements
- CMake 3.28+
- C++20 compatible compiler
- CUDA toolkit (required for the GPU integration tests; the CPU-only unit tests build without it)

Clone the repository and enter the `kvhdf5` subdirectory:
```bash
git clone https://github.com/lukemartinlogan/gpu-io-libs
cd gpu-io-libs/libs/kvhdf5
```

Configure and build. The simplest path is the bundled CMake presets (`dev` = Debug, `release` = Release), which pin the clang-18 + nvcc toolchain:
```bash
cmake --preset release
cmake --build --preset release
```

Or configure manually. When building `kvhdf5` standalone, point CMake at the CUDA compiler explicitly, otherwise it silently drops to CPU-only and skips the GPU integration target:
```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build build -j"$(nproc)"
```

Build targets:
- `kvhdf5_unit_tests` — CPU-only unit tests (chunking, dataset I/O, tag paths). No CUDA or CLIO runtime.
- `clio_contract_tests` — the CLIO dependency-contract canary: exercises the raw CLIO CTE producer contract with no kvhdf5 code, so a failure here points at CLIO, not at kvhdf5.
- `kvhdf5_e2e_tests` — GPU end-to-end tests of kvhdf5's own producer surface (`GpuCteDataset`, chunking, tag paths) running on the CLIO runtime.

Binaries are written to `build/bin/`. Test sources are organized as `test/unit` (CPU logic), `test/clio` (the contract canary), `test/e2e` (kvhdf5 GPU end-to-end), and `test/support` (shared CTE bring-up).

### Running Tests
Unit tests:
```bash
./build/bin/kvhdf5_unit_tests
```

Run a specific tag:
```bash
./build/bin/kvhdf5_unit_tests '[chunking]'
```

The GPU integration tests need the CLIO runtime environment (`CLIO_BIND_ADDR`, `CHI_REPO_PATH`, `LD_LIBRARY_PATH`), which CMake wires into the CTest registration. Run them via CTest from the build directory:
```bash
cd build && ctest --output-on-failure
```

### Three-way I/O benchmark
The Gray-Scott three-way benchmark (raw disk vs CLIO-sync vs CLIO-async) lives as hidden Catch2 cases inside `kvhdf5_e2e_tests` and is driven by a wrapper script that runs each arm in its own process. Build the `threeway_bench` target to build-and-run it:
```bash
cmake --build build --target threeway_bench
```
Or run the script directly after building `kvhdf5_e2e_tests`:
```bash
bash test/e2e/run_threeway_bench.sh
```
Knobs (grid size, chunking, storage backend, durability) are environment variables — see the header comment in the script and in `gray_scott_threeway_bench.cu`.

### Cleaning build
```bash
rm -rf build
```
