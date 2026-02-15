# Docker Usage

## VSCode Dev Containers (Recommended)

**Open in VSCode:**
1. Open project folder in VSCode
2. Click "Reopen in Container" when prompted (or use Command Palette: "Dev Containers: Reopen in Container")
3. Container builds automatically with all dependencies

**Inside container:**
```bash
# Project is at /workspaces/gpu-io-libs
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Run benchmarks
./build/benches/hdf5_benchmarks
```

## Manual Docker Usage

**Build container:**
```bash
docker build -t gpu-io-libs-dev -f .devcontainer/Dockerfile .devcontainer
```

**Run container:**
```bash
docker run -it --gpus=all --shm-size=2gb \
  -v $(pwd):/workspace \
  -w /workspace \
  gpu-io-libs-dev
```

**Inside container:**
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
./build/benches/hdf5_benchmarks
```

## Container Details

- **Base image**: `nvidia/cuda:12.6.0-devel-ubuntu24.04`
- **CMake**: 4.2.1
- **Compiler**: gcc-13/g++-13
- **User**: `iowarp` (non-root)
- **GPU access**: `--gpus=all`, `--shm-size=2gb` required

## Common Issues

**Can't drop page cache**: Normal - container doesn't have host privileges. Benchmarks measure hot cache only.

**Permission errors**: Files created in container owned by root. Fix: `sudo chown -R $USER:$USER .` on host.
