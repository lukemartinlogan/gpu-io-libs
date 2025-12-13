# Development Container

Development environment for IOWarp GPU I/O Libraries.

## What's Included

- GCC 13 (C++20)
- CMake 3.31.2
- CUDA 12.6.3
- Build tools (ninja, ccache, gdb)

## Usage

1. Open in VS Code
2. Click "Reopen in Container"
3. Build the project:
   ```bash
   cmake -B build -S .
   cmake --build build
   ```

## Requirements

### For GPU Support
- NVIDIA GPU (compute capability 7.0+)
- NVIDIA drivers installed on host
- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

### For CPU-Only
- Just Docker (project auto-detects and builds without CUDA)

## GPU Setup

If you have an NVIDIA GPU, install the Container Toolkit:

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

Verify GPU access in container:
```bash
nvidia-smi
```

## Notes

- First build takes ~5-10 minutes
- Subsequent starts take seconds
- To rebuild: Command Palette → "Dev Containers: Rebuild Container"
