# kvhdf5 Gray-Scott example

A Gray-Scott reaction-diffusion simulation that runs end-to-end on a GPU
through the kvhdf5 storage API. The iteration kernel reads / writes its
state through `Dataset<>` (CTE-backed), and the host periodically snapshots
the current state into pre-created snapshot datasets via the same API.

The point isn't the simulation itself — it's that the GPU kernel does
all the per-step compute *and* I/O through kvhdf5's hierarchical
container/group/dataset/chunk abstractions, with the host reading the
results back through the exact same API.

## What you need

- An NVIDIA GPU (sm_89 / RTX 4090 is what's hardcoded in `CMakeLists.txt`;
  another arch will compile but will fail at kernel launch).
- Docker with the NVIDIA Container Toolkit installed and working
  (`docker run --gpus=all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi`
  should print your GPU).
- ~5 GB of disk for the build directory.

Everything else (CUDA 12.6, clang-18, CMake, iowarp-core, etc.) is provided
by the devcontainer.

## 1. Clone

```bash
git clone <repo-url> gpu-io-libs
cd gpu-io-libs
```

## 2. Build and start the devcontainer

If you use VS Code, "Reopen in Container" handles everything. From the CLI:

```bash
# Build the image (one-time, several minutes — installs CUDA, clang-18, etc.)
docker build -t gpu-io-libs:dev .devcontainer/

# Start a long-running container with the project bind-mounted at /workspace
docker run -d --name gpu-io-libs-dev \
  --privileged --gpus=all --shm-size=4gb \
  -v "$PWD:/workspace" \
  --user iowarp -w /workspace \
  gpu-io-libs:dev sleep infinity

# Sanity check
docker exec gpu-io-libs-dev whoami        # → iowarp
docker exec gpu-io-libs-dev nvidia-smi    # confirms GPU is visible
```

If the container already exists, just `docker start gpu-io-libs-dev`.

## 3. Configure CMake (one-time)

This fetches iowarp-core and CCCL over the network. Cold runs take a few
minutes:

```bash
docker exec gpu-io-libs-dev bash -c \
  "cd /workspace/libs/kvhdf5 && cmake --preset dev"
```

## 4. Build the Gray-Scott example

```bash
docker exec gpu-io-libs-dev bash -c \
  "cd /workspace/libs/kvhdf5 && cmake --build --preset dev --target kvhdf5_gray_scott"
```

Output: `libs/kvhdf5/build/examples/kvhdf5_gray_scott`.

## 5. Run

The Chimaera runtime uses POSIX shared memory; if a previous run crashed
it can leave segments behind, so clean before and after. The example
internally selects SHM-only IPC (no TCP port collisions across runs); you
don't need to set anything on the command line.

```bash
# pre-cleanup (idempotent)
docker exec gpu-io-libs-dev bash -c \
  "pkill -9 -f kvhdf5_ ; rm -f /dev/shm/*chimaera* /dev/shm/*hshm* /dev/shm/*iowarp* /dev/shm/*hermes* ; true"

# run: <num_steps> <snap_interval>
docker exec gpu-io-libs-dev bash -c \
  "cd /workspace/libs/kvhdf5/build && ./examples/kvhdf5_gray_scott 1500 500"

# post-cleanup
docker exec gpu-io-libs-dev bash -c \
  "pkill -9 -f kvhdf5_ ; rm -f /dev/shm/*chimaera* /dev/shm/*hshm* /dev/shm/*iowarp* /dev/shm/*hermes* ; true"
```

Expected output: a "simulation complete" line with timing, then ASCII
heatmaps for the final state and each snapshot. At 32×32 with the default
parameters (F=0.055, k=0.062) the system reaches a stable radial spot
(central V-rich region with diffusion rings) around step ~500.

```
Final state (V concentration, 32x32)
                              ......
                          ..::======::..
                        ..--**##%%##**--..
                        ::**%%@@%%@@%%**::
                      ..==##%%%%%%%%%%##==..
                      ..==%%%%%%%%%%%%%%==..
                      ..==##%%%%%%%%%%##==..
                        ::**%%@@%%@@%%**::
                        ..--**##%%##**--..
                          ..::======::..
                              ......
range: [7.29e-12 .. 0.460]
```

`@` is the V-concentration peak; the ramp goes through `%#*+=-:.` down to
background `(space)`.

## Quick reference

| Action | Command |
|---|---|
| Configure | `cmake --preset dev` (inside container, in `libs/kvhdf5`) |
| Build example | `cmake --build --preset dev --target kvhdf5_gray_scott` |
| Build everything | `cmake --build --preset dev` |
| Run example | `./examples/kvhdf5_gray_scott <steps> <snap_interval>` |
| Clean shm before/after run | `pkill -9 -f kvhdf5_ ; rm -f /dev/shm/*chimaera* /dev/shm/*hshm* /dev/shm/*iowarp* /dev/shm/*hermes*` |
| Clean entire build | `rm -rf libs/kvhdf5/build` (then re-run step 3) |

## Internals at a glance

- **`gray_scott_init.cc`** — Chimaera + CTE runtime bring-up via the test
  fixture. Isolated to a plain `.cc` because the fixture's CHI macros
  don't compose with the CUDA device-pass.
- **`gray_scott_host.cc/.h`** — host helpers: builds the
  `/sim` and `/snapshots` group hierarchy, pre-creates ping-pong (`u_a/u_b`,
  `v_a/v_b`) and snapshot datasets, host-side initial seeding and snapshot
  copying.
- **`gray_scott_gpu.cu`** — the iteration kernel + `main()`. The
  iteration kernel does the full Gray-Scott step: reads `u_curr`/`v_curr`
  via `Dataset::Read`, computes a 5-point Laplacian + reaction-diffusion
  update, writes `u_next`/`v_next` via `Dataset::Write`. Single-thread
  launch (`<<<1, 1>>>`) per iteration; the host swaps which IDs are "curr"
  between launches.

## Troubleshooting

- **"timeout: the monitored command dumped core" / SIGSEGV at startup** —
  almost always stale shared-memory state from a previous crash. Run the
  pre-cleanup command and try again.
- **Nothing prints / runtime warnings then nothing** — wait. Cold first
  configure pulls iowarp-core and rebuilds the entire dependency graph;
  it's not unusual for the first build to take 5–10 minutes.
- **Kernel launch fails with "out of memory" (CUDA error 2)** — per-thread
  local memory limit. The kernel sizes its in-register state to fit a
  32×32 grid with the current MaxChunkBytes overrides. If you bump `kN`
  (line 61 of `gray_scott_gpu.cu`) you'll likely need to drop the
  `Read<...>` / `Write<...>` template parameters too.
- **Patterns evolve into all-zero / all-flat** — increase `dt` carefully or
  pick a different (F, k) point in `gray_scott_gpu.cu` (search for the
  `GrayScottParams` literal). The defaults `F=0.055, k=0.062` produce a
  stable spot on a 32×32 grid; `F=0.029, k=0.057` (fingerprints) needs a
  larger grid to be visually interesting.
