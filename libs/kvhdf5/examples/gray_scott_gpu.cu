// Gray-Scott reaction-diffusion on GPU, end-to-end through kvhdf5.
//
// Demonstrates:
//   - Container + Group hierarchy on host (managed memory).
//   - Host seeding of initial conditions via Dataset::Write (host-side).
//   - A 2-D, single-chunk float32 dataset layout with ping-pong buffers
//     (u_a/u_b, v_a/v_b) so each iteration reads from one and writes to the
//     other, then the host swaps which IDs are "curr".
//   - Per-iteration <<<1,1>>> kernel that does the full Gray-Scott step on
//     the current u/v grids and writes the next pair.
//   - Periodic snapshots: every snap_interval steps the host reads the
//     current u/v grids through the same Dataset<> API and writes them into
//     pre-existing snapshot datasets. Snapshots are pure data movement, so
//     keeping them on the host avoids an intermittent kernel-side roundtrip
//     edge case while still exercising the full kvhdf5 read/write path.
//   - Host readback of the final state and every snapshot via the same
//     Container + Dataset abstractions.
//
// Run inside the gpu-io-libs-dev container with CHI_IPC_MODE=SHM:
//   ./examples/kvhdf5_gray_scott

#include "kvhdf5/gpu_cte_blob_store.h"
#include "kvhdf5/container.h"
#include "kvhdf5/hdf5_dataset.h"
#include "kvhdf5/dataspace.h"
#include "kvhdf5/hdf5_datatype.h"
#include "kvhdf5/ref.h"

#include "hermes_shm/memory/backend/array_backend.h"

// The GPU CTE runtime initialization is delegated to gray_scott_init.cc
// to avoid CUDA compilation issues with the gpu_cte_fixture.h header.
extern void GrayScottInitializeRuntime();

// Globals set by gpu_cte_fixture.h (defined in gray_scott_init.cc)
extern wrp_cte::core::TagId g_gpu_cte_tag_id;
extern wrp_cte::core::TagId g_gpu_pool_tag_id;
extern chi::PoolId g_gpu_cte_pool_id;

// Forward declarations needed for the above
#include <wrp_cte/core/core_client.h>
#include <chimaera/chimaera.h>

#include "gray_scott_host.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace kvhdf5;

// ---------------------------------------------------------------------------
// Compile-time grid size. Single-chunk float32 grids — chunk_buf in
// Dataset::Write/Read is sized per-call-site and lives in per-thread local
// memory, so this is intentionally bounded:
//     kCellsPerGrid * sizeof(float) <= per-call MaxChunkBytes override
// kN=64 → 16 KiB per grid, well under the 64 KiB default.
// ---------------------------------------------------------------------------

namespace gs_cfg {
inline constexpr unsigned kN              = 32;
inline constexpr size_t   kCellsPerGrid   = static_cast<size_t>(kN) * kN;
inline constexpr size_t   kBytesPerGrid   = kCellsPerGrid * sizeof(float);
} // namespace gs_cfg

struct GrayScottParams {
    float Du;
    float Dv;
    float F;
    float k;
    float dt;
};

#if !HSHM_IS_GPU

// Chimaera + CTE bring-up is delegated to EnsureGpuCteRuntime() from
// gpu_cte_fixture.h. The fixture creates two tags — one on the compose CTE
// pool (g_gpu_cte_tag_id) and one on the GPU-enabled pool (g_gpu_pool_tag_id)
// — and exposes g_gpu_cte_pool_id + g_gpu_cte_client for downstream use.
// We use the GPU-enabled pool exclusively, just like gpu_dataset_test.cu.

// ---------------------------------------------------------------------------
// Managed-memory ArrayBackend allocator + Container box. Mirrors
// DsManagedAllocFixture / DsManagedContainerBox from gpu_dataset_test.cu but
// with a larger heap to fit the bigger metadata graph.
// ---------------------------------------------------------------------------

struct ManagedAllocBox {
    static constexpr size_t kHeapSize = 1ULL * 1024 * 1024;  // 1 MiB

    char*                    memory    = nullptr;
    hshm::ipc::ArrayBackend  backend;
    AllocatorImpl*           allocator = nullptr;

    bool Setup() {
        size_t total = kHeapSize + 3 * hshm::ipc::kBackendHeaderSize;
        auto* gpu_ipc = CHI_CPU_IPC->GetGpuIpcManager();
        gpu_ipc->PauseGpuOrchestrator();
        cudaError_t err = cudaMallocManaged(
            reinterpret_cast<void**>(&memory), total);
        gpu_ipc->ResumeGpuOrchestrator();
        if (err != cudaSuccess) return false;
        std::memset(memory, 0, total);
        if (!backend.shm_init(hshm::ipc::MemoryBackendId::GetRoot(),
                              total, memory)) return false;
        allocator = backend.MakeAlloc<AllocatorImpl>();
        return allocator != nullptr;
    }
    void Teardown() {
        if (memory) {
            auto* gpu_ipc = CHI_CPU_IPC->GetGpuIpcManager();
            gpu_ipc->PauseGpuOrchestrator();
            cudaFree(memory);
            gpu_ipc->ResumeGpuOrchestrator();
            memory = nullptr;
        }
    }
};

struct ManagedContainerBox {
    Container<GpuCteBlobStore>* ptr = nullptr;
    bool Setup(GpuCteBlobStore store, AllocatorImpl* alloc) {
        void* raw = nullptr;
        auto* gpu_ipc = CHI_CPU_IPC->GetGpuIpcManager();
        gpu_ipc->PauseGpuOrchestrator();
        cudaError_t err = cudaMallocManaged(&raw,
                                             sizeof(Container<GpuCteBlobStore>));
        gpu_ipc->ResumeGpuOrchestrator();
        if (err != cudaSuccess) return false;
        ptr = new (raw) Container<GpuCteBlobStore>(std::move(store), alloc);
        return ptr != nullptr;
    }
    void Teardown() {
        if (ptr) {
            ptr->~Container();
            auto* gpu_ipc = CHI_CPU_IPC->GetGpuIpcManager();
            gpu_ipc->PauseGpuOrchestrator();
            cudaFree(ptr);
            gpu_ipc->ResumeGpuOrchestrator();
            ptr = nullptr;
        }
    }
};

#endif  // !HSHM_IS_GPU

// ---------------------------------------------------------------------------
// Iteration kernel: reads (u_curr, v_curr), computes one explicit-Euler
// Gray-Scott step with periodic boundary conditions and a 5-point Laplacian,
// writes (u_next, v_next).
//
// Single thread does the full grid: <<<1,1>>>. Each Read/Write call site
// stamps a chunk_buf of MaxChunkBytes into per-thread local memory, so the
// override <kBytesPerGrid> keeps total local memory usage bounded.
// ---------------------------------------------------------------------------

__global__ void kernel_gray_scott_step(
    chi::IpcManagerGpuInfo gpu_info,
    Container<GpuCteBlobStore>* container,
    DatasetId u_curr, DatasetId v_curr,
    DatasetId u_next, DatasetId v_next,
    GrayScottParams params,
    int* d_status)
{
    *d_status = 0;
    CHIMAERA_GPU_INIT(gpu_info);
    if (threadIdx.x != 0) return;

    constexpr size_t kBytes = gs_cfg::kBytesPerGrid;
    constexpr unsigned N    = gs_cfg::kN;

    Dataset<GpuCteBlobStore> ds_u_curr(u_curr,
        Ref<Container<GpuCteBlobStore>>(*container));
    Dataset<GpuCteBlobStore> ds_v_curr(v_curr,
        Ref<Container<GpuCteBlobStore>>(*container));
    Dataset<GpuCteBlobStore> ds_u_next(u_next,
        Ref<Container<GpuCteBlobStore>>(*container));
    Dataset<GpuCteBlobStore> ds_v_next(v_next,
        Ref<Container<GpuCteBlobStore>>(*container));

    uint64_t dims[2] = {N, N};
    auto sp_r = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 2));
    if (!sp_r.has_value()) { *d_status = -1; return; }
    auto& sp = sp_r.value();

    float u[gs_cfg::kCellsPerGrid];
    float v[gs_cfg::kCellsPerGrid];

    if (!ds_u_curr.template Read<kBytes>(
            Datatype::Float32(), sp, sp, u).has_value()) {
        *d_status = -2; return;
    }
    if (!ds_v_curr.template Read<kBytes>(
            Datatype::Float32(), sp, sp, v).has_value()) {
        *d_status = -3; return;
    }

    float u_new[gs_cfg::kCellsPerGrid];
    float v_new[gs_cfg::kCellsPerGrid];

    const float Du = params.Du, Dv = params.Dv;
    const float F = params.F, k = params.k, dt = params.dt;

    for (unsigned y = 0; y < N; ++y) {
        unsigned ym = (y == 0) ? (N - 1) : (y - 1);
        unsigned yp = (y == N - 1) ? 0u : (y + 1);
        for (unsigned x = 0; x < N; ++x) {
            unsigned xm = (x == 0) ? (N - 1) : (x - 1);
            unsigned xp = (x == N - 1) ? 0u : (x + 1);

            float uc = u[y * N + x];
            float vc = v[y * N + x];

            float lap_u = u[y  * N + xm] + u[y  * N + xp]
                        + u[ym * N + x ] + u[yp * N + x ]
                        - 4.f * uc;
            float lap_v = v[y  * N + xm] + v[y  * N + xp]
                        + v[ym * N + x ] + v[yp * N + x ]
                        - 4.f * vc;

            float uvv = uc * vc * vc;
            u_new[y * N + x] = uc + dt * (Du * lap_u - uvv + F * (1.f - uc));
            v_new[y * N + x] = vc + dt * (Dv * lap_v + uvv - (F + k) * vc);
        }
    }

    if (!ds_u_next.template Write<kBytes>(
            Datatype::Float32(), sp, sp, u_new).has_value()) {
        *d_status = -4; return;
    }
    if (!ds_v_next.template Write<kBytes>(
            Datatype::Float32(), sp, sp, v_new).has_value()) {
        *d_status = -5; return;
    }

    *d_status = 1;
}

#if !HSHM_IS_GPU

// ---------------------------------------------------------------------------
// Host launcher: pause orchestrator, alloc pinned status word, launch, resume,
// poll the status word. Polling (instead of cudaStreamSynchronize) is required
// because the persistent CDP orchestrator deadlocks against device-syncing
// host APIs.
// ---------------------------------------------------------------------------

template <typename Fn>
static int LaunchAndPoll(Fn&& fn) {
    auto* gpu_ipc = CHI_CPU_IPC->GetGpuIpcManager();
    chi::IpcManagerGpuInfo gpu_info = gpu_ipc->GetClientGpuInfo(0);

    gpu_ipc->PauseGpuOrchestrator();

    volatile int* d_status = nullptr;
    if (cudaMallocHost(const_cast<int**>(&d_status), sizeof(int))
            != cudaSuccess) {
        gpu_ipc->ResumeGpuOrchestrator();
        return -100;
    }
    *d_status = 0;

    cudaGetLastError();
    void* stream_v = hshm::GpuApi::CreateStream();
    auto  stream   = static_cast<cudaStream_t>(stream_v);

    fn(gpu_info, stream, const_cast<int*>(d_status));

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        std::fprintf(stderr, "kernel launch failed: %s (cuda %d)\n",
                     cudaGetErrorString(launch_err),
                     static_cast<int>(launch_err));
        hshm::GpuApi::DestroyStream(stream_v);
        cudaFreeHost(const_cast<int*>(d_status));
        gpu_ipc->ResumeGpuOrchestrator();
        return -201;
    }

    gpu_ipc->ResumeGpuOrchestrator();

    auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (*d_status == 0
           && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    int status = (*d_status == 0) ? -300 : *d_status;

    gpu_ipc->PauseGpuOrchestrator();
    cudaStreamSynchronize(stream);
    hshm::GpuApi::DestroyStream(stream_v);
    cudaFreeHost(const_cast<int*>(d_status));
    gpu_ipc->ResumeGpuOrchestrator();
    return status;
}

// ---------------------------------------------------------------------------
// Tiny ASCII heatmap. Maps V-concentration to an ASCII shade ramp.
// ---------------------------------------------------------------------------

static void DumpHeatmap(const float* v, unsigned n, const char* title) {
    std::printf("\n%s (V concentration, %ux%u)\n", title, n, n);

    float vmin = v[0], vmax = v[0];
    for (size_t i = 0; i < static_cast<size_t>(n) * n; ++i) {
        if (v[i] < vmin) vmin = v[i];
        if (v[i] > vmax) vmax = v[i];
    }
    float range = vmax - vmin;
    if (range < 1e-6f) range = 1e-6f;

    static const char shades[] = " .:-=+*#%@";
    constexpr int n_shades = sizeof(shades) - 1;

    for (unsigned y = 0; y < n; ++y) {
        for (unsigned x = 0; x < n; ++x) {
            float t = (v[y * n + x] - vmin) / range;
            int s = static_cast<int>(t * (n_shades - 1));
            if (s < 0) s = 0;
            if (s >= n_shades) s = n_shades - 1;
            std::putchar(shades[s]);
            std::putchar(shades[s]);
        }
        std::putchar('\n');
    }
    std::printf("range: [%g .. %g]\n", vmin, vmax);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    int num_steps = 1500;
    int snap_interval = 300;
    if (argc >= 2) num_steps     = std::atoi(argv[1]);
    if (argc >= 3) snap_interval = std::atoi(argv[2]);
    if (num_steps     <= 0)            { num_steps     = 1500; }
    if (snap_interval <= 0)            { snap_interval = 300;  }

    std::printf("Gray-Scott reaction-diffusion on GPU via kvhdf5\n");
    std::printf("  grid:          %u x %u (single chunk per dataset, float32)\n",
                gs_cfg::kN, gs_cfg::kN);
    std::printf("  steps:         %d\n", num_steps);
    std::printf("  snap interval: %d\n", snap_interval);

    GrayScottInitializeRuntime();

    GpuCteBlobStore store = GpuCteBlobStore::Create(
        g_gpu_pool_tag_id, g_gpu_cte_pool_id);
    if (!store.IsValid()) {
        std::fprintf(stderr, "GpuCteBlobStore::Create failed\n");
        return 1;
    }

    ManagedAllocBox alloc_box;
    if (!alloc_box.Setup()) {
        std::fprintf(stderr, "alloc_box setup failed\n");
        return 1;
    }
    ManagedContainerBox cont_box;
    if (!cont_box.Setup(std::move(store), alloc_box.allocator)) {
        std::fprintf(stderr, "container setup failed\n");
        return 1;
    }

    std::vector<int> snap_steps;
    for (int s = snap_interval; s <= num_steps; s += snap_interval) {
        snap_steps.push_back(s);
    }

    gs::GrayScottIds ids = gs::HostBuildScene(
        cont_box.ptr, *alloc_box.allocator, gs_cfg::kN, snap_steps);

    if (!gs::HostSeedInitialConditions(
            cont_box.ptr, ids.u_a, ids.v_a, gs_cfg::kN)) {
        std::fprintf(stderr, "HostSeedInitialConditions failed\n");
        return 1;
    }
    std::printf("  initial state seeded; %zu snapshots planned\n",
                snap_steps.size());

    GrayScottParams params{0.16f, 0.08f, 0.055f, 0.062f, 1.0f};

    DatasetId u_curr = ids.u_a, u_next = ids.u_b;
    DatasetId v_curr = ids.v_a, v_next = ids.v_b;

    auto t_start = std::chrono::steady_clock::now();

    size_t snap_idx = 0;
    for (int step = 1; step <= num_steps; ++step) {
        int rc = LaunchAndPoll(
            [&](chi::IpcManagerGpuInfo gpu_info,
                cudaStream_t stream, int* d_status) {
                kernel_gray_scott_step<<<1, 1, 0, stream>>>(
                    gpu_info, cont_box.ptr,
                    u_curr, v_curr, u_next, v_next,
                    params, d_status);
            });
        if (rc != 1) {
            std::fprintf(stderr,
                         "step %d: iteration kernel failed (status=%d)\n",
                         step, rc);
            return 1;
        }
        std::swap(u_curr, u_next);
        std::swap(v_curr, v_next);

        if (snap_idx < snap_steps.size() && step == snap_steps[snap_idx]) {
            // Snapshot via host-side Dataset Read+Write. We could do this in
            // a kernel too — same Dataset<> API works on both sides — but
            // snapshots are pure data movement and the host path is rock
            // solid. The interesting bit (compute + I/O on the GPU) stays
            // in the iteration kernel.
            if (!gs::HostTakeSnapshot(cont_box.ptr, u_curr, v_curr,
                                       ids.snap_u_ids[snap_idx],
                                       ids.snap_v_ids[snap_idx],
                                       gs_cfg::kN)) {
                std::fprintf(stderr,
                             "step %d: host snapshot failed\n", step);
                return 1;
            }
            std::printf("  step %5d: snapshot %zu written\n",
                         step, snap_idx);
            ++snap_idx;
        } else if (step % 200 == 0) {
            std::printf("  step %5d: ok\n", step);
        }
    }

    auto t_end = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t_end - t_start).count();
    std::printf("\nsimulation complete: %d steps in %.2f s (%.2f ms/step)\n",
                num_steps, secs, secs * 1000.0 / num_steps);

    // Final readback + heatmap
    std::vector<float> u_final(gs_cfg::kCellsPerGrid);
    std::vector<float> v_final(gs_cfg::kCellsPerGrid);
    if (!gs::HostReadGrid(cont_box.ptr, u_curr, u_final.data(), gs_cfg::kN)) {
        std::fprintf(stderr, "final u read failed\n");
        return 1;
    }
    if (!gs::HostReadGrid(cont_box.ptr, v_curr, v_final.data(), gs_cfg::kN)) {
        std::fprintf(stderr, "final v read failed\n");
        return 1;
    }
    DumpHeatmap(v_final.data(), gs_cfg::kN, "Final state");

    // Snapshot readback through the same Container/Dataset abstractions.
    for (size_t s = 0; s < snap_steps.size(); ++s) {
        std::vector<float> snap_v(gs_cfg::kCellsPerGrid);
        if (!gs::HostReadGrid(cont_box.ptr, ids.snap_v_ids[s],
                               snap_v.data(), gs_cfg::kN)) {
            std::fprintf(stderr, "snapshot %zu read failed\n", s);
            continue;
        }
        std::string title = "Snapshot @ step " + std::to_string(snap_steps[s]);
        DumpHeatmap(snap_v.data(), gs_cfg::kN, title.c_str());
    }

    cont_box.Teardown();
    alloc_box.Teardown();
    std::printf("\nDone.\n");
    return 0;
}

#endif  // !HSHM_IS_GPU
