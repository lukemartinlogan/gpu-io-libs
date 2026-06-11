#if HSHM_ENABLE_CUDA

#include <catch2/catch_test_macros.hpp>
#include "gpu_cte_fixture.h"
#include "gpu_container_helpers.h"
#include "kvhdf5/gpu_cte_blob_store.h"
#include "kvhdf5/file.h"
#include "kvhdf5/hdf5_group.h"
#include "kvhdf5/hdf5_dataset.h"
#include "kvhdf5/hdf5_datatype.h"
#include "kvhdf5/dataspace.h"
#include "kvhdf5/ref.h"
#include "hermes_shm/memory/backend/array_backend.h"
#include <thread>
#include <chrono>
#include <cmath>

using namespace kvhdf5;

// ---------------------------------------------------------------------------
// Integration scenario: particle binning + reduction pipeline.
//
// Host pre-builds an empty File via File::Create (kernel-side File::Create is
// known broken; tracked separately as gpu_file_test's [!mayfail] case).
// The kernel then does ALL of:
//   - CreateGroup x2 (/input, /results)
//   - SetAttribute x2 (input.count, results.num_bins)
//   - CreateDataset x3 (positions f32[32] chunk=16, histogram i32[8],
//     stats f32[2])
//   - 3 Dataset::Write calls (positions, histogram, stats)
//
// Per-thread local memory: each Dataset::Write/Read declares a chunk_buf on
// the kernel stack sized by the MaxChunkBytes template parameter (default
// kMaxChunkBytes = 64 KB). At default size, 3 Writes plus the kernel-side
// metadata creation paths exceed the per-launch device-memory budget that
// CUDA pre-provisions for max-occupancy local memory (cuda error 2 at
// launch). This kernel uses tight Write<N> bounds — 256 / 128 / 64 bytes —
// since the actual chunks are tiny (positions chunk = 16*4 = 64 bytes,
// histogram chunk = 8*4 = 32 bytes, stats chunk = 2*4 = 8 bytes).
//
// Compute is pure-register: positions[i] = i / 32.0f → 8-bin histogram,
// mean, variance. After the kernel returns, the host reads all three
// datasets and both attributes back via gpu_container_helpers to confirm
// everything written from the kernel round-trips through the store.
//
// Cross-kernel kernel-write-then-kernel-read of metadata is currently broken
// (BufferDeserializer assert), so verification is host-side rather than a
// second kernel.
// ---------------------------------------------------------------------------

// Tight chunk_buf bounds for the three Write call sites. Each is a generous
// upper bound on the actual chunk size; the runtime KVHDF5_ASSERT inside
// Dataset::Write enforces the real chunk fits.
static constexpr size_t kPosWriteCap   = 256;  // positions chunk = 16 * 4 = 64
static constexpr size_t kHistWriteCap  = 128;  // histogram chunk = 8 * 4 = 32
static constexpr size_t kStatsWriteCap = 64;   // stats chunk     = 2 * 4 = 8

// Pinned result struct. Kernel echoes its computed values so the host can
// sanity-check compute correctness independently of the dataset round-trip.
struct IntegrationTestResult {
    int     status;        // 1 = pass, negative = step error, 0 = not done
    int     hist[8];       // kernel-computed histogram
    float   mean;          // kernel-computed mean
    float   var;           // kernel-computed variance
};

// ---------------------------------------------------------------------------
// KERNEL
// ---------------------------------------------------------------------------

__global__ void kernel_pipeline(
    chi::IpcManagerGpuInfo gpu_info,
    File<GpuCteBlobStore>* file,
    IntegrationTestResult* d_result)
{
    d_result->status = 0;
    d_result->mean   = 0.0f;
    d_result->var    = 0.0f;
    for (int i = 0; i < 8; ++i) d_result->hist[i] = 0;

    CHIMAERA_GPU_INIT(gpu_info);
    if (threadIdx.x != 0) return;

    // ----- 1. Open root, create groups -----
    auto root = file->OpenRootGroup();

    auto input_g_r = root.CreateGroup("input");
    if (!input_g_r.has_value()) { d_result->status = -1; return; }
    auto& input_g = input_g_r.value();

    auto results_g_r = root.CreateGroup("results");
    if (!results_g_r.has_value()) { d_result->status = -2; return; }
    auto& results_g = results_g_r.value();

    // ----- 2. Set attributes -----
    int32_t count_val = 32;
    if (!input_g.SetAttribute("count", Datatype::Int32(), &count_val).has_value()) {
        d_result->status = -3; return;
    }

    int32_t num_bins_val = 8;
    if (!results_g.SetAttribute("num_bins", Datatype::Int32(), &num_bins_val).has_value()) {
        d_result->status = -4; return;
    }

    // ----- 3. Create datasets -----
    uint64_t pos_dims[1] = {32};
    auto pos_space_r = Dataspace::CreateSimple(cstd::span<const uint64_t>(pos_dims, 1));
    if (!pos_space_r.has_value()) { d_result->status = -5; return; }
    auto& pos_space = pos_space_r.value();

    DatasetCreateProps pos_props;
    pos_props.chunk_dims.push_back(16);  // 2 chunks of 16 elements each
    auto pos_ds_r = input_g.CreateDataset("positions", Datatype::Float32(),
                                          pos_space, pos_props);
    if (!pos_ds_r.has_value()) { d_result->status = -6; return; }
    auto& pos_ds = pos_ds_r.value();

    uint64_t hist_dims[1] = {8};
    auto hist_space_r = Dataspace::CreateSimple(cstd::span<const uint64_t>(hist_dims, 1));
    if (!hist_space_r.has_value()) { d_result->status = -7; return; }
    auto& hist_space = hist_space_r.value();
    auto hist_ds_r = results_g.CreateDataset("histogram", Datatype::Int32(), hist_space);
    if (!hist_ds_r.has_value()) { d_result->status = -8; return; }
    auto& hist_ds = hist_ds_r.value();

    uint64_t stats_dims[1] = {2};
    auto stats_space_r = Dataspace::CreateSimple(cstd::span<const uint64_t>(stats_dims, 1));
    if (!stats_space_r.has_value()) { d_result->status = -9; return; }
    auto& stats_space = stats_space_r.value();
    auto stats_ds_r = results_g.CreateDataset("stats", Datatype::Float32(), stats_space);
    if (!stats_ds_r.has_value()) { d_result->status = -10; return; }
    auto& stats_ds = stats_ds_r.value();

    // ----- 4. Build input pattern in registers + write input dataset -----
    float positions[32];
    for (int i = 0; i < 32; ++i) positions[i] = (float)i / 32.0f;

    // chunk_buf #1 — tight Write<256> bound (actual chunk = 64 bytes)
    if (!pos_ds.template Write<kPosWriteCap>(
            Datatype::Float32(), pos_space, pos_space, positions).has_value()) {
        d_result->status = -11; return;
    }

    // ----- 5. Compute histogram + mean + variance in registers -----
    int32_t hist[8] = {0};
    double sum    = 0.0;
    double sum_sq = 0.0;
    for (int i = 0; i < 32; ++i) {
        float p = positions[i];
        int b = (int)(p * 8.0f);
        if (b >= 8) b = 7;
        if (b < 0)  b = 0;
        hist[b]++;
        sum    += p;
        sum_sq += (double)p * (double)p;
    }
    float mean = (float)(sum / 32.0);
    float var  = (float)(sum_sq / 32.0 - (double)mean * (double)mean);

    // ----- 6. Write outputs -----
    // chunk_buf #2 — tight Write<128> bound (actual chunk = 32 bytes)
    if (!hist_ds.template Write<kHistWriteCap>(
            Datatype::Int32(), hist_space, hist_space, hist).has_value()) {
        d_result->status = -12; return;
    }

    // chunk_buf #3 — tight Write<64> bound (actual chunk = 8 bytes)
    float stats_buf[2] = { mean, var };
    if (!stats_ds.template Write<kStatsWriteCap>(
            Datatype::Float32(), stats_space, stats_space, stats_buf).has_value()) {
        d_result->status = -13; return;
    }

    // ----- 7. Echo computed values to pinned result for host cross-check -----
    for (int i = 0; i < 8; ++i) d_result->hist[i] = hist[i];
    d_result->mean   = mean;
    d_result->var    = var;
    d_result->status = 1;
}

// ---------------------------------------------------------------------------
// Host-only: fixture, launcher, TEST_CASE.
// ---------------------------------------------------------------------------

#if !HSHM_IS_GPU

// Inline copies of the FlManaged* fixtures from gpu_file_test.cu, prefixed
// In* to keep symbols distinct in the linked test binary.
struct InManagedAllocFixture {
    static constexpr size_t kHeapSize = 128 * 1024;

    char*                    memory    = nullptr;
    hshm::ipc::ArrayBackend  backend;
    AllocatorImpl*           allocator = nullptr;

    bool Setup() {
        size_t alloc_size = kHeapSize + 3 * hshm::ipc::kBackendHeaderSize;

        auto* gpu_ipc = CHI_CPU_IPC->GetGpuIpcManager();
        gpu_ipc->PauseGpuOrchestrator();
        cudaError_t err = cudaMallocManaged(
            reinterpret_cast<void**>(&memory), alloc_size);
        gpu_ipc->ResumeGpuOrchestrator();
        if (err != cudaSuccess) return false;

        memset(memory, 0, alloc_size);

        if (!backend.shm_init(hshm::ipc::MemoryBackendId::GetRoot(),
                              alloc_size, memory)) {
            return false;
        }

        allocator = backend.MakeAlloc<AllocatorImpl>();
        return allocator != nullptr;
    }

    void Teardown() {
        if (memory) {
            auto* gpu_ipc = CHI_CPU_IPC->GetGpuIpcManager();
            gpu_ipc->PauseGpuOrchestrator();
            cudaFree(memory);
            gpu_ipc->ResumeGpuOrchestrator();
            memory    = nullptr;
            allocator = nullptr;
        }
    }
};

struct InManagedFileBox {
    using FileT = File<GpuCteBlobStore>;

    FileT* ptr = nullptr;

    bool Setup(GpuCteBlobStore blob_store, AllocatorImpl* alloc) {
        void* raw = nullptr;
        auto* gpu_ipc = CHI_CPU_IPC->GetGpuIpcManager();
        gpu_ipc->PauseGpuOrchestrator();
        cudaError_t err = cudaMallocManaged(&raw, sizeof(FileT));
        gpu_ipc->ResumeGpuOrchestrator();
        if (err != cudaSuccess) return false;

        auto fr = FileT::Create(cstd::move(blob_store), Context(alloc));
        if (!fr.has_value()) {
            gpu_ipc->PauseGpuOrchestrator();
            cudaFree(raw);
            gpu_ipc->ResumeGpuOrchestrator();
            return false;
        }
        ptr = new (raw) FileT(cstd::move(fr.value()));
        return ptr != nullptr;
    }

    void Teardown() {
        if (ptr) {
            ptr->~FileT();
            auto* gpu_ipc = CHI_CPU_IPC->GetGpuIpcManager();
            gpu_ipc->PauseGpuOrchestrator();
            cudaFree(ptr);
            gpu_ipc->ResumeGpuOrchestrator();
            ptr = nullptr;
        }
    }
};

struct IntegrationTestFixture {
    InManagedAllocFixture alloc_fixture;
    InManagedFileBox      file_box;

    bool Setup(const char* tag_name) {
        auto tag_task = g_gpu_cte_client->AsyncGetOrCreateTag(tag_name);
        tag_task.Wait();
        wrp_cte::core::TagId tag_id = tag_task->tag_id_;

        GpuCteBlobStore store = GpuCteBlobStore::Create(tag_id, g_gpu_cte_pool_id);
        if (!store.IsValid()) return false;

        if (!alloc_fixture.Setup()) return false;
        if (!file_box.Setup(std::move(store), alloc_fixture.allocator))
            return false;

        return true;
    }

    void Teardown() {
        file_box.Teardown();
        alloc_fixture.Teardown();
    }

    File<GpuCteBlobStore>* FilePtr() { return file_box.ptr; }
};

// ---------------------------------------------------------------------------
// Launcher — Pause/Resume + pinned-status-poll, mirrors gpu_file_test.cu.
// ---------------------------------------------------------------------------

using IntegrationKernelFn = void (*)(chi::IpcManagerGpuInfo,
                                      File<GpuCteBlobStore>*,
                                      IntegrationTestResult*);

static IntegrationTestResult RunIntegrationKernel(
    IntegrationKernelFn kernel,
    File<GpuCteBlobStore>* file_ptr)
{
    auto* gpu_ipc = CHI_CPU_IPC->GetGpuIpcManager();
    chi::IpcManagerGpuInfo gpu_info = gpu_ipc->GetClientGpuInfo(0);

    gpu_ipc->PauseGpuOrchestrator();

    volatile IntegrationTestResult* d_result;
    cudaMallocHost(const_cast<IntegrationTestResult**>(&d_result),
                   sizeof(IntegrationTestResult));
    d_result->status = 0;
    d_result->mean   = 0.0f;
    d_result->var    = 0.0f;
    for (int i = 0; i < 8; ++i) d_result->hist[i] = 0;

    cudaGetLastError();
    void* stream = hshm::GpuApi::CreateStream();
    kernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(
        gpu_info, file_ptr,
        const_cast<IntegrationTestResult*>(d_result));

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        fprintf(stderr,
                "[gpu_integration_test] kernel launch failed: %s (cuda %d)\n",
                cudaGetErrorString(launch_err),
                static_cast<int>(launch_err));
        hshm::GpuApi::DestroyStream(stream);
        cudaFreeHost(const_cast<IntegrationTestResult*>(d_result));
        gpu_ipc->ResumeGpuOrchestrator();
        IntegrationTestResult bad{};
        bad.status = -201;
        return bad;
    }

    gpu_ipc->ResumeGpuOrchestrator();

    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (d_result->status == 0
           && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }

    IntegrationTestResult result;
    result.status = d_result->status;
    result.mean   = d_result->mean;
    result.var    = d_result->var;
    for (int i = 0; i < 8; ++i) result.hist[i] = d_result->hist[i];
    if (result.status == 0) result.status = -300;

    gpu_ipc->PauseGpuOrchestrator();
    cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
    hshm::GpuApi::DestroyStream(stream);
    cudaFreeHost(const_cast<IntegrationTestResult*>(d_result));
    gpu_ipc->ResumeGpuOrchestrator();
    return result;
}

// ---------------------------------------------------------------------------
// TEST_CASE — full pipeline integration.
// ---------------------------------------------------------------------------

TEST_CASE("Integration GPU - kernel pipeline: groups + datasets + attrs + writes, host verifies",
          "[unit][gpu_integration][pipeline]")
{
    EnsureGpuCteRuntime();

    IntegrationTestFixture fx;
    REQUIRE(fx.Setup("gpu_integration_pipeline"));

    auto result = RunIntegrationKernel(kernel_pipeline, fx.FilePtr());

    INFO("kernel status: " << result.status
         << " mean: " << result.mean
         << " var: " << result.var);

    REQUIRE(result.status == 1);

    // Compute kernel correctness: positions[i] = i / 32.0f produces histogram
    // [4,4,4,4,4,4,4,4], mean = (0+1+...+31)/32/32 = 15.5/32 = 0.484375,
    // variance = mean(p^2) - mean^2.
    static constexpr float kExpectedMean = 0.484375f;
    static constexpr float kMeanTol      = 1e-5f;
    static constexpr float kVarTol       = 1e-3f;
    {
        double sum_sq = 0.0;
        for (int i = 0; i < 32; ++i) {
            double p = (double)i / 32.0;
            sum_sq += p * p;
        }
        const float expected_var = (float)(sum_sq / 32.0
            - (double)kExpectedMean * (double)kExpectedMean);

        // Cross-check the kernel's echoed compute values.
        for (int i = 0; i < 8; ++i) {
            INFO("hist[" << i << "] echoed = " << result.hist[i]);
            REQUIRE(result.hist[i] == 4);
        }
        REQUIRE(std::fabs(result.mean - kExpectedMean) < kMeanTol);
        REQUIRE(std::fabs(result.var  - expected_var)  < kVarTol);
    }

    // ----- Host-side dataset round-trip verification -----
    auto& container = fx.FilePtr()->GetContainer();
    GroupId root_gid = container.RootGroup();

    GroupId input_gid;
    REQUIRE(test::HostFindChildGroup(&container, root_gid, "input", &input_gid));
    GroupId results_gid;
    REQUIRE(test::HostFindChildGroup(&container, root_gid, "results", &results_gid));

    DatasetId pos_did;
    REQUIRE(test::HostFindChildDataset(&container, input_gid, "positions", &pos_did));
    DatasetId hist_did;
    REQUIRE(test::HostFindChildDataset(&container, results_gid, "histogram", &hist_did));
    DatasetId stats_did;
    REQUIRE(test::HostFindChildDataset(&container, results_gid, "stats", &stats_did));

    // Read positions back and verify pattern.
    float positions_back[32] = {0};
    REQUIRE(test::HostReadFloat32Dataset(&container, pos_did, positions_back, 32));
    for (int i = 0; i < 32; ++i) {
        INFO("positions[" << i << "] = " << positions_back[i]);
        REQUIRE(std::fabs(positions_back[i] - (float)i / 32.0f) < 1e-6f);
    }

    // Read histogram back and verify uniform distribution.
    int32_t hist_back[8] = {0};
    REQUIRE(test::HostReadInt32Dataset(&container, hist_did, hist_back, 8));
    for (int i = 0; i < 8; ++i) {
        INFO("hist_back[" << i << "] = " << hist_back[i]);
        REQUIRE(hist_back[i] == 4);
    }

    // Read stats back and verify mean+var.
    float stats_back[2] = {0};
    REQUIRE(test::HostReadFloat32Dataset(&container, stats_did, stats_back, 2));
    REQUIRE(std::fabs(stats_back[0] - kExpectedMean) < kMeanTol);
    REQUIRE(std::fabs(stats_back[1] - result.var)    < 1e-5f);

    // Read attributes back and verify.
    int32_t count_back = 0;
    REQUIRE(test::HostGetInt32Attribute(&container, input_gid, "count", &count_back));
    REQUIRE(count_back == 32);

    int32_t num_bins_back = 0;
    REQUIRE(test::HostGetInt32Attribute(&container, results_gid, "num_bins", &num_bins_back));
    REQUIRE(num_bins_back == 8);

    fx.Teardown();
}

#endif  // !HSHM_IS_GPU

#endif  // HSHM_ENABLE_CUDA
