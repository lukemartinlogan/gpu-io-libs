/*
 * Multi-chunk GPU PutBlob+GetBlob integration test (Phase 2).
 *
 * Proves the device-facing handle's N-chunk path: an N-chunk dataset whose
 * per-chunk Put/Get tasks + distinct data regions live behind one
 * GpuDatasetHandle (a device-side ChunkDesc array). One fused kernel fills each
 * chunk's region and submits its PutBlob; one kernel Gets every chunk back.
 *
 * The kernels are written in the *general* grid-stride form (a block strides
 * over a range of chunks). This test launches a SINGLE block that strides over
 * all N chunks (a genuine range: N loop iterations) — the proven path. Launching
 * multiple concurrent blocks hangs on this iowarp pin: GPU->CPU ClientSend pushes
 * every block to one hardcoded lane (GetLane(0,0)) and concurrent multi-block
 * submit deadlocks (no iowarp test exercises it — all are <<<1,...>>>). See the
 * deferral note at the launch site; cross-block parallelism is Phase 2.5/3.
 *
 * Verification reads every chunk back, asserts byte-identical to the pattern it
 * was filled with, AND asserts distinct chunks differ (so a stuck chunk index
 * can't pass). Reuses the one-time SharedCteEnv server bring-up.
 */

#if (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL

#include <clio_runtime/singletons.h>
#include <clio_ctp/util/gpu_api.h>

#include <kvhdf5/cpu_dataset.h>      // Layout
#include <kvhdf5/gpu_cte_dataset.h>

#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

#if !CTP_IS_DEVICE_PASS
#include <catch2/catch_test_macros.hpp>
#endif
#include "shared_cte_env.h"

using kvhdf5::byte_t;  // raw blob-payload bytes (codebase convention)

namespace {

constexpr unsigned kChunkBytes = 256;
constexpr unsigned kChunkCount = 4;
constexpr unsigned kSeedBase = 0x40u;  // chunk c uses seed (kSeedBase + c)

// Device pattern for a chunk: byte i = (seed ^ i) & 0xFF. Distinct seeds give
// distinct chunk contents (so a stuck chunk index fails verification).
constexpr byte_t Pattern(unsigned seed, unsigned i) {
    return static_cast<byte_t>((seed ^ i) & 0xFFu);
}

}  // namespace

/**
 * Fused fill + PutBlob for every chunk, grid-stride. A block strides over chunks
 * (blockIdx, blockIdx+gridDim, ...). For each chunk: all threads fill its region,
 * fence system-wide so the CPU-side PutBlob's D2H read sees the writes while the
 * kernel is resident, barrier, then thread-0 submits that chunk's Put and waits.
 * The trailing barrier keeps the block in lockstep before the next chunk.
 */
__global__ void McFillWriteKernel(kvhdf5::GpuDatasetHandle h, unsigned seed_base) {
    CHIMAERA_GPU_INIT(h.info_, /*ipc_ptr=*/nullptr);
    (void)g_ipc_manager;
    for (uint32_t c = blockIdx.x; c < h.Count(); c += gridDim.x) {
        byte_t* dst = h.Data(c);
        uint64_t n = h.Size(c);
        for (uint64_t i = threadIdx.x; i < n; i += blockDim.x)
            dst[i] = static_cast<byte_t>(((seed_base + c) ^ i) & 0xFFu);
        __threadfence_system();
        __syncthreads();
        h.Write(c);       // thread-0 only (internal guard)
        __syncthreads();
    }
}

/** GetBlob every chunk back, grid-stride. */
__global__ void McReadKernel(kvhdf5::GpuDatasetHandle h) {
    CHIMAERA_GPU_INIT(h.info_, /*ipc_ptr=*/nullptr);
    (void)g_ipc_manager;
    for (uint32_t c = blockIdx.x; c < h.Count(); c += gridDim.x) {
        h.Read(c);        // thread-0 only (internal guard)
        __syncthreads();
    }
}

#if !CTP_IS_DEVICE_PASS

// Single-block range fill+Write -> zero -> Read -> verify every chunk
// byte-identical to its pattern, and chunks mutually distinct. General over
// chunk count + per-chunk size (read from `ds`). grid=1 = one block strides over
// all chunks (the proven path; >1 block hangs — see the file header note).
static void RunAndVerify(kvhdf5::GpuCteDataset& ds, unsigned seed_base,
                         const char* label) {
    kvhdf5::GpuDatasetHandle h = ds.Handle();
    const uint32_t n = ds.ChunkCount();

    std::vector<std::vector<byte_t>> expected(n);
    for (uint32_t c = 0; c < n; ++c) {
        const uint64_t bytes = ds.ChunkBytes(c);
        expected[c].resize(bytes);
        for (uint64_t i = 0; i < bytes; ++i)
            expected[c][i] = Pattern(seed_base + c, static_cast<unsigned>(i));
    }
    auto ZeroAll = [&] {
        for (uint32_t c = 0; c < n; ++c) {
            std::vector<byte_t> z(ds.ChunkBytes(c));
            ctp::GpuApi::Memcpy(ds.DeviceData(c), z.data(), ds.ChunkBytes(c));
        }
    };

    ZeroAll();
    McFillWriteKernel<<<1, 32>>>(h, seed_base);
    ctp::GpuApi::Synchronize();

    ZeroAll();  // clobber so the GetBlob readback is real
    McReadKernel<<<1, 32>>>(h);
    ctp::GpuApi::Synchronize();

    for (uint32_t c = 0; c < n; ++c) {
        const uint64_t bytes = ds.ChunkBytes(c);
        std::vector<byte_t> back(bytes);
        ctp::GpuApi::Memcpy(back.data(), ds.DeviceData(c), bytes);
        if (std::memcmp(back.data(), expected[c].data(), bytes) != 0)
            std::fprintf(stderr, "[%s] chunk %u mismatch\n", label, c);
        REQUIRE(std::memcmp(back.data(), expected[c].data(), bytes) == 0);
    }
    // Distinct chunks differ: each readback is compared to its own distinct
    // pattern above, so a stuck/aliased chunk index fails the per-chunk check.
    for (uint32_t c = 1; c < n; ++c) REQUIRE(expected[c] != expected[c - 1]);
    std::fprintf(stderr, "[ok] multichunk %s: %u chunks round-tripped (grid=1)\n",
                 label, n);
}

TEST_CASE("GPU multi-chunk PutBlob+GetBlob round trip via dataset handle",
          "[integration][gpu][cte][multichunk]") {
    auto& env = kvhdf5::itest::SharedCteEnv();
    auto* ipc = CLIO_CPU_IPC;
    REQUIRE(ipc->GetGpuIpcManager() != nullptr);
    REQUIRE(ipc->GetGpuQueueCount() >= 1u);

    chi::IpcManagerGpuInfo gpu_info =
        ipc->GetGpuIpcManager()->GetGpuInfo(/*gpu_id=*/0);
    REQUIRE(gpu_info.gpu2cpu_queue != nullptr);

    // 1-D layout split into kChunkCount equal chunks of kChunkBytes (elem_size=1).
    kvhdf5::Layout layout{/*dims=*/{kChunkCount * kChunkBytes},
                          /*chunk_dims=*/{kChunkBytes},
                          /*elem_size=*/1};
    REQUIRE(layout.ChunkCount() == kChunkCount);

    kvhdf5::GpuCteDataset ds(ipc, gpu_info, /*gpu_id=*/0, env.tag_id, layout);
    REQUIRE(ds.ChunkCount() == kChunkCount);

    // Single block grid-strides over ALL chunks (a real range: kChunkCount
    // iterations) — the "B" form (one block handles a range), the proven path on
    // this iowarp pin. >1 block HANGS (concurrent multi-block gpu2cpu Send to the
    // single hardcoded lane deadlocks) — see the file header note; deferred to
    // Phase 2.5/3.
    RunAndVerify(ds, kSeedBase, "1d");
}

// Covers two things the 1-D case above doesn't: (a) the Layout ctor's
// multi-dimensional chunk-coord name derivation (2-D -> "r_c" names via
// ChunkIndexToCoord with rank>1), and (b) GpuCteDataset's MULTI-chunk move ctor
// (3 device allocs + a populated host_descs_ vector). A move that fails to null
// the source's allocs would double-free at teardown and abort the process —
// which ctest reports as a failure.
TEST_CASE("GPU multi-chunk dataset 2-D layout names + move ctor",
          "[integration][gpu][cte][multichunk][move]") {
    auto& env = kvhdf5::itest::SharedCteEnv();
    auto* ipc = CLIO_CPU_IPC;
    REQUIRE(ipc->GetGpuIpcManager() != nullptr);
    chi::IpcManagerGpuInfo gpu_info =
        ipc->GetGpuIpcManager()->GetGpuInfo(/*gpu_id=*/0);
    REQUIRE(gpu_info.gpu2cpu_queue != nullptr);

    // 2x2 grid of 1x1 chunks -> 4 chunks named "0_0","0_1","1_0","1_1".
    constexpr unsigned kElemBytes = 64;
    kvhdf5::Layout layout{/*dims=*/{2, 2}, /*chunk_dims=*/{1, 1},
                          /*elem_size=*/kElemBytes};
    REQUIRE(layout.ChunkCount() == 4u);

    kvhdf5::GpuCteDataset src(ipc, gpu_info, /*gpu_id=*/0, env.tag_id, layout);
    REQUIRE(src.ChunkCount() == 4u);

    // Move-construct; `src` is left null (count 0). Round-trip through the
    // moved-into object to prove its handle + 3 allocs + host_descs_ survived.
    kvhdf5::GpuCteDataset ds(std::move(src));
    REQUIRE(src.ChunkCount() == 0u);
    REQUIRE(ds.ChunkCount() == 4u);
    RunAndVerify(ds, /*seed_base=*/0x80u, "2d-moved");
    // At scope exit: `ds` frees its 3 backends once; `src` (nulled) frees nothing.
    // A double-free here would abort and fail this case.
}

#endif  // !CTP_IS_DEVICE_PASS

#else

// Non-GPU build: nothing to test here.

#endif  // (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL
