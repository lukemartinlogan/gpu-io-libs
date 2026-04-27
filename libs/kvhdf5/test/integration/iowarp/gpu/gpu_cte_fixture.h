#pragma once

#include "cuda_compat.h"
#include <chimaera/chimaera.h>
#include <chimaera/bdev/bdev_client.h>
#include <chimaera/bdev/bdev_tasks.h>
#include <wrp_cte/core/core_client.h>
#include <wrp_cte/core/core_tasks.h>
#include <hermes_shm/memory/backend/gpu_shm_mmap.h>
#include <hermes_shm/util/gpu_api.h>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <chrono>

// Initialize the Chimaera runtime and CTE pool once per process, then register
// a RAM storage target and a tag for blob operations.  Callers read the
// module-level globals below after calling EnsureGpuCteRuntime().

inline wrp_cte::core::TagId g_gpu_cte_tag_id{};       // tag on kCtePoolId (512,0)
inline wrp_cte::core::TagId g_gpu_pool_tag_id{};      // tag on g_gpu_cte_pool_id (513,0)
inline chi::PoolId g_gpu_cte_pool_id{};
inline std::unique_ptr<wrp_cte::core::Client> g_gpu_cte_client{};

#if !HSHM_IS_GPU
inline void EnsureGpuCteRuntime() {
    static bool initialized = []() {
        setenv("CHI_IPC_MODE", "SHM", 1);
        setenv("WRP_RUNTIME_CONF", KVHDF5_CHIMAERA_CONF, 1);

        bool ok = chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
        if (!ok) {
            throw std::runtime_error("EnsureGpuCteRuntime: CHIMAERA_INIT failed");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        bool cte_ok = wrp_cte::core::WRP_CTE_CLIENT_INIT();
        if (!cte_ok) {
            throw std::runtime_error("EnsureGpuCteRuntime: WRP_CTE_CLIENT_INIT failed");
        }

        // Compose's wrp_cte_core pool at (512, 0) is CPU-only: its yaml
        // entry uses `pool_query: local`, and the GPU orchestrator isn't up
        // when compose runs. So create a fresh CTE pool with
        // PoolQuery::Dynamic() now that the orchestrator is running —
        // PoolManager::CreatePool will then allocate a GPU container for the
        // pool. Without a GPU container, PutBlobTasks submitted from a GPU
        // kernel complete as no-ops (handler never dispatches).
        chi::PoolId gpu_pool_id(wrp_cte::core::kCtePoolId.major_ + 1,
                                wrp_cte::core::kCtePoolId.minor_);
        g_gpu_cte_client = std::make_unique<wrp_cte::core::Client>(gpu_pool_id);
        wrp_cte::core::CreateParams params;
        auto create_task = g_gpu_cte_client->AsyncCreate(
            chi::PoolQuery::Dynamic(),
            "cte_gpu_blob_pool", gpu_pool_id, params);
        create_task.Wait();
        chi::u32 create_ret = create_task->GetReturnCode();
        if (create_ret != 0) {
            throw std::runtime_error(
                "EnsureGpuCteRuntime: CTE pool create failed (code=" +
                std::to_string(create_ret) + ")");
        }

        // Allow the GPU container for the new pool to finish initializing.
        std::this_thread::sleep_for(std::chrono::milliseconds(200));

        // Create a fresh pinned bdev owned by this test fixture. Compose's
        // bdev at (512, 1) doesn't reliably have a GPU container (its pool
        // was created during compose init before the orchestrator is
        // running, so PoolManager::CreatePool skips GPU container alloc).
        // Matching test_gpu_core.cc's GpuCoreGpuFixture pattern exactly:
        //   1) AsyncRegisterTarget(Local()) — creates bdev on CPU side,
        //      and PoolManager::CreatePool now allocates a GPU container
        //      for it too (orchestrator is up).
        //   2) AsyncRegisterTarget(LocalGpuBcast()) — registers the same
        //      target on the GPU side of our CTE pool.
        chi::PoolId fresh_bdev_id(800, 0);
        auto cpu_reg_task = g_gpu_cte_client->AsyncRegisterTarget(
            "pinned::gpu_cte_fixture_tier",
            chimaera::bdev::BdevType::kPinned,
            16ULL * 1024 * 1024,
            chi::PoolQuery::Local(),
            fresh_bdev_id);
        cpu_reg_task.Wait();
        chi::u32 cpu_reg_ret = cpu_reg_task->GetReturnCode();
        if (cpu_reg_ret != 0) {
            throw std::runtime_error(
                "EnsureGpuCteRuntime: CPU RegisterTarget failed (code=" +
                std::to_string(cpu_reg_ret) + ")");
        }

        // Give the bdev's PostGpuContainerCreate UpdateTask time to land.
        std::this_thread::sleep_for(std::chrono::milliseconds(200));

        auto gpu_reg_task = g_gpu_cte_client->AsyncRegisterTarget(
            "pinned::gpu_cte_fixture_tier",
            chimaera::bdev::BdevType::kPinned,
            16ULL * 1024 * 1024,
            chi::PoolQuery::Local(),
            fresh_bdev_id,
            chi::PoolQuery::LocalGpuBcast());
        gpu_reg_task.Wait();
        chi::u32 ret = gpu_reg_task->GetReturnCode();
        if (ret != 0) {
            throw std::runtime_error(
                "EnsureGpuCteRuntime: GPU RegisterTarget failed (code=" +
                std::to_string(ret) + ")");
        }

        // Create the shared tag on the compose-created CTE pool (512, 0)
        // for test_cpu_client.cc which routes through WRP_CTE_CLIENT.
        auto tag_task = WRP_CTE_CLIENT->AsyncGetOrCreateTag("gpu_cte_test_tag");
        tag_task.Wait();
        g_gpu_cte_tag_id = tag_task->tag_id_;
        g_gpu_cte_pool_id = gpu_pool_id;

        // Create a tag on the GPU-enabled pool (513, 0) for kernel-side and
        // host-side ops that need a GPU container. test_gpu_kernel.cu's
        // PutBlob/GetBlob target this pool/tag; blob_store/typed/container
        // tests create their own per-test tags via g_gpu_cte_client.
        auto gpu_tag_task =
            g_gpu_cte_client->AsyncGetOrCreateTag("gpu_pool_test_tag");
        gpu_tag_task.Wait();
        g_gpu_pool_tag_id = gpu_tag_task->tag_id_;

        // Let the GPU orchestrator fully finish processing the LocalGpuBcast
        // RegisterTarget task before any subsequent test pauses the
        // orchestrator (test_gpu_kernel.cu does this). Without this delay,
        // PauseGpuOrchestrator() can deadlock against an in-flight task.
        std::this_thread::sleep_for(std::chrono::milliseconds(200));

        return true;
    }();
    (void)initialized;
}
#endif  // !HSHM_IS_GPU
