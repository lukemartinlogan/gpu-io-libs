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

// -----------------------------------------------------------------------------
// GPU CTE test fixture.
//
// Approach
// --------
// The compose config at `libs/kvhdf5/config/chimaera_ram.yaml` declares the
// CTE pool `wrp_cte_core` at pool_id 512.0 together with its GPU-visible
// storage tiers (`pinned::cte_pinned_tier`, `hbm::cte_hbm_tier`) inline.
//
// There is a documented hazard with compose-created GPU pools:
//
//     "Compose creates CTE before the GPU orchestrator launches, so the CTE
//      GPU container is never registered."
//       — iowarp-core `context-transfer-engine/benchmark/wrp_cte_gpu_bench.cc`,
//         client-overhead branch (around line 512).
//
// The fix documented in that same file is to manually register the CTE's GPU
// container after CHIMAERA_INIT has brought the orchestrator up:
//
//     PauseGpuOrchestrator()
//     void *gpu_cte = AllocGpuContainer(pool_id, 0, "wrp_cte_core")
//     RegisterGpuOrchestratorContainer(pool_id, gpu_cte)
//     ResumeGpuOrchestrator()
//
// That is exactly what this fixture does. We deliberately DO NOT try to do
// dynamic `AsyncRegisterTarget(LocalGpuBcast())` on a new pool — the YAML
// itself (same file, comment at the top) warns:
//
//     "... the GPU CTE runtime requires GPU-visible storage tiers to be
//      declared in the compose section at startup; dynamic
//      AsyncRegisterTarget() only registers on the CPU side and leaves
//      the GPU-side target list empty."
//
// An earlier version of this fixture used `AsyncRegisterTarget(..., Local())`
// followed by `AsyncRegisterTarget(..., LocalGpuBcast())` on a fresh pool
// (513.0) backed by a fresh bdev (800.0). The second call would submit to
// the GPU orchestrator, the orchestrator logged `RunTask launched OK` and
// then spun in its main `[ORCH] START` loop forever while `.Wait()` on the
// CPU side never returned. That behaviour matches the YAML's documented
// limitation — the GPU target list for a dynamically created pool is empty,
// and the broadcast RegisterTarget task has nowhere to run.
//
// iowarp-core's nominal "working" reference `test_gpu_core.cc`
// (GpuCoreGpuFixture) is not actually a working reference: it declares a
// `WorkOrchestrator *orch = nullptr` and then polls through a null pointer,
// which always falls through the 10-second timeout and returns without ever
// issuing the LocalGpuBcast RegisterTarget call, so every `[gpu][cte]` test
// there is silently skipped. We intentionally do not copy that pattern.
//
// Note on NVHPC: this header is also included from `.cu` files that are
// compiled for device code. The HSHM_IS_GPU guard keeps all host-only code
// (std::thread, std::string, cudaHost* usage, std::unique_ptr) off the
// device path; the global declarations are kept visible so kernels can
// reference the PoolId / TagId PODs.
// -----------------------------------------------------------------------------

// Globals read by tests after EnsureGpuCteRuntime() has returned.
//   g_gpu_cte_tag_id   — shared tag on the CTE pool, created via the CPU client.
//   g_gpu_cte_pool_id  — the CTE pool ID (equals wrp_cte::core::kCtePoolId,
//                        because we reuse the compose-declared pool).
//   g_gpu_cte_client   — client bound to the CTE pool, used by tests to submit
//                        AsyncGetOrCreateTag / AsyncPutBlob / AsyncGetBlob.
inline wrp_cte::core::TagId g_gpu_cte_tag_id{};
inline chi::PoolId g_gpu_cte_pool_id{};
inline std::unique_ptr<wrp_cte::core::Client> g_gpu_cte_client{};

#if !HSHM_IS_GPU
inline void EnsureGpuCteRuntime() {
    static bool initialized = []() {
        // CHI_IPC_MODE=SHM and WRP_RUNTIME_CONF point the client at the
        // chimaera_ram.yaml that declares the CTE pool + GPU storage tiers.
        setenv("CHI_IPC_MODE", "SHM", 1);
        setenv("WRP_RUNTIME_CONF", KVHDF5_CHIMAERA_CONF, 1);

        // kClient + with_runtime=true: same mode used by iowarp-core's
        // GPU CTE benchmark (wrp_cte_gpu_bench.cc). Launches a background
        // runtime (including the GPU orchestrator kernel) in-process.
        bool ok = chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
        if (!ok) {
            throw std::runtime_error(
                "EnsureGpuCteRuntime: CHIMAERA_INIT failed");
        }

        // Let the runtime settle: the orchestrator kernel needs to be up
        // before we ask it to register a GPU container. 500 ms matches
        // iowarp-core's GpuCoreFixture sleep after CHIMAERA_INIT.
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        // WRP_CTE_CLIENT_INIT binds the global WRP_CTE_CLIENT to the
        // compose-created CTE pool (512.0). test_cpu_client.cc uses that
        // global directly, so we must initialize it.
        bool cte_ok = wrp_cte::core::WRP_CTE_CLIENT_INIT();
        if (!cte_ok) {
            throw std::runtime_error(
                "EnsureGpuCteRuntime: WRP_CTE_CLIENT_INIT failed");
        }

        // Use the compose-declared CTE pool. Its storage tiers
        // (pinned::cte_pinned_tier, hbm::cte_hbm_tier) are already visible
        // to both CPU and GPU sides because compose populated them before
        // the runtime finished initialising.
        const chi::PoolId cte_pool_id = wrp_cte::core::kCtePoolId;
        g_gpu_cte_pool_id = cte_pool_id;
        g_gpu_cte_client =
            std::make_unique<wrp_cte::core::Client>(cte_pool_id);

        // ---------------------------------------------------------------
        // Manually register the CTE's GPU container.
        //
        // Compose creates the CTE pool before the GPU orchestrator launches,
        // so PoolManager::CreatePool on the CPU side sees no orchestrator
        // and skips GPU container allocation. Any task submitted with
        // PoolQuery::LocalGpuBcast() afterwards has no GPU container to
        // dispatch into and hangs forever inside the orchestrator's poll
        // loop. The mitigation (from wrp_cte_gpu_bench.cc) is to pause the
        // orchestrator, allocate+register a GPU container ourselves, then
        // resume.
        //
        // Pausing the orchestrator is required because AllocGpuContainer
        // touches shared state the orchestrator reads every tick; without
        // the pause we race and can crash the kernel.
        // ---------------------------------------------------------------
#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
        // CHI_CPU_IPC resolves to chi::IpcManager* in both .cc and .cu
        // translation units; plain CHI_IPC becomes chi::gpu::IpcManager* in
        // .cu code where device methods shadow the host ones.
        auto *ipc = CHI_CPU_IPC;
        if (ipc == nullptr) {
            throw std::runtime_error(
                "EnsureGpuCteRuntime: CHI_CPU_IPC is null after CHIMAERA_INIT");
        }
        auto *gpu_ipc = ipc->GetGpuIpcManager();
        if (gpu_ipc == nullptr) {
            throw std::runtime_error(
                "EnsureGpuCteRuntime: GetGpuIpcManager returned null");
        }
        if (gpu_ipc->gpu_devices_.empty()) {
            throw std::runtime_error(
                "EnsureGpuCteRuntime: no GPU devices available");
        }
        bool did_pause = gpu_ipc->PauseGpuOrchestrator();
        if (!did_pause) {
            throw std::runtime_error(
                "EnsureGpuCteRuntime: PauseGpuOrchestrator returned false");
        }
        void *gpu_cte =
            ipc->AllocGpuContainer(cte_pool_id, 0, "wrp_cte_core");
        if (gpu_cte == nullptr) {
            gpu_ipc->ResumeGpuOrchestrator();
            throw std::runtime_error(
                "EnsureGpuCteRuntime: AllocGpuContainer returned "
                "nullptr for CTE pool");
        }
        gpu_ipc->RegisterGpuOrchestratorContainer(cte_pool_id, gpu_cte);
        gpu_ipc->ResumeGpuOrchestrator();

        // Give the orchestrator a chance to pick up the newly
        // registered container before any subsequent task is
        // submitted; 200 ms matches the bench's sleep.
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
#endif

        // Create the shared tag via the global CPU client. Used by:
        //  - test_cpu_client.cc (submits CPU tasks through WRP_CTE_CLIENT).
        //  - test_gpu_kernel.cu (submits CPU tasks using g_gpu_cte_tag_id).
        // Blob store / container tests each create their own per-test tag
        // via CreateGpuCteTag() below.
        auto tag_task =
            WRP_CTE_CLIENT->AsyncGetOrCreateTag("gpu_cte_test_tag");
        tag_task.Wait();
        if (tag_task->GetReturnCode() != 0) {
            throw std::runtime_error(
                "EnsureGpuCteRuntime: AsyncGetOrCreateTag(gpu_cte_test_tag) "
                "failed (code=" +
                std::to_string(tag_task->GetReturnCode()) + ")");
        }
        g_gpu_cte_tag_id = tag_task->tag_id_;

        // Quiesce before returning. test_gpu_kernel.cu tests will call
        // PauseGpuOrchestrator() immediately after EnsureGpuCteRuntime(),
        // and pausing while an in-flight task is still being processed
        // has been observed to deadlock.
        std::this_thread::sleep_for(std::chrono::milliseconds(200));

        return true;
    }();
    (void)initialized;
}

// Create a per-test tag on the CTE pool.
//
// We only register the tag on the CPU side. The GpuCteBlobStore kernels
// submit every Put/Get/Delete/Exists task with PoolQuery::Local(), meaning
// they always route to the CPU-side container for execution (the GPU
// orchestrator just forwards via the gpu2cpu queue). Dual registering with
// PoolQuery::LocalGpuBcast() was observed to hang indefinitely: the
// GetOrCreateTag task shows "RunTask launched OK" on the GPU orchestrator
// and then never completes, because the CTE GPU container created via the
// wrp_cte_gpu_bench.cc workaround cannot service GetOrCreateTag directly —
// it has no populated target list from dynamic registration (see the
// "dynamic AsyncRegisterTarget ... leaves the GPU-side target list empty"
// note at the top of config/chimaera_ram.yaml). A Local()-only tag is
// sufficient for all tag-id lookups the GPU-routed Put/Get tasks perform
// because those execute on the CPU side.
inline wrp_cte::core::TagId CreateGpuCteTag(const std::string &tag_name) {
    auto cpu_task = g_gpu_cte_client->AsyncGetOrCreateTag(
        tag_name,
        wrp_cte::core::TagId::GetNull(),
        chi::PoolQuery::Local());
    cpu_task.Wait();
    if (cpu_task->GetReturnCode() != 0) {
        throw std::runtime_error(
            "CreateGpuCteTag: CPU GetOrCreateTag failed for '" + tag_name +
            "' (code=" + std::to_string(cpu_task->GetReturnCode()) + ")");
    }
    return cpu_task->tag_id_;
}
#endif  // !HSHM_IS_GPU
