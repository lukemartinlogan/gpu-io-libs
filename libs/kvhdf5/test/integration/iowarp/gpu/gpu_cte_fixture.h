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
#include <stdexcept>
#include <string>
#include <thread>
#include <chrono>

// Initialize the Chimaera runtime and CTE pool once per process, then register
// a RAM storage target and a tag for blob operations.  Callers read the
// module-level globals below after calling EnsureGpuCteRuntime().

inline wrp_cte::core::TagId g_gpu_cte_tag_id{};

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

        auto reg_task = WRP_CTE_CLIENT->AsyncRegisterTarget(
            "gpu_cte_test_storage",
            chimaera::bdev::BdevType::kRam,
            256ULL * 1024 * 1024,
            chi::PoolQuery::Local(),
            chi::PoolId(601, 0));
        reg_task.Wait();
        chi::u32 ret = reg_task->GetReturnCode();
        if (ret != 0) {
            throw std::runtime_error(
                "EnsureGpuCteRuntime: RegisterTarget failed (code=" +
                std::to_string(ret) + ")");
        }

        auto tag_task = WRP_CTE_CLIENT->AsyncGetOrCreateTag("gpu_cte_test_tag");
        tag_task.Wait();
        g_gpu_cte_tag_id = tag_task->tag_id_;

        return true;
    }();
    (void)initialized;
}
