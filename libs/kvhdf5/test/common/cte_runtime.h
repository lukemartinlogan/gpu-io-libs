#pragma once

#include "cuda_compat.h"
#include "chimaera/chimaera.h"
#include "wrp_cte/core/core_client.h"
#include <stdexcept>
#include <string>

// Initialize the Chimaera runtime and CTE pool once per process.
// Inline + static local ensures thread-safe single initialization
// even when included from multiple translation units.
inline void EnsureCteRuntime() {
    static bool initialized = []() {
        bool success = chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
        if (!success) {
            throw std::runtime_error("Failed to initialize Chimaera");
        }
        bool cte_ok = wrp_cte::core::WRP_CTE_CLIENT_INIT();
        if (!cte_ok) {
            throw std::runtime_error("Failed to initialize CTE client pool");
        }
        auto reg_task = WRP_CTE_CLIENT->AsyncRegisterTarget(
            "ram_test_storage",
            chimaera::bdev::BdevType::kRam,
            256ULL * 1024 * 1024,  // 256 MB
            chi::PoolQuery::Local(),
            chi::PoolId(600, 0));
        reg_task.Wait();
        chi::u32 ret = reg_task->GetReturnCode();
        if (ret != 0) {
            throw std::runtime_error(
                "Failed to register RAM storage target (code=" + std::to_string(ret) + ")");
        }
        return true;
    }();
    (void)initialized;
}
