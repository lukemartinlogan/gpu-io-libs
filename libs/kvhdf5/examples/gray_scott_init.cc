// Initialization code for Gray-Scott example.
// Separated into a .cc file to avoid CUDA compilation issues with the
// gpu_cte_fixture.h header which expects host-only compilation.

#include "gpu_cte_fixture.h"
#include <cstdio>

// This function is called from gray_scott_gpu.cu
void GrayScottInitializeRuntime() {
    std::printf("  initializing GPU CTE runtime...\n");
    std::fflush(stdout);
    EnsureGpuCteRuntime();
    std::printf("  GPU CTE runtime ready\n");
    std::fflush(stdout);
}

// Export the globals set by the fixture
extern wrp_cte::core::TagId g_gpu_cte_tag_id;
extern wrp_cte::core::TagId g_gpu_pool_tag_id;
extern chi::PoolId g_gpu_cte_pool_id;
extern std::unique_ptr<wrp_cte::core::Client> g_gpu_cte_client;

void GrayScottGetRuntimeInfo(
    wrp_cte::core::TagId* out_tag_id,
    chi::PoolId* out_pool_id) {
    *out_tag_id = g_gpu_pool_tag_id;
    *out_pool_id = g_gpu_cte_pool_id;
}
