// Minimal test of EnsureGpuCteRuntime
#include "gpu_cte_fixture.h"
#include <cstdio>

int main() {
    std::printf("Calling EnsureGpuCteRuntime...\n");
    std::fflush(stdout);
    EnsureGpuCteRuntime();
    std::printf("Success!\n");
    return 0;
}
