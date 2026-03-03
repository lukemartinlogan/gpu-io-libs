#pragma once

// CUDA compatibility header
// Ensures __nanosleep is available for hermes_shm's CUDA thread model

#if defined(__CUDACC__)

extern __host__ __device__ void __nanosleep(unsigned int ns);

// Now include the full runtime
#include <cuda_runtime.h>

#endif
