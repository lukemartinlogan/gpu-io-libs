#pragma once

template<typename T>
__device__ __host__
T EightBytesAlignedSize(T val) {
    return val + 7 & ~7;
}
