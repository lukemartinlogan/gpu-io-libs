#pragma once
#include <cstdint>
#include "../hdf5/types.h"

extern "C" {
    uint32_t hashword(const uint32_t* key, size_t length, uint32_t init_val);
    uint32_t hashlittle(const void* key, size_t length, uint32_t initval);
    void hashlittle2(const void* key, size_t length, uint32_t* pc, uint32_t* pb);
    uint32_t hashbig(const void* key, size_t length, uint32_t initval);
}

namespace lookup3 {

    __device__ __host__
    inline uint32_t HashWord(cstd::span<const uint32_t> key, uint32_t init_val = 0) {
        return hashword(key.data(), key.size(), init_val);
    }

    __device__ __host__
    inline uint32_t HashLittle(cstd::span<const byte_t> data, uint32_t init_val = 0) {
        return hashlittle(data.data(), data.size(), init_val);
    }

    __device__ __host__
    inline cstd::tuple<uint32_t, uint32_t> HashLittle2(cstd::span<const byte_t> data, uint32_t init_val1 = 0, uint32_t init_val2 = 0) {
        hashlittle2(data.data(), data.size(), &init_val1, &init_val2);
        return { init_val1, init_val2 };
    }

    __device__ __host__
    inline uint32_t HashBig(cstd::span<const byte_t> data, uint32_t init_val = 0) {
        return hashbig(data.data(), data.size(), init_val);
    }
} // namespace lookup3
