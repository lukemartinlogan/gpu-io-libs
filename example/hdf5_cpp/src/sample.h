#pragma once

#include <H5Cpp.h>

struct Sample {
    uint64_t timestamp_ms;
    uint64_t memory_kb;
    uint64_t cpu_time;

    static H5::CompType h5_comp_ty() {
        H5::CompType ty(sizeof(Sample));

        ty.insertMember("timestamp_ms", HOFFSET(Sample, timestamp_ms), H5::PredType::NATIVE_UINT64);
        ty.insertMember("memory_kb", HOFFSET(Sample, memory_kb), H5::PredType::NATIVE_UINT64);
        ty.insertMember("cpu_time", HOFFSET(Sample, cpu_time), H5::PredType::NATIVE_UINT64);

        return ty;
    }
};
