// IowarpKernelGrayScottStep definition. Logic identical to the original
// kernel that lived in gray_scott_gpu.cu before the bench-driven extraction.

#include "iowarp_step.h"

#include "kvhdf5/hdf5_dataset.h"
#include "kvhdf5/dataspace.h"
#include "kvhdf5/hdf5_datatype.h"
#include "kvhdf5/ref.h"

namespace gs_iowarp {

__global__ void IowarpKernelGrayScottStep(
    chi::IpcManagerGpuInfo gpu_info,
    kvhdf5::Container<kvhdf5::GpuCteBlobStore>* container,
    kvhdf5::DatasetId u_curr, kvhdf5::DatasetId v_curr,
    kvhdf5::DatasetId u_next, kvhdf5::DatasetId v_next,
    GrayScottParams params,
    int* d_status)
{
    *d_status = 0;
    CHIMAERA_GPU_INIT(gpu_info);
    if (threadIdx.x != 0) return;

    constexpr size_t kBytes = cfg::kBytesPerGrid;
    constexpr unsigned N    = cfg::kN;

    kvhdf5::Dataset<kvhdf5::GpuCteBlobStore> ds_u_curr(u_curr,
        kvhdf5::Ref<kvhdf5::Container<kvhdf5::GpuCteBlobStore>>(*container));
    kvhdf5::Dataset<kvhdf5::GpuCteBlobStore> ds_v_curr(v_curr,
        kvhdf5::Ref<kvhdf5::Container<kvhdf5::GpuCteBlobStore>>(*container));
    kvhdf5::Dataset<kvhdf5::GpuCteBlobStore> ds_u_next(u_next,
        kvhdf5::Ref<kvhdf5::Container<kvhdf5::GpuCteBlobStore>>(*container));
    kvhdf5::Dataset<kvhdf5::GpuCteBlobStore> ds_v_next(v_next,
        kvhdf5::Ref<kvhdf5::Container<kvhdf5::GpuCteBlobStore>>(*container));

    uint64_t dims[2] = {N, N};
    auto sp_r = kvhdf5::Dataspace::CreateSimple(
        cstd::span<const uint64_t>(dims, 2));
    if (!sp_r.has_value()) { *d_status = -1; return; }
    auto& sp = sp_r.value();

    float u[cfg::kCellsPerGrid];
    float v[cfg::kCellsPerGrid];

    if (!ds_u_curr.template Read<kBytes>(
            kvhdf5::Datatype::Float32(), sp, sp, u).has_value()) {
        *d_status = -2; return;
    }
    if (!ds_v_curr.template Read<kBytes>(
            kvhdf5::Datatype::Float32(), sp, sp, v).has_value()) {
        *d_status = -3; return;
    }

    float u_new[cfg::kCellsPerGrid];
    float v_new[cfg::kCellsPerGrid];

    const float Du = params.Du, Dv = params.Dv;
    const float F = params.F, k = params.k, dt = params.dt;

    for (unsigned y = 0; y < N; ++y) {
        unsigned ym = (y == 0) ? (N - 1) : (y - 1);
        unsigned yp = (y == N - 1) ? 0u : (y + 1);
        for (unsigned x = 0; x < N; ++x) {
            unsigned xm = (x == 0) ? (N - 1) : (x - 1);
            unsigned xp = (x == N - 1) ? 0u : (x + 1);

            float uc = u[y * N + x];
            float vc = v[y * N + x];

            float lap_u = u[y  * N + xm] + u[y  * N + xp]
                        + u[ym * N + x ] + u[yp * N + x ]
                        - 4.f * uc;
            float lap_v = v[y  * N + xm] + v[y  * N + xp]
                        + v[ym * N + x ] + v[yp * N + x ]
                        - 4.f * vc;

            float uvv = uc * vc * vc;
            u_new[y * N + x] = uc + dt * (Du * lap_u - uvv + F * (1.f - uc));
            v_new[y * N + x] = vc + dt * (Dv * lap_v + uvv - (F + k) * vc);
        }
    }

    if (!ds_u_next.template Write<kBytes>(
            kvhdf5::Datatype::Float32(), sp, sp, u_new).has_value()) {
        *d_status = -4; return;
    }
    if (!ds_v_next.template Write<kBytes>(
            kvhdf5::Datatype::Float32(), sp, sp, v_new).has_value()) {
        *d_status = -5; return;
    }

    *d_status = 1;
}

} // namespace gs_iowarp
