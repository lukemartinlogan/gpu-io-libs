#include "gs_kernel.h"

namespace gs_trad {

__global__ void gs_step_kernel(
    const float* __restrict__ u_in,
    const float* __restrict__ v_in,
    float* __restrict__ u_out,
    float* __restrict__ v_out,
    GrayScottParams p)
{
    if (threadIdx.x != 0) return;

    const unsigned N = kN;
    const size_t off = static_cast<size_t>(blockIdx.x) * kCellsPerGrid;
    const float* u = u_in  + off;
    const float* v = v_in  + off;
    float* uo      = u_out + off;
    float* vo      = v_out + off;

    const float Du = p.Du, Dv = p.Dv;
    const float F  = p.F,  k  = p.k,  dt = p.dt;

    for (unsigned y = 0; y < N; ++y) {
        unsigned ym = (y == 0)     ? (N - 1) : (y - 1);
        unsigned yp = (y == N - 1) ? 0u      : (y + 1);
        for (unsigned x = 0; x < N; ++x) {
            unsigned xm = (x == 0)     ? (N - 1) : (x - 1);
            unsigned xp = (x == N - 1) ? 0u      : (x + 1);

            float uc = u[y * N + x];
            float vc = v[y * N + x];

            float lap_u = u[y  * N + xm] + u[y  * N + xp]
                        + u[ym * N + x ] + u[yp * N + x ]
                        - 4.f * uc;
            float lap_v = v[y  * N + xm] + v[y  * N + xp]
                        + v[ym * N + x ] + v[yp * N + x ]
                        - 4.f * vc;

            float uvv = uc * vc * vc;
            uo[y * N + x] = uc + dt * (Du * lap_u - uvv + F * (1.f - uc));
            vo[y * N + x] = vc + dt * (Dv * lap_v + uvv - (F + k) * vc);
        }
    }
}

void LaunchGsStep(const float* d_u_in, const float* d_v_in,
                  float* d_u_out, float* d_v_out,
                  int n_grids, GrayScottParams params,
                  cudaStream_t stream) {
    gs_step_kernel<<<n_grids, 1, 0, stream>>>(
        d_u_in, d_v_in, d_u_out, d_v_out, params);
}

} // namespace gs_trad
