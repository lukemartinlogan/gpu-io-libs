#pragma once

#include <cstddef>

namespace gs_trad {

struct GrayScottParams {
    float Du;
    float Dv;
    float F;
    float k;
    float dt;
};

// Matches the iowarp example: 32x32 single-chunk float32 grids.
inline constexpr unsigned kN            = 32;
inline constexpr size_t   kCellsPerGrid = static_cast<size_t>(kN) * kN;
inline constexpr size_t   kBytesPerGrid = kCellsPerGrid * sizeof(float);

} // namespace gs_trad
