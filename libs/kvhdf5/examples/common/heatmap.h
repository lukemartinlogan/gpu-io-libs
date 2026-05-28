#pragma once

// Tiny ASCII heatmap shared between Gray-Scott examples. Maps a scalar field
// (typically V concentration) to a 10-step shade ramp and prints it. Each cell
// renders as two characters wide so the output is roughly square in a
// terminal.

#include <cstddef>
#include <cstdio>

namespace gs_common {

inline void DumpHeatmap(const float* v, unsigned n, const char* title) {
    std::printf("\n%s (V concentration, %ux%u)\n", title, n, n);

    float vmin = v[0], vmax = v[0];
    for (size_t i = 0; i < static_cast<size_t>(n) * n; ++i) {
        if (v[i] < vmin) vmin = v[i];
        if (v[i] > vmax) vmax = v[i];
    }
    float range = vmax - vmin;
    if (range < 1e-6f) range = 1e-6f;

    static const char shades[] = " .:-=+*#%@";
    constexpr int n_shades = sizeof(shades) - 1;

    for (unsigned y = 0; y < n; ++y) {
        for (unsigned x = 0; x < n; ++x) {
            float t = (v[y * n + x] - vmin) / range;
            int s = static_cast<int>(t * (n_shades - 1));
            if (s < 0) s = 0;
            if (s >= n_shades) s = n_shades - 1;
            std::putchar(shades[s]);
            std::putchar(shades[s]);
        }
        std::putchar('\n');
    }
    std::printf("range: [%g .. %g]\n", vmin, vmax);
}

} // namespace gs_common
