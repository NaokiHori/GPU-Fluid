#include <metal_stdlib>

using namespace metal;

kernel void solve_poisson_y(
    device const float * const lower_diagonals [[buffer(0)]],
    device const float * const upper_diagonals [[buffer(1)]],
    device const float * const inverse_values [[buffer(2)]],
    device float2 * const values [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
  const uint32_t nxh = NX / 2 + 1;
  if (nxh <= i) {
    return;
  }
#define VALUES(j) values[(j) * (nxh) + (i)]
#define INVERSE_VALUES(j) inverse_values[(j) * (nxh) + (i)]
  // forward sweep
  VALUES(0) *= INVERSE_VALUES(0);
  for (int j = 1; j < NY; j++) {
    const float lower_diagonal = lower_diagonals[j];
    VALUES(j) = fma(- lower_diagonal, VALUES(j - 1), VALUES(j)) * INVERSE_VALUES(j);
  }
  // backward substitution
  for (int j = NY - 2; j >= 0; j--) {
    const float upper_diagonal = upper_diagonals[j];
    VALUES(j) = fma(- upper_diagonal * INVERSE_VALUES(j), VALUES(j + 1), VALUES(j));
  }
}

