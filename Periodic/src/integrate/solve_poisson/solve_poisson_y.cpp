#include <metal_stdlib>

using namespace metal;

kernel void solve_poisson_y(
    device const float * const lower_diagonals [[buffer(0)]],
    device const float * const upper_diagonals [[buffer(1)]],
    device const float * const eigenvalues [[buffer(2)]],
    device float2 * const q [[buffer(3)]],
    device float * const tmps [[buffer(4)]],
    uint i [[thread_position_in_grid]]
) {
  const uint32_t nxh = NX / 2 + 1;
  if (nxh <= i) {
    return;
  }
  const float eigenvalue = eigenvalues[i];
  const float lower_diagonal = lower_diagonals[0]; // NOTE: to impose BC
  const float upper_diagonal = upper_diagonals[0];
  const float main_diagonal = - lower_diagonal - upper_diagonal + eigenvalue;
  tmps[0 * nxh + i] = upper_diagonal / main_diagonal;
  q[0 * nxh + i] /= main_diagonal;
  for (int j = 1; j < NY - 1; j++) {
    const float lower_diagonal = lower_diagonals[j];
    const float upper_diagonal = upper_diagonals[j];
    const float main_diagonal = - lower_diagonal - upper_diagonal + eigenvalue;
    const float m = 1.f / (main_diagonal - lower_diagonal * tmps[(j - 1) * nxh + i]);
    tmps[j * nxh + i] = upper_diagonal * m;
    q[j * nxh + i] = (q[j * nxh + i] - lower_diagonal * q[(j - 1) * nxh + i]) * m;
  }
  {
    const int j = NY - 1;
    const float lower_diagonal = lower_diagonals[j];
    const float upper_diagonal = upper_diagonals[j]; // NOTE: to impose BC
    const float main_diagonal = - lower_diagonal - upper_diagonal + eigenvalue;
    const float denominator = main_diagonal - lower_diagonal * tmps[(j - 1) * nxh + i];
    // NOTE: assuming i = 0 stores zero eigenvalue
    if (0 == i) {
      q[j * nxh + i] = 0.f;
    } else {
      q[j * nxh + i] = (q[j * nxh + i] - lower_diagonal * q[(j - 1) * nxh + i]) / denominator;
    }
  }
  for (int j = NY - 2; j >= 0; j--) {
    q[j * nxh + i] -= tmps[j * nxh + i] * q[(j + 1) * nxh + i];
  }
}

