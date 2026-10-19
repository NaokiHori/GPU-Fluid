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
#define Q(j) q[(j) * (nxh) + (i)]
#define TMPS(j) tmps[(j) * (nxh) + (i)]
  const float eigenvalue = eigenvalues[i];
  const float lower_diagonal = lower_diagonals[0]; // NOTE: to impose BC
  const float upper_diagonal = upper_diagonals[0];
  const float main_diagonal = - lower_diagonal - upper_diagonal + eigenvalue;
  TMPS(0) = upper_diagonal / main_diagonal;
  Q(0) /= main_diagonal;
  for (int j = 1; j < NY - 1; j++) {
    const float lower_diagonal = lower_diagonals[j];
    const float upper_diagonal = upper_diagonals[j];
    const float main_diagonal = - lower_diagonal - upper_diagonal + eigenvalue;
    const float m = 1.f / (main_diagonal - lower_diagonal * TMPS(j - 1));
    TMPS(j) = upper_diagonal * m;
    Q(j) = (Q(j) - lower_diagonal * Q(j - 1)) * m;
  }
  {
    const int j = NY - 1;
    const float lower_diagonal = lower_diagonals[j];
    const float upper_diagonal = upper_diagonals[j]; // NOTE: to impose BC
    const float main_diagonal = - lower_diagonal - upper_diagonal + eigenvalue;
    const float denominator = main_diagonal - lower_diagonal * TMPS(j - 1);
    // NOTE: assuming i = 0 stores zero eigenvalue
    if (0 == i) {
      Q(j) = 0.f;
    } else {
      Q(j) = (Q(j) - lower_diagonal * Q(j - 1)) / denominator;
    }
  }
  for (int j = NY - 2; j >= 0; j--) {
    Q(j) -= TMPS(j) * Q(j + 1);
  }
}

