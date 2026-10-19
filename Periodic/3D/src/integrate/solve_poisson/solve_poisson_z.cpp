#include <metal_stdlib>

using namespace metal;

#define Q(k) q[((k) * (NY) + (j)) * (nxh) + (i)]
#define TMPS(k) tmps[((k) * (NY) + (j)) * (nxh) + (i)]

kernel void solve_poisson_z(
    device const float * const lower_diagonals [[buffer(0)]],
    device const float * const upper_diagonals [[buffer(1)]],
    device const float * const x_eigenvalues [[buffer(2)]],
    device const float * const y_eigenvalues [[buffer(3)]],
    device float2 * const q [[buffer(4)]],
    device float * const tmps [[buffer(5)]],
    uint2 id [[thread_position_in_grid]]
) {
  const uint32_t i = id.x;
  const uint32_t j = id.y;
  const uint32_t nxh = NX / 2 + 1;
  if (nxh <= i || NY <= j) {
    return;
  }
  const float eigenvalue = x_eigenvalues[i] + y_eigenvalues[j];
  const float lower_diagonal = lower_diagonals[0]; // NOTE: to impose BC
  const float upper_diagonal = upper_diagonals[0];
  const float main_diagonal = - lower_diagonal - upper_diagonal + eigenvalue;
  TMPS(0) = upper_diagonal / main_diagonal;
  Q(0) /= main_diagonal;
  for (int k = 1; k < NZ - 1; k++) {
    const float lower_diagonal = lower_diagonals[k];
    const float upper_diagonal = upper_diagonals[k];
    const float main_diagonal = - lower_diagonal - upper_diagonal + eigenvalue;
    const float m = 1.f / (main_diagonal - lower_diagonal * TMPS(k - 1));
    TMPS(k) = upper_diagonal * m;
    Q(k) = (Q(k) - lower_diagonal * Q(k - 1)) * m;
  }
  {
    const int k = NZ - 1;
    const float lower_diagonal = lower_diagonals[k];
    const float upper_diagonal = upper_diagonals[k]; // NOTE: to impose BC
    const float main_diagonal = - lower_diagonal - upper_diagonal + eigenvalue;
    const float denominator = main_diagonal - lower_diagonal * TMPS(k - 1);
    // NOTE: assuming i = j = 0 stores zero eigenvalue
    if (0 == i && 0 == j) {
      Q(k) = 0.f;
    } else {
      Q(k) = (Q(k) - lower_diagonal * Q(k - 1)) / denominator;
    }
  }
  for (int k = NZ - 2; k >= 0; k--) {
    Q(k) -= TMPS(k) * Q(k + 1);
  }
}

