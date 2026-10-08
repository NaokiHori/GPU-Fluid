#include <metal_stdlib>

using namespace metal;

inline void tdma_x(
    const uint32_t nx,
    const uint32_t ny,
    const float off_diagonal_prefactor,
    device const float * const lower_diagonals,
    device const float * const upper_diagonals,
    device float * const inout,
    device float * const scratchpad,
    const uint j
) {
  if (ny <= j) {
    return;
  }
  device float * const inout_j = inout + j * nx;
  // NOTE: scratchpad should use NY as stencil size
  //   to avoid race between ux/uy/temperature executions
#define SCRATCHPAD(i) scratchpad[(i) * NY + j]
  {
    const uint i = 0;
    const float lower_diagonal = off_diagonal_prefactor * lower_diagonals[i]; // NOTE: to impose BC
    const float upper_diagonal = off_diagonal_prefactor * upper_diagonals[i];
    const float main_diagonal = 1.f - lower_diagonal - upper_diagonal;
    const float main_diagonal_inv = 1.f / main_diagonal;
    SCRATCHPAD(i) = upper_diagonal * main_diagonal_inv;
    inout_j[i] *= main_diagonal_inv;
  }
  for (uint i = 1; i < nx - 1; i++) {
    const float lower_diagonal = off_diagonal_prefactor * lower_diagonals[i];
    const float upper_diagonal = off_diagonal_prefactor * upper_diagonals[i];
    const float main_diagonal = 1.f - lower_diagonal - upper_diagonal;
    const float denominator_inv = 1.f / (main_diagonal - lower_diagonal * SCRATCHPAD(i - 1));
    SCRATCHPAD(i) = upper_diagonal * denominator_inv;
    inout_j[i] = (inout_j[i] - lower_diagonal * inout_j[i - 1]) * denominator_inv;
  }
  {
    const uint i = nx - 1;
    const float lower_diagonal = off_diagonal_prefactor * lower_diagonals[i];
    const float upper_diagonal = off_diagonal_prefactor * upper_diagonals[i]; // NOTE: to impose BC
    const float main_diagonal = 1.f - lower_diagonal - upper_diagonal;
    const float denominator = main_diagonal - lower_diagonal * SCRATCHPAD(i - 1);
    // NOTE: assuming full rank matrix
    inout_j[i] = (inout_j[i] - lower_diagonal * inout_j[i - 1]) / denominator;
  }
  for (int i = nx - 2; i >= 0; i--) {
    inout_j[i] -= SCRATCHPAD(i) * inout_j[i + 1];
  }
}

kernel void solve_implicit_x(
    constant const float &off_diagonal_prefactor [[buffer(0)]],
    device const float * const dt [[buffer(1)]],
    device const float * const ux_lower_diagonals [[buffer(2)]],
    device const float * const ux_upper_diagonals [[buffer(3)]],
    device const float * const uy_lower_diagonals [[buffer(4)]],
    device const float * const uy_upper_diagonals [[buffer(5)]],
    device const float * const temperature_lower_diagonals [[buffer(6)]],
    device const float * const temperature_upper_diagonals [[buffer(7)]],
    device float * const dux [[buffer(8)]],
    device float * const duy [[buffer(9)]],
    device float * const dtemperature [[buffer(10)]],
    device float * const scratchpad [[buffer(11)]],
    uint j [[thread_position_in_grid]]
) {
  tdma_x(
      NX - 1,
      NY,
      off_diagonal_prefactor * *dt,
      ux_lower_diagonals,
      ux_upper_diagonals,
      dux,
      scratchpad,
      j
  );
  tdma_x(
      NX,
      NY - 1,
      off_diagonal_prefactor * *dt,
      uy_lower_diagonals,
      uy_upper_diagonals,
      duy,
      scratchpad,
      j
  );
  tdma_x(
      NX,
      NY,
      off_diagonal_prefactor * *dt,
      temperature_lower_diagonals,
      temperature_upper_diagonals,
      dtemperature,
      scratchpad,
      j
  );
}
