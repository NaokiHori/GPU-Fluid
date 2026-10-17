#include <metal_stdlib>

using namespace metal;

inline void tdma_y(
    const uint32_t nx,
    const uint32_t ny,
    const float off_diagonal_prefactor,
    device const float * const lower_diagonals,
    device const float * const upper_diagonals,
    device float * const inout,
    device float * const scratchpad,
    uint i
) {
  if (nx <= i) {
    return;
  }
  // NOTE: scratchpad should use NX as stencil size
  //   to avoid race between ux/uy/temperature executions
#define SCRATCHPAD(j) scratchpad[(j) * NX + i]
  {
    const uint j = 0;
    const float lower_diagonal = off_diagonal_prefactor * lower_diagonals[j]; // NOTE: to impose BC
    const float upper_diagonal = off_diagonal_prefactor * upper_diagonals[j];
    const float main_diagonal = 1.f - lower_diagonal - upper_diagonal;
    const float main_diagonal_inv = 1.f / main_diagonal;
    SCRATCHPAD(j) = upper_diagonal * main_diagonal_inv;
    inout[j * nx + i] *= main_diagonal_inv;
  }
  for (uint j = 1; j < ny - 1; j++) {
    const float lower_diagonal = off_diagonal_prefactor * lower_diagonals[j];
    const float upper_diagonal = off_diagonal_prefactor * upper_diagonals[j];
    const float main_diagonal = 1.f - lower_diagonal - upper_diagonal;
    const float denominator_inv = 1.f / (main_diagonal - lower_diagonal * SCRATCHPAD(j - 1));
    SCRATCHPAD(j) = upper_diagonal * denominator_inv;
    inout[j * nx + i] = (inout[j * nx + i] - lower_diagonal * inout[(j - 1) * nx + i]) * denominator_inv;
  }
  {
    const uint j = ny - 1;
    const float lower_diagonal = off_diagonal_prefactor * lower_diagonals[j];
    const float upper_diagonal = off_diagonal_prefactor * upper_diagonals[j]; // NOTE: to impose BC
    const float main_diagonal = 1.f - lower_diagonal - upper_diagonal;
    const float denominator = main_diagonal - lower_diagonal * SCRATCHPAD(j - 1);
    // NOTE: assuming full rank matrix
    inout[j * nx + i] = (inout[j * nx + i] - lower_diagonal * inout[(j - 1) * nx + i]) / denominator;
  }
  for (int j = ny - 2; j >= 0; j--) {
    inout[j * nx + i] -= SCRATCHPAD(j) * inout[(j + 1) * nx + i];
  }
}

kernel void solve_implicit_y(
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
    uint i [[thread_position_in_grid]]
) {
  tdma_y(
      NX - 1,
      NY,
      off_diagonal_prefactor * *dt,
      ux_lower_diagonals,
      ux_upper_diagonals,
      dux,
      scratchpad,
      i
  );
  tdma_y(
      NX,
      NY - 1,
      off_diagonal_prefactor * *dt,
      uy_lower_diagonals,
      uy_upper_diagonals,
      duy,
      scratchpad,
      i
  );
  tdma_y(
      NX,
      NY,
      off_diagonal_prefactor * *dt,
      temperature_lower_diagonals,
      temperature_upper_diagonals,
      dtemperature,
      scratchpad,
      i
  );
}

