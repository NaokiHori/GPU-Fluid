#include <metal_stdlib>

using namespace metal;

inline void tdma_z(
    const uint32_t nx,
    const uint32_t ny,
    const uint32_t nz,
    const float off_diagonal_prefactor,
    device const float * const lower_diagonals,
    device const float * const upper_diagonals,
    device float * const inout,
    device float * const scratchpad,
    uint2 id
) {
  const uint32_t i = id.x;
  const uint32_t j = id.y;
  if (nx <= i || ny <= j) {
    return;
  }
#define SCRATCHPAD(k) scratchpad[((k) * ny + (j)) * nx + i]
#define INOUT(k) inout[((k) * ny + (j)) * nx + i]
  {
    const uint k = 0;
    const float lower_diagonal = off_diagonal_prefactor * lower_diagonals[k]; // NOTE: to impose BC
    const float upper_diagonal = off_diagonal_prefactor * upper_diagonals[k];
    const float main_diagonal = 1.f - lower_diagonal - upper_diagonal;
    const float main_diagonal_inv = 1.f / main_diagonal;
    SCRATCHPAD(k) = upper_diagonal * main_diagonal_inv;
    INOUT(k) *= main_diagonal_inv;
  }
  for (uint k = 1; k < nz - 1; k++) {
    const float lower_diagonal = off_diagonal_prefactor * lower_diagonals[k];
    const float upper_diagonal = off_diagonal_prefactor * upper_diagonals[k];
    const float main_diagonal = 1.f - lower_diagonal - upper_diagonal;
    const float denominator_inv = 1.f / (main_diagonal - lower_diagonal * SCRATCHPAD(k - 1));
    SCRATCHPAD(k) = upper_diagonal * denominator_inv;
    INOUT(k) = (INOUT(k) - lower_diagonal * INOUT(k - 1)) * denominator_inv;
  }
  {
    const uint k = nz - 1;
    const float lower_diagonal = off_diagonal_prefactor * lower_diagonals[k];
    const float upper_diagonal = off_diagonal_prefactor * upper_diagonals[k]; // NOTE: to impose BC
    const float main_diagonal = 1.f - lower_diagonal - upper_diagonal;
    const float denominator = main_diagonal - lower_diagonal * SCRATCHPAD(k - 1);
    // NOTE: assuming full rank matrix
    INOUT(k) = (INOUT(k) - lower_diagonal * INOUT(k - 1)) / denominator;
  }
  for (int k = nz - 2; k >= 0; k--) {
    INOUT(k) -= SCRATCHPAD(k) * INOUT(k + 1);
  }
}

kernel void solve_implicit_z(
    constant const float &off_diagonal_prefactor [[buffer(0)]],
    device const float * const dt [[buffer(1)]],
    device const float * const ux_lower_diagonals [[buffer(2)]],
    device const float * const ux_upper_diagonals [[buffer(3)]],
    device const float * const uy_lower_diagonals [[buffer(4)]],
    device const float * const uy_upper_diagonals [[buffer(5)]],
    device const float * const uz_lower_diagonals [[buffer(6)]],
    device const float * const uz_upper_diagonals [[buffer(7)]],
    device const float * const temperature_lower_diagonals [[buffer(8)]],
    device const float * const temperature_upper_diagonals [[buffer(9)]],
    device float * const dux [[buffer(10)]],
    device float * const duy [[buffer(11)]],
    device float * const duz [[buffer(12)]],
    device float * const dtemperature [[buffer(13)]],
    device float * const scratchpad [[buffer(14)]],
    uint2 id [[thread_position_in_grid]]
) {
  tdma_z(
      NX,
      NY,
      NZ,
      off_diagonal_prefactor * *dt,
      ux_lower_diagonals,
      ux_upper_diagonals,
      dux,
      scratchpad,
      id
  );
  tdma_z(
      NX,
      NY,
      NZ,
      off_diagonal_prefactor * *dt,
      uy_lower_diagonals,
      uy_upper_diagonals,
      duy,
      scratchpad,
      id
  );
  tdma_z(
      NX,
      NY,
      NZ - 1,
      off_diagonal_prefactor * *dt,
      uz_lower_diagonals,
      uz_upper_diagonals,
      duz,
      scratchpad,
      id
  );
  tdma_z(
      NX,
      NY,
      NZ,
      off_diagonal_prefactor * *dt,
      temperature_lower_diagonals,
      temperature_upper_diagonals,
      dtemperature,
      scratchpad,
      id
  );
}

