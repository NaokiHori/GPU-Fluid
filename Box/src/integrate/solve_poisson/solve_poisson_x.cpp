#include <metal_stdlib>

using namespace metal;

// apply linear transform "a" of size Nx against "b"
kernel void solve_poisson_x(
    // Nx rows, Nx columns
    device const float * const a [[buffer(0)]],
    // Ny rows, Nx columns
    device const float * const b [[buffer(1)]],
    // Ny rows, Nx columns
    device float * const c [[buffer(2)]],
    uint2 id [[thread_position_in_grid]]
) {
  // i: row index, j: column index
  uint i = id.x;
  uint j = id.y;
  if (NX <= i || NY <= j) {
    return;
  }
  float sum = 0.0f;
  for (uint k = 0; k < NX; k++) {
    sum += a[i * NX + k] * b[j * NX + k];
  }
  c[j * NX + i] = sum;
}

