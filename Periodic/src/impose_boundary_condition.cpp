#include <metal_stdlib>

using namespace metal;

#define UX(i, j) ux[(j) * (NX + 2) + (i)]
#define UY(i, j) uy[(j) * (NX + 2) + (i)]
#define TEMPERATURE(i, j) temperature[(j) * (NX + 2) + (i)]
#define P(i, j) p[(j) * (NX + 2) + (i)]

kernel void impose_boundary_condition_x(
    device float * const ux [[buffer(0)]],
    device float * const uy [[buffer(1)]],
    device float * const temperature [[buffer(2)]],
    device float * const p [[buffer(3)]],
    uint j [[thread_position_in_grid]]
) {
  if (NY + 1 < j) {
    return;
  }
  UX(0, j) = UX(NX, j);
  UX(NX + 1, j) = UX(1, j);
  UY(0, j) = UY(NX, j);
  UY(NX + 1, j) = UY(1, j);
  TEMPERATURE(0, j) = TEMPERATURE(NX, j);
  TEMPERATURE(NX + 1, j) = TEMPERATURE(1, j);
  P(0, j) = P(NX, j);
  P(NX + 1, j) = P(1, j);
}

kernel void impose_boundary_condition_y(
    device float * const ux [[buffer(0)]],
    device float * const uy [[buffer(1)]],
    device float * const temperature [[buffer(2)]],
    device float * const p [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
  if (NX + 1 < i) {
    return;
  }
  UX(i, 0) = 0.f;
  UX(i, NY + 1) = 0.f;
  UY(i, 0) = 0.f; // not in use
  UY(i, 1) = 0.f;
  UY(i, NY + 1) = 0.f;
  TEMPERATURE(i, 0) = 0.5f;
  TEMPERATURE(i, NY + 1) = - 0.5f;
  P(i, 0) = P(i, 1);
  P(i, NY + 1) = P(i, NY);
}

