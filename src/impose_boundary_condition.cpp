#include <metal_stdlib>

using namespace metal;

#define UX(i, j) ux[(j) * (NX + 2) + (i)]
#define UY(i, j) uy[(j) * (NX + 2) + (i)]
#define TEMPERATURE(i, j) temperature[(j) * (NX + 2) + (i)]

kernel void impose_boundary_condition_x(
    device float * const ux [[buffer(0)]],
    device float * const uy [[buffer(1)]],
    device float * const temperature [[buffer(2)]],
    uint j [[thread_position_in_grid]]
) {
  if (NY + 1 < j) {
    return;
  }
  UX(0, j) = 0.f; // not in use
  UX(1, j) = 0.f;
  UX(NX + 1, j) = 0.f;
  UY(0, j) = 0.f;
  UY(NX + 1, j) = 0.f;
  TEMPERATURE(0, j) = TEMPERATURE(1, j);
  TEMPERATURE(NX + 1, j) = TEMPERATURE(NX, j);
}

kernel void impose_boundary_condition_y(
    device float * const ux [[buffer(0)]],
    device float * const uy [[buffer(1)]],
    device float * const temperature [[buffer(2)]],
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
}

