#include <metal_stdlib>

using namespace metal;

#define UX(i, j, k) ux[((k) * (NY + 2) + (j)) * (NX + 2) + (i)]
#define UY(i, j, k) uy[((k) * (NY + 2) + (j)) * (NX + 2) + (i)]
#define UZ(i, j, k) uz[((k) * (NY + 2) + (j)) * (NX + 2) + (i)]
#define TEMPERATURE(i, j, k) temperature[((k) * (NY + 2) + (j)) * (NX + 2) + (i)]
#define P(i, j, k) p[((k) * (NY + 2) + (j)) * (NX + 2) + (i)]

kernel void impose_boundary_condition_x(
    device float * const ux [[buffer(0)]],
    device float * const uy [[buffer(1)]],
    device float * const uz [[buffer(2)]],
    device float * const temperature [[buffer(3)]],
    device float * const p [[buffer(4)]],
    uint3 id [[thread_position_in_grid]]
) {
  const uint32_t j = id.y;
  const uint32_t k = id.z;
  if (NY + 1 < j || NZ + 1 < k) {
    return;
  }
  UX(     0, j, k) = UX(NX, j, k);
  UX(NX + 1, j, k) = UX( 1, j, k);
  UY(     0, j, k) = UY(NX, j, k);
  UY(NX + 1, j, k) = UY( 1, j, k);
  UZ(     0, j, k) = UZ(NX, j, k);
  UZ(NX + 1, j, k) = UZ( 1, j, k);
  TEMPERATURE(     0, j, k) = TEMPERATURE(NX, j, k);
  TEMPERATURE(NX + 1, j, k) = TEMPERATURE( 1, j, k);
  P(     0, j, k) = P(NX, j, k);
  P(NX + 1, j, k) = P( 1, j, k);
}

kernel void impose_boundary_condition_y(
    device float * const ux [[buffer(0)]],
    device float * const uy [[buffer(1)]],
    device float * const uz [[buffer(2)]],
    device float * const temperature [[buffer(3)]],
    device float * const p [[buffer(4)]],
    uint3 id [[thread_position_in_grid]]
) {
  const uint32_t i = id.x;
  const uint32_t k = id.z;
  if (NX + 1 < i || NZ + 1 < k) {
    return;
  }
  UX(i,      0, k) = UX(i, NY, k);
  UX(i, NY + 1, k) = UX(i,  1, k);
  UY(i,      0, k) = UY(i, NY, k);
  UY(i, NY + 1, k) = UY(i,  1, k);
  UZ(i,      0, k) = UZ(i, NY, k);
  UZ(i, NY + 1, k) = UZ(i,  1, k);
  TEMPERATURE(i,      0, k) = TEMPERATURE(i, NY, k);
  TEMPERATURE(i, NY + 1, k) = TEMPERATURE(i,  1, k);
  P(i,      0, k) = P(i, NY, k);
  P(i, NY + 1, k) = P(i,  1, k);
}

kernel void impose_boundary_condition_z(
    device float * const ux [[buffer(0)]],
    device float * const uy [[buffer(1)]],
    device float * const uz [[buffer(2)]],
    device float * const temperature [[buffer(3)]],
    device float * const p [[buffer(4)]],
    uint3 id [[thread_position_in_grid]]
) {
  const uint32_t i = id.x;
  const uint32_t j = id.y;
  if (NX + 1 < i || NY + 1 < j) {
    return;
  }
  UX(i, j,      0) = 0.f;
  UX(i, j, NZ + 1) = 0.f;
  UY(i, j,      0) = 0.f;
  UY(i, j, NZ + 1) = 0.f;
  UZ(i, j,      0) = 0.f; // not in use
  UZ(i, j,      1) = 0.f;
  UZ(i, j, NZ + 1) = 0.f;
  TEMPERATURE(i, j,      0) = 0.5f;
  TEMPERATURE(i, j, NZ + 1) = - 0.5f;
  P(i, j,      0) = P(i, j,  1);
  P(i, j, NZ + 1) = P(i, j, NZ);
}

