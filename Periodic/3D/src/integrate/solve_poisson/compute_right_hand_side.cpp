#include <metal_stdlib>

using namespace metal;

#define UX(i, j, k) ux[((k) * (NY + 2) + (j)) * (NX + 2) + (i)]
#define UY(i, j, k) uy[((k) * (NY + 2) + (j)) * (NX + 2) + (i)]
#define UZ(i, j, k) uz[((k) * (NY + 2) + (j)) * (NX + 2) + (i)]

kernel void compute_right_hand_side(
    constant const float &runge_kutta_coefficient_implicit [[buffer(0)]],
    device const float * const dt [[buffer(1)]],
    device const float * const z_center_scale_factors [[buffer(2)]],
    device const float * const ux [[buffer(3)]],
    device const float * const uy [[buffer(4)]],
    device const float * const uz [[buffer(5)]],
    device float * const out [[buffer(6)]],
    uint3 id [[thread_position_in_grid]]
) {
  const uint i = id.x + 1;
  const uint j = id.y + 1;
  const uint k = id.z + 1;
  if (NX < i || NY < j || NZ < k) {
    return;
  }
  const float prefactor = 1.f / runge_kutta_coefficient_implicit / *dt;
  const float x_grad = (
      - UX(i    , j    , k    )
      + UX(i + 1, j    , k    )
  ) / HX;
  const float y_grad = (
      - UY(i    , j    , k    )
      + UY(i    , j + 1, k    )
  ) / HY;
  const float z_grad = (
      - UZ(i    , j    , k    )
      + UZ(i    , j    , k + 1)
  ) / z_center_scale_factors[k];
  out[((k - 1) * NY + (j - 1)) * NX + (i - 1)] = prefactor * (x_grad + y_grad + z_grad);
}

