#include <metal_stdlib>

using namespace metal;

kernel void compute_right_hand_side(
    constant const float &runge_kutta_coefficient_implicit [[buffer(0)]],
    device const float * const dt [[buffer(1)]],
    device const float * const x_center_scale_factors [[buffer(2)]],
    device const float * const y_center_scale_factors [[buffer(3)]],
    device const float * const ux [[buffer(4)]],
    device const float * const uy [[buffer(5)]],
    device float * const out [[buffer(6)]],
    uint2 id [[thread_position_in_grid]]
) {
  uint i = id.x;
  uint j = id.y;
  if (i < 1 || NX < i || j < 1 || NY < j) {
    return;
  }
  const float prefactor = 1.f / runge_kutta_coefficient_implicit / *dt;
  const float x_grad = (
      - ux[(j    ) * (NX + 2) + (i    )]
      + ux[(j    ) * (NX + 2) + (i + 1)]
  ) / x_center_scale_factors[i];
  const float y_grad = (
      - uy[(j    ) * (NX + 2) + (i    )]
      + uy[(j + 1) * (NX + 2) + (i    )]
  ) / y_center_scale_factors[j];
  out[(j - 1) * NX + (i - 1)] = prefactor * (x_grad + y_grad);
}

