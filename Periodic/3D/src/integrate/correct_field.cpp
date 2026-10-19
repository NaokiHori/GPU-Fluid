#include <metal_stdlib>

using namespace metal;

#define ASSERT_STRUCT_SIZE(s, size) \
  static_assert(sizeof(s) == size, "unexpected struct size")
#define ASSERT_STRUCT_OFFSET(s, prop, offset) \
  static_assert(__builtin_offsetof(s, prop) == offset, "unexpected offset of struct member")

// TODO: use shared definition between host and device
typedef struct metric {
  float scale_factor;
  float lower_diagonal;
  float main_diagonal;
  float upper_diagonal;
} metric_t;
ASSERT_STRUCT_SIZE(metric_t, SIZE_OF_METRIC_T);
ASSERT_STRUCT_OFFSET(metric_t, scale_factor, OFFSET_OF_METRIC_T_SCALE_FACTOR);
ASSERT_STRUCT_OFFSET(metric_t, lower_diagonal, OFFSET_OF_METRIC_T_LOWER_DIAGONAL);
ASSERT_STRUCT_OFFSET(metric_t, main_diagonal, OFFSET_OF_METRIC_T_MAIN_DIAGONAL);
ASSERT_STRUCT_OFFSET(metric_t, upper_diagonal, OFFSET_OF_METRIC_T_UPPER_DIAGONAL);

#define UX(i, j, k) ux[((k) * (NY + 2) + (j)) * (NX + 2) + (i)]
#define UY(i, j, k) uy[((k) * (NY + 2) + (j)) * (NX + 2) + (i)]
#define UZ(i, j, k) uz[((k) * (NY + 2) + (j)) * (NX + 2) + (i)]
#define P(i, j, k) p[((k) * (NY + 2) + (j)) * (NX + 2) + (i)]
// no halo
#define SCALAR_POTENTIAL(i, j, k) scalar_potential[((k - 1) * (NY) + (j - 1)) * (NX) + (i - 1)]

inline void correct_ux(
    constant const float &runge_kutta_coefficient_implicit,
    const float dt,
    device const float * const scalar_potential,
    device float * const ux,
    uint3 id
) {
  const uint i = id.x + 1;
  const uint j = id.y + 1;
  const uint k = id.z + 1;
  if (NX < i || NY < j || NZ < k) {
    return;
  }
  const float scalar_potential_xm = 1 == i
    ? SCALAR_POTENTIAL(NX, j, k)
    : SCALAR_POTENTIAL(i - 1, j, k);
  const float scalar_potential_xp = SCALAR_POTENTIAL(i, j, k);
  UX(i, j, k) -= runge_kutta_coefficient_implicit * dt * (
      - scalar_potential_xm
      + scalar_potential_xp
  ) / HX;
}

inline void correct_uy(
    constant const float &runge_kutta_coefficient_implicit,
    const float dt,
    device const float * const scalar_potential,
    device float * const uy,
    uint3 id
) {
  const uint i = id.x + 1;
  const uint j = id.y + 1;
  const uint k = id.z + 1;
  if (NX < i || NY < j || NZ < k) {
    return;
  }
  const float scalar_potential_ym = 1 == j
    ? SCALAR_POTENTIAL(i, NY, k)
    : SCALAR_POTENTIAL(i, j - 1, k);
  const float scalar_potential_yp = SCALAR_POTENTIAL(i, j, k);
  UY(i, j, k) -= runge_kutta_coefficient_implicit * dt * (
      - scalar_potential_ym
      + scalar_potential_yp
  ) / HY;
}

inline void correct_uz(
    constant const float &runge_kutta_coefficient_implicit,
    const float dt,
    device const float * const z_face_scale_factors,
    device const float * const scalar_potential,
    device float * const uz,
    uint3 id
) {
  const uint i = id.x + 1;
  const uint j = id.y + 1;
  const uint k = id.z + 1;
  if (NX < i || NY < j || k < 2 || NZ < k) {
    return;
  }
  const float scalar_potential_zm = SCALAR_POTENTIAL(i, j, k - 1);
  const float scalar_potential_zp = SCALAR_POTENTIAL(i, j, k    );
  UZ(i, j, k) -= runge_kutta_coefficient_implicit * dt * (
      - scalar_potential_zm
      + scalar_potential_zp
  ) / z_face_scale_factors[k];
}

inline void correct_p(
    constant const float &runge_kutta_coefficient_implicit,
    constant const float &diffusivity,
    const float dt,
    device const metric_t * const z_center_metrics,
    device const float * const scalar_potential,
    device float * const p,
    uint3 id
) {
  const uint i = id.x + 1;
  const uint j = id.y + 1;
  const uint k = id.z + 1;
  if (NX < i || NY < j || NZ < k) {
    return;
  }
  const float scalar_potential_0 = SCALAR_POTENTIAL(i, j, k);
  const float scalar_potential_m = 1 == k
    ? scalar_potential_0
    : SCALAR_POTENTIAL(i, j, k - 1);
  const float scalar_potential_p = NZ == k
    ? scalar_potential_0
    : SCALAR_POTENTIAL(i, j, k + 1);
  P(i, j, k) +=
    + scalar_potential_0
    - 0.5f * diffusivity * runge_kutta_coefficient_implicit * dt * (
        + z_center_metrics[k].lower_diagonal * scalar_potential_m
        + z_center_metrics[k].main_diagonal  * scalar_potential_0
        + z_center_metrics[k].upper_diagonal * scalar_potential_p
    );
}

kernel void correct_field(
    constant const float &runge_kutta_coefficient_implicit [[buffer(0)]],
    constant const float &diffusivity [[buffer(1)]],
    device const float * const dt [[buffer(2)]],
    device const float * const z_face_scale_factors [[buffer(3)]],
    device const metric_t * const z_center_metrics [[buffer(4)]],
    device const float * const scalar_potential [[buffer(5)]],
    device float * const ux [[buffer(6)]],
    device float * const uy [[buffer(7)]],
    device float * const uz [[buffer(8)]],
    device float * const p [[buffer(9)]],
    uint3 id [[thread_position_in_grid]]
) {
  correct_ux(
      runge_kutta_coefficient_implicit,
      *dt,
      scalar_potential,
      ux,
      id
  );
  correct_uy(
      runge_kutta_coefficient_implicit,
      *dt,
      scalar_potential,
      uy,
      id
  );
  correct_uz(
      runge_kutta_coefficient_implicit,
      *dt,
      z_face_scale_factors,
      scalar_potential,
      uz,
      id
  );
  correct_p(
      runge_kutta_coefficient_implicit,
      diffusivity,
      *dt,
      z_center_metrics,
      scalar_potential,
      p,
      id
  );
}

