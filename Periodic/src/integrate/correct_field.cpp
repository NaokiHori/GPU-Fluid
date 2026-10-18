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

inline void correct_ux(
    constant const float &runge_kutta_coefficient_implicit,
    const float dt,
    device const float * const scalar_potential,
    device float * const ux,
    uint2 id
) {
  const uint i = id.x;
  const uint j = id.y;
  if (i < 1 || NX < i || j < 1 || NY < j) {
    return;
  }
  const float scalar_potential_xm = 1 == i
    ? scalar_potential[(j - 1) * NX + (NX - 1)]
    : scalar_potential[(j - 1) * NX + (i - 2)];
  const float scalar_potential_xp = scalar_potential[(j - 1) * NX + (i - 1)];
  ux[j * (NX + 2) + i] -= runge_kutta_coefficient_implicit * dt * (
      - scalar_potential_xm
      + scalar_potential_xp
  ) / HX;
}

inline void correct_uy(
    constant const float &runge_kutta_coefficient_implicit,
    const float dt,
    device const float * const y_face_scale_factors,
    device const float * const scalar_potential,
    device float * const uy,
    uint2 id
) {
  const uint i = id.x;
  const uint j = id.y;
  if (i < 1 || NX < i || j < 2 || NY < j) {
    return;
  }
  const float scalar_potential_ym = scalar_potential[(j - 2) * NX + (i - 1)];
  const float scalar_potential_yp = scalar_potential[(j - 1) * NX + (i - 1)];
  uy[j * (NX + 2) + i] -= runge_kutta_coefficient_implicit * dt * (
      - scalar_potential_ym
      + scalar_potential_yp
  ) / y_face_scale_factors[j];
}

inline void correct_p(
    constant const float &runge_kutta_coefficient_implicit,
    constant const float &diffusivity,
    const float dt,
    device const metric_t * const y_center_metrics,
    device const float * const scalar_potential,
    device float * const p,
    uint2 id
) {
  uint i = id.x;
  uint j = id.y;
  if (i < 1 || NX < i || j < 1 || NY < j) {
    return;
  }
  const float scalar_potential_00 = scalar_potential[(j - 1) * NX + (i - 1)];
  const float scalar_potential_0m = 1 == j
    ? scalar_potential_00
    : scalar_potential[(j - 2) * NX + (i - 1)];
  const float scalar_potential_0p = NY == j
    ? scalar_potential_00
    : scalar_potential[(j    ) * NX + (i - 1)];
  p[j * (NX + 2) + i] +=
    + scalar_potential_00
    - 0.5f * diffusivity * runge_kutta_coefficient_implicit * dt * (
        + y_center_metrics[j].lower_diagonal * scalar_potential_0m
        + y_center_metrics[j].main_diagonal * scalar_potential_00
        + y_center_metrics[j].upper_diagonal * scalar_potential_0p
    );
}

kernel void correct_field(
    constant const float &runge_kutta_coefficient_implicit [[buffer(0)]],
    constant const float &diffusivity [[buffer(1)]],
    device const float * const dt [[buffer(2)]],
    device const float * const y_face_scale_factors [[buffer(3)]],
    device const metric_t * const y_center_metrics [[buffer(4)]],
    device const float * const scalar_potential [[buffer(5)]],
    device float * const ux [[buffer(6)]],
    device float * const uy [[buffer(7)]],
    device float * const p [[buffer(8)]],
    uint2 id [[thread_position_in_grid]]
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
      y_face_scale_factors,
      scalar_potential,
      uy,
      id
  );
  correct_p(
      runge_kutta_coefficient_implicit,
      diffusivity,
      *dt,
      y_center_metrics,
      scalar_potential,
      p,
      id
  );
}

