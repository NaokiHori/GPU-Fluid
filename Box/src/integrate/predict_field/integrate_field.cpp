#include <metal_stdlib>

using namespace metal;

#define UX(i, j) ux[(j) * (NX + 2) + (i)]
#define UY(i, j) uy[(j) * (NX + 2) + (i)]
#define P(i, j) p[(j) * (NX + 2) + (i)]
#define TEMPERATURE(i, j) temperature[(j) * (NX + 2) + (i)]

#define ASSERT_STRUCT_SIZE(s, size) \
  static_assert(sizeof(s) == size, "unexpected struct size")
#define ASSERT_STRUCT_OFFSET(s, prop, offset) \
  static_assert(__builtin_offsetof(s, prop) == offset, "unexpected offset of struct member")

// TODO: use shared definition between host and device
typedef struct {
  float explicit_previous;
  float explicit_current;
  float implicit;
} runge_kutta_coefficient_t;
ASSERT_STRUCT_SIZE(
    runge_kutta_coefficient_t,
    SIZE_OF_RUNGE_KUTTA_COEFFICIENT_T
);
ASSERT_STRUCT_OFFSET(
    runge_kutta_coefficient_t,
    explicit_previous,
    OFFSET_OF_RUNGE_KUTTA_COEFFICIENT_T_EXPLICIT_PREVIOUS
);
ASSERT_STRUCT_OFFSET(
    runge_kutta_coefficient_t,
    explicit_current,
    OFFSET_OF_RUNGE_KUTTA_COEFFICIENT_T_EXPLICIT_CURRENT
);
ASSERT_STRUCT_OFFSET(
    runge_kutta_coefficient_t,
    implicit,
    OFFSET_OF_RUNGE_KUTTA_COEFFICIENT_T_IMPLICIT
);

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

inline void integrate_ux(
    constant const runge_kutta_coefficient_t &runge_kutta_coefficient,
    constant const float &dt,
    constant const float &diffusivity,
    device const metric_t * const x_center_metrics,
    device const metric_t * const y_center_metrics,
    device const metric_t * const x_face_metrics,
    device const float * const ux,
    device const float * const uy,
    device const float * const p,
    device float * const dux,
    device float * const dux_explicit_previous,
    uint2 id
) {
  const uint32_t i = id.x;
  const uint32_t j = id.y;
  if (i < 2 || NX < i || j < 1 || NY < j) {
    return;
  }
  const metric_t x_center_metric_xm = x_center_metrics[i - 1];
  const metric_t x_center_metric_xp = x_center_metrics[i    ];
  const metric_t y_center_metric = y_center_metrics[j];
  const metric_t x_face_metric = x_face_metrics[i];
  const float hx_c = x_face_metric.scale_factor;
  const float hx_m = x_center_metric_xm.scale_factor;
  const float hx_p = x_center_metric_xp.scale_factor;
  const float hy_c = y_center_metric.scale_factor;
  const float j_cc = hx_c * hy_c;
  const float ux_xm = + 0.5f * hy_c * UX(i - 1, j    )
                      + 0.5f * hy_c * UX(i    , j    );
  const float ux_xp = + 0.5f * hy_c * UX(i    , j    )
                      + 0.5f * hy_c * UX(i + 1, j    );
  const float uy_ym = + 0.5f * hx_m * UY(i - 1, j    )
                      + 0.5f * hx_p * UY(i    , j    );
  const float uy_yp = + 0.5f * hx_m * UY(i - 1, j + 1)
                      + 0.5f * hx_p * UY(i    , j + 1);
  const float dux_xm = - UX(i - 1, j    ) + UX(i    , j    );
  const float dux_xp = - UX(i    , j    ) + UX(i + 1, j    );
  const float dux_ym = - UX(i    , j - 1) + UX(i    , j    );
  const float dux_yp = - UX(i    , j    ) + UX(i    , j + 1);
  const float advection_x = - 1.f / j_cc * (
      + 0.5f * ux_xm * dux_xm
      + 0.5f * ux_xp * dux_xp
  );
  const float advection_y = - 1.f / j_cc * (
      + 0.5f * uy_ym * dux_ym
      + 0.5f * uy_yp * dux_yp
  );
  const float diffusion_x = diffusivity * (
      + x_face_metric.lower_diagonal * UX(i - 1, j    )
      + x_face_metric.main_diagonal  * UX(i    , j    )
      + x_face_metric.upper_diagonal * UX(i + 1, j    )
  );
  const float diffusion_y = diffusivity * (
      + y_center_metric.lower_diagonal * UX(i    , j - 1)
      + y_center_metric.main_diagonal  * UX(i    , j    )
      + y_center_metric.upper_diagonal * UX(i    , j + 1)
  );
  const float pressure = - 1.f / hx_c * (
      - P(i - 1, j    )
      + P(i    , j    )
  );
  const uint32_t index = (j - 1) * (NX - 1) + (i - 2);
  dux[index] = dt * (
      + runge_kutta_coefficient.explicit_current * (
          + advection_x
          + advection_y
      )
      + runge_kutta_coefficient.explicit_previous * dux_explicit_previous[index]
      + runge_kutta_coefficient.implicit * (
          + pressure
          + diffusion_x
          + diffusion_y
      )
  );
  dux_explicit_previous[index] = advection_x + advection_y;
}

inline void integrate_uy(
    constant const runge_kutta_coefficient_t &runge_kutta_coefficient,
    constant const float &dt,
    constant const float &diffusivity,
    device const metric_t * const x_center_metrics,
    device const metric_t * const y_center_metrics,
    device const metric_t * const y_face_metrics,
    device const float * const ux,
    device const float * const uy,
    device const float * const p,
    device const float * const temperature,
    device float * const duy,
    device float * const duy_explicit_previous,
    uint2 id
) {
  const uint32_t i = id.x;
  const uint32_t j = id.y;
  if (i < 1 || NX < i || j < 2 || NY < j) {
    return;
  }
  const metric_t x_center_metric = x_center_metrics[i];
  const metric_t y_center_metric_ym = y_center_metrics[j - 1];
  const metric_t y_center_metric_yp = y_center_metrics[j    ];
  const metric_t y_face_metric = y_face_metrics[j];
  const float hx_c = x_center_metric.scale_factor;
  const float hy_c = y_face_metric.scale_factor;
  const float hy_m = y_center_metric_ym.scale_factor;
  const float hy_p = y_center_metric_yp.scale_factor;
  const float j_cc = hx_c * hy_c;
  const float ux_xm = + 0.5f * hy_m * UX(i    , j - 1)
                      + 0.5f * hy_p * UX(i    , j    );
  const float ux_xp = + 0.5f * hy_m * UX(i + 1, j - 1)
                      + 0.5f * hy_p * UX(i + 1, j    );
  const float uy_ym = + 0.5f * hx_c * UY(i    , j - 1)
                      + 0.5f * hx_c * UY(i    , j    );
  const float uy_yp = + 0.5f * hx_c * UY(i    , j    )
                      + 0.5f * hx_c * UY(i    , j + 1);
  const float duy_xm = - UY(i - 1, j    ) + UY(i    , j    );
  const float duy_xp = - UY(i    , j    ) + UY(i + 1, j    );
  const float duy_ym = - UY(i    , j - 1) + UY(i    , j    );
  const float duy_yp = - UY(i    , j    ) + UY(i    , j + 1);
  const float advection_x = - 1.f / j_cc * (
      + 0.5f * ux_xm * duy_xm
      + 0.5f * ux_xp * duy_xp
  );
  const float advection_y = - 1.f / j_cc * (
      + 0.5f * uy_ym * duy_ym
      + 0.5f * uy_yp * duy_yp
  );
  const float pressure = - 1.f / hy_c * (
      - P(i    , j - 1)
      + P(i    , j    )
  );
  const float diffusion_x = diffusivity * (
      + x_center_metric.lower_diagonal * UY(i - 1, j    )
      + x_center_metric.main_diagonal  * UY(i    , j    )
      + x_center_metric.upper_diagonal * UY(i + 1, j    )
  );
  const float diffusion_y = diffusivity * (
      + y_face_metric.lower_diagonal * UY(i    , j - 1)
      + y_face_metric.main_diagonal  * UY(i    , j    )
      + y_face_metric.upper_diagonal * UY(i    , j + 1)
  );
  const float buoyancy =
    + 0.5f * TEMPERATURE(i    , j - 1)
    + 0.5f * TEMPERATURE(i    , j    );
  const uint32_t index = (j - 2) * NX + (i - 1);
  duy[index] = dt * (
      + runge_kutta_coefficient.explicit_current * (
          + advection_x
          + advection_y
          + buoyancy
      )
      + runge_kutta_coefficient.explicit_previous * duy_explicit_previous[index]
      + runge_kutta_coefficient.implicit * (
          + pressure
          + diffusion_x
          + diffusion_y
      )
  );
  duy_explicit_previous[index] = advection_x + advection_y + buoyancy;
}

inline void integrate_temperature(
    constant const runge_kutta_coefficient_t &runge_kutta_coefficient,
    constant const float &dt,
    constant const float &diffusivity,
    device const metric_t * const x_center_metrics,
    device const metric_t * const y_center_metrics,
    device const float * const ux,
    device const float * const uy,
    device const float * const temperature,
    device float * const dtemperature,
    device float * const dtemperature_explicit_previous,
    uint2 id
) {
  const uint32_t i = id.x;
  const uint32_t j = id.y;
  if (i < 1 || NX < i || j < 1 || NY < j) {
    return;
  }
  const metric_t x_center_metric = x_center_metrics[i];
  const metric_t y_center_metric = y_center_metrics[j];
  const float hx_c = x_center_metric.scale_factor;
  const float hy_c = y_center_metric.scale_factor;
  const float j_cc = hx_c * hy_c;
  const float ux_xm = hy_c * UX(i    , j    );
  const float ux_xp = hy_c * UX(i + 1, j    );
  const float uy_ym = hx_c * UY(i    , j    );
  const float uy_yp = hx_c * UY(i    , j + 1);
  const float dtemperature_xm = - TEMPERATURE(i - 1, j    ) + TEMPERATURE(i    , j    );
  const float dtemperature_xp = - TEMPERATURE(i    , j    ) + TEMPERATURE(i + 1, j    );
  const float dtemperature_ym = - TEMPERATURE(i    , j - 1) + TEMPERATURE(i    , j    );
  const float dtemperature_yp = - TEMPERATURE(i    , j    ) + TEMPERATURE(i    , j + 1);
  const float advection_x = - 1.f / j_cc * (
      + 0.5f * ux_xm * dtemperature_xm
      + 0.5f * ux_xp * dtemperature_xp
  );
  const float advection_y = - 1.f / j_cc * (
      + 0.5f * uy_ym * dtemperature_ym
      + 0.5f * uy_yp * dtemperature_yp
  );
  const float diffusion_x = diffusivity * (
      + x_center_metric.lower_diagonal * TEMPERATURE(i - 1, j    )
      + x_center_metric.main_diagonal * TEMPERATURE(i    , j    )
      + x_center_metric.upper_diagonal * TEMPERATURE(i + 1, j    )
  );
  const float diffusion_y = diffusivity * (
      + y_center_metric.lower_diagonal * TEMPERATURE(i    , j - 1)
      + y_center_metric.main_diagonal * TEMPERATURE(i    , j    )
      + y_center_metric.upper_diagonal * TEMPERATURE(i    , j + 1)
  );
  const uint32_t index = (j - 1) * NX + (i - 1);
  dtemperature[index] = dt * (
      + runge_kutta_coefficient.explicit_current * (
          + advection_x
          + advection_y
      )
      + runge_kutta_coefficient.explicit_previous * dtemperature_explicit_previous[index]
      + runge_kutta_coefficient.implicit * (
          + diffusion_x
          + diffusion_y
      )
  );
  dtemperature_explicit_previous[index] = advection_x + advection_y;
}

kernel void integrate_field(
    constant const runge_kutta_coefficient_t &runge_kutta_coefficient [[buffer(0)]],
    constant const float &dt [[buffer(1)]],
    constant const float &diffusivity [[buffer(2)]],
    device const metric_t * const x_center_metrics [[buffer(3)]],
    device const metric_t * const y_center_metrics [[buffer(4)]],
    device const metric_t * const x_face_metrics [[buffer(5)]],
    device const metric_t * const y_face_metrics [[buffer(6)]],
    device const float * const ux [[buffer(7)]],
    device const float * const uy [[buffer(8)]],
    device const float * const p [[buffer(9)]],
    device const float * const temperature [[buffer(10)]],
    device float * const dux [[buffer(11)]],
    device float * const duy [[buffer(12)]],
    device float * const dtemperature [[buffer(13)]],
    device float * const dux_explicit_previous [[buffer(14)]],
    device float * const duy_explicit_previous [[buffer(15)]],
    device float * const dtemperature_explicit_previous [[buffer(16)]],
    uint2 id [[thread_position_in_grid]]
) {
  integrate_ux(
      runge_kutta_coefficient,
      dt,
      diffusivity,
      x_center_metrics,
      y_center_metrics,
      x_face_metrics,
      ux,
      uy,
      p,
      dux,
      dux_explicit_previous,
      id
  );
  integrate_uy(
      runge_kutta_coefficient,
      dt,
      diffusivity,
      x_center_metrics,
      y_center_metrics,
      y_face_metrics,
      ux,
      uy,
      p,
      temperature,
      duy,
      duy_explicit_previous,
      id
  );
  integrate_temperature(
      runge_kutta_coefficient,
      dt,
      diffusivity,
      x_center_metrics,
      y_center_metrics,
      ux,
      uy,
      temperature,
      dtemperature,
      dtemperature_explicit_previous,
      id
  );
}

