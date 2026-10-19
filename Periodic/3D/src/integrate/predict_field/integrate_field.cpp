#include <metal_stdlib>

using namespace metal;

#define UX(i, j, k) ux[((k) * (NY + 2) + (j)) * (NX + 2) + (i)]
#define UY(i, j, k) uy[((k) * (NY + 2) + (j)) * (NX + 2) + (i)]
#define UZ(i, j, k) uz[((k) * (NY + 2) + (j)) * (NX + 2) + (i)]
#define P(i, j, k) p[((k) * (NY + 2) + (j)) * (NX + 2) + (i)]
#define TEMPERATURE(i, j, k) temperature[((k) * (NY + 2) + (j)) * (NX + 2) + (i)]

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
    device const metric_t * const z_center_metrics,
    device const float * const ux,
    device const float * const uy,
    device const float * const uz,
    device const float * const p,
    device float * const dux,
    device float * const dux_explicit_previous,
    uint3 id
) {
  const uint32_t i = id.x + 1;
  const uint32_t j = id.y + 1;
  const uint32_t k = id.z + 1;
  if (NX < i || NY < j || NZ < k) {
    return;
  }
  const metric_t z_center_metric = z_center_metrics[k];
  const float hz_c = z_center_metric.scale_factor;
  const float j_cc = HX * HY * hz_c;
  const float ux_xm = + 0.5f * HY * hz_c * UX(i - 1, j    , k    )
                      + 0.5f * HY * hz_c * UX(i    , j    , k    );
  const float ux_xp = + 0.5f * HY * hz_c * UX(i    , j    , k    )
                      + 0.5f * HY * hz_c * UX(i + 1, j    , k    );
  const float uy_ym = + 0.5f * HX * hz_c * UY(i - 1, j    , k    )
                      + 0.5f * HX * hz_c * UY(i    , j    , k    );
  const float uy_yp = + 0.5f * HX * hz_c * UY(i - 1, j + 1, k    )
                      + 0.5f * HX * hz_c * UY(i    , j + 1, k    );
  const float uz_zm = + 0.5f * HX * HY * UZ(i - 1, j    , k    )
                      + 0.5f * HX * HY * UZ(i    , j    , k    );
  const float uz_zp = + 0.5f * HX * HY * UZ(i - 1, j    , k + 1)
                      + 0.5f * HX * HY * UZ(i    , j    , k + 1);
  const float dux_xm = - UX(i - 1, j    , k    ) + UX(i    , j    , k    );
  const float dux_xp = - UX(i    , j    , k    ) + UX(i + 1, j    , k    );
  const float dux_ym = - UX(i    , j - 1, k    ) + UX(i    , j    , k    );
  const float dux_yp = - UX(i    , j    , k    ) + UX(i    , j + 1, k    );
  const float dux_zm = - UX(i    , j    , k - 1) + UX(i    , j    , k    );
  const float dux_zp = - UX(i    , j    , k    ) + UX(i    , j    , k + 1);
  const float advection_x = - 1.f / j_cc * (
      + 0.5f * ux_xm * dux_xm
      + 0.5f * ux_xp * dux_xp
  );
  const float advection_y = - 1.f / j_cc * (
      + 0.5f * uy_ym * dux_ym
      + 0.5f * uy_yp * dux_yp
  );
  const float advection_z = - 1.f / j_cc * (
      + 0.5f * uz_zm * dux_zm
      + 0.5f * uz_zp * dux_zp
  );
  const float diffusion_x = diffusivity * (
      + (1.f / HX / HX) * UX(i - 1, j    , k    )
      - (2.f / HX / HX) * UX(i    , j    , k    )
      + (1.f / HX / HX) * UX(i + 1, j    , k    )
  );
  const float diffusion_y = diffusivity * (
      + (1.f / HY / HY) * UX(i    , j - 1, k    )
      - (2.f / HY / HY) * UX(i    , j    , k    )
      + (1.f / HY / HY) * UX(i    , j + 1, k    )
  );
  const float diffusion_z = diffusivity * (
      + z_center_metric.lower_diagonal * UX(i    , j    , k - 1)
      + z_center_metric.main_diagonal  * UX(i    , j    , k    )
      + z_center_metric.upper_diagonal * UX(i    , j    , k + 1)
  );
  const float pressure = - 1.f / HX * (
      - P(i - 1, j    , k    )
      + P(i    , j    , k    )
  );
  const uint32_t index = ((k - 1) * NY + (j - 1)) * NX + (i - 1);
  dux[index] = dt * (
      + runge_kutta_coefficient.explicit_current * (
          + advection_x
          + advection_y
          + advection_z
          + diffusion_x
          + diffusion_y
      )
      + runge_kutta_coefficient.explicit_previous * dux_explicit_previous[index]
      + runge_kutta_coefficient.implicit * (
          + pressure
          + diffusion_z
      )
  );
  dux_explicit_previous[index] =
      + advection_x
      + advection_y
      + advection_z
      + diffusion_x
      + diffusion_y;
}

inline void integrate_uy(
    constant const runge_kutta_coefficient_t &runge_kutta_coefficient,
    constant const float &dt,
    constant const float &diffusivity,
    device const metric_t * const z_center_metrics,
    device const float * const ux,
    device const float * const uy,
    device const float * const uz,
    device const float * const p,
    device float * const duy,
    device float * const duy_explicit_previous,
    uint3 id
) {
  const uint32_t i = id.x + 1;
  const uint32_t j = id.y + 1;
  const uint32_t k = id.z + 1;
  if (NX < i || NY < j || NZ < k) {
    return;
  }
  const metric_t z_center_metric = z_center_metrics[k];
  const float hz_c = z_center_metric.scale_factor;
  const float j_cc = HX * HY * hz_c;
  const float ux_xm = + 0.5f * HY * hz_c * UX(i    , j - 1, k    )
                      + 0.5f * HY * hz_c * UX(i    , j    , k    );
  const float ux_xp = + 0.5f * HY * hz_c * UX(i + 1, j - 1, k    )
                      + 0.5f * HY * hz_c * UX(i + 1, j    , k    );
  const float uy_ym = + 0.5f * HX * hz_c * UY(i    , j - 1, k    )
                      + 0.5f * HX * hz_c * UY(i    , j    , k    );
  const float uy_yp = + 0.5f * HX * hz_c * UY(i    , j    , k    )
                      + 0.5f * HX * hz_c * UY(i    , j + 1, k    );
  const float uz_zm = + 0.5f * HX * HY * UZ(i    , j - 1, k    )
                      + 0.5f * HX * HY * UZ(i    , j    , k    );
  const float uz_zp = + 0.5f * HX * HY * UZ(i    , j - 1, k + 1)
                      + 0.5f * HX * HY * UZ(i    , j    , k + 1);
  const float duy_xm = - UY(i - 1, j    , k    ) + UY(i    , j    , k    );
  const float duy_xp = - UY(i    , j    , k    ) + UY(i + 1, j    , k    );
  const float duy_ym = - UY(i    , j - 1, k    ) + UY(i    , j    , k    );
  const float duy_yp = - UY(i    , j    , k    ) + UY(i    , j + 1, k    );
  const float duy_zm = - UY(i    , j    , k - 1) + UY(i    , j    , k    );
  const float duy_zp = - UY(i    , j    , k    ) + UY(i    , j    , k + 1);
  const float advection_x = - 1.f / j_cc * (
      + 0.5f * ux_xm * duy_xm
      + 0.5f * ux_xp * duy_xp
  );
  const float advection_y = - 1.f / j_cc * (
      + 0.5f * uy_ym * duy_ym
      + 0.5f * uy_yp * duy_yp
  );
  const float advection_z = - 1.f / j_cc * (
      + 0.5f * uz_zm * duy_zm
      + 0.5f * uz_zp * duy_zp
  );
  const float pressure = - 1.f / HY * (
      - P(i    , j - 1, k    )
      + P(i    , j    , k    )
  );
  const float diffusion_x = diffusivity * (
      + (1.f / HX / HX) * UY(i - 1, j    , k    )
      - (2.f / HX / HX) * UY(i    , j    , k    )
      + (1.f / HX / HX) * UY(i + 1, j    , k    )
  );
  const float diffusion_y = diffusivity * (
      + (1.f / HY / HY) * UY(i    , j - 1, k    )
      - (2.f / HY / HY) * UY(i    , j    , k    )
      + (1.f / HY / HY) * UY(i    , j + 1, k    )
  );
  const float diffusion_z = diffusivity * (
      + z_center_metric.lower_diagonal * UY(i    , j    , k - 1)
      + z_center_metric.main_diagonal  * UY(i    , j    , k    )
      + z_center_metric.upper_diagonal * UY(i    , j    , k + 1)
  );
  const uint32_t index = ((k - 1) * NY + (j - 1)) * NX + (i - 1);
  duy[index] = dt * (
      + runge_kutta_coefficient.explicit_current * (
          + advection_x
          + advection_y
          + advection_z
          + diffusion_x
          + diffusion_y
      )
      + runge_kutta_coefficient.explicit_previous * duy_explicit_previous[index]
      + runge_kutta_coefficient.implicit * (
          + pressure
          + diffusion_z
      )
  );
  duy_explicit_previous[index] =
      + advection_x
      + advection_y
      + advection_z
      + diffusion_x
      + diffusion_y;
}

inline void integrate_uz(
    constant const runge_kutta_coefficient_t &runge_kutta_coefficient,
    constant const float &dt,
    constant const float &diffusivity,
    device const metric_t * const z_center_metrics,
    device const metric_t * const z_face_metrics,
    device const float * const ux,
    device const float * const uy,
    device const float * const uz,
    device const float * const p,
    device const float * const temperature,
    device float * const duz,
    device float * const duz_explicit_previous,
    uint3 id
) {
  const uint32_t i = id.x + 1;
  const uint32_t j = id.y + 1;
  const uint32_t k = id.z + 1;
  if (NX < i || NY < j || k < 2 || NZ < k) {
    return;
  }
  const metric_t z_center_metric_zm = z_center_metrics[k - 1];
  const metric_t z_center_metric_zp = z_center_metrics[k    ];
  const metric_t z_face_metric = z_face_metrics[k];
  const float hz_c = z_face_metric.scale_factor;
  const float hz_m = z_center_metric_zm.scale_factor;
  const float hz_p = z_center_metric_zp.scale_factor;
  const float j_cc = HX * HY * hz_c;
  const float ux_xm = + 0.5f * HY * hz_m * UX(i    , j    , k - 1)
                      + 0.5f * HY * hz_p * UX(i    , j    , k    );
  const float ux_xp = + 0.5f * HY * hz_m * UX(i + 1, j    , k - 1)
                      + 0.5f * HY * hz_p * UX(i + 1, j    , k    );
  const float uy_ym = + 0.5f * HX * hz_m * UY(i    , j    , k - 1)
                      + 0.5f * HX * hz_p * UY(i    , j    , k    );
  const float uy_yp = + 0.5f * HX * hz_m * UY(i    , j + 1, k - 1)
                      + 0.5f * HX * hz_p * UY(i    , j + 1, k    );
  const float uz_zm = + 0.5f * HX * HY * UZ(i    , j    , k - 1)
                      + 0.5f * HX * HY * UZ(i    , j    , k    );
  const float uz_zp = + 0.5f * HX * HY * UZ(i    , j    , k    )
                      + 0.5f * HX * HY * UZ(i    , j    , k + 1);
  const float duz_xm = - UZ(i - 1, j    , k    ) + UZ(i    , j    , k    );
  const float duz_xp = - UZ(i    , j    , k    ) + UZ(i + 1, j    , k    );
  const float duz_ym = - UZ(i    , j - 1, k    ) + UZ(i    , j    , k    );
  const float duz_yp = - UZ(i    , j    , k    ) + UZ(i    , j + 1, k    );
  const float duz_zm = - UZ(i    , j    , k - 1) + UZ(i    , j    , k    );
  const float duz_zp = - UZ(i    , j    , k    ) + UZ(i    , j    , k + 1);
  const float advection_x = - 1.f / j_cc * (
      + 0.5f * ux_xm * duz_xm
      + 0.5f * ux_xp * duz_xp
  );
  const float advection_y = - 1.f / j_cc * (
      + 0.5f * uy_ym * duz_ym
      + 0.5f * uy_yp * duz_yp
  );
  const float advection_z = - 1.f / j_cc * (
      + 0.5f * uz_zm * duz_zm
      + 0.5f * uz_zp * duz_zp
  );
  const float pressure = - 1.f / hz_c * (
      - P(i    , j    , k - 1)
      + P(i    , j    , k    )
  );
  const float diffusion_x = diffusivity * (
      + (1.f / HX / HX) * UZ(i - 1, j    , k    )
      - (2.f / HX / HX) * UZ(i    , j    , k    )
      + (1.f / HX / HX) * UZ(i + 1, j    , k    )
  );
  const float diffusion_y = diffusivity * (
      + (1.f / HY / HY) * UZ(i    , j - 1, k    )
      - (2.f / HY / HY) * UZ(i    , j    , k    )
      + (1.f / HY / HY) * UZ(i    , j + 1, k    )
  );
  const float diffusion_z = diffusivity * (
      + z_face_metric.lower_diagonal * UZ(i    , j    , k - 1)
      + z_face_metric.main_diagonal  * UZ(i    , j    , k    )
      + z_face_metric.upper_diagonal * UZ(i    , j    , k + 1)
  );
  const float buoyancy =
    + 0.5f * TEMPERATURE(i    , j    , k - 1)
    + 0.5f * TEMPERATURE(i    , j    , k    );
  const uint32_t index = ((k - 2) * NY + (j - 1)) * NX + (i - 1);
  duz[index] = dt * (
      + runge_kutta_coefficient.explicit_current * (
          + advection_x
          + advection_y
          + advection_z
          + diffusion_x
          + diffusion_y
          + buoyancy
      )
      + runge_kutta_coefficient.explicit_previous * duz_explicit_previous[index]
      + runge_kutta_coefficient.implicit * (
          + pressure
          + diffusion_z
      )
  );
  duz_explicit_previous[index] =
    + advection_x
    + advection_y
    + advection_z
    + diffusion_x
    + diffusion_y
    + buoyancy;
}

inline void integrate_temperature(
    constant const runge_kutta_coefficient_t &runge_kutta_coefficient,
    constant const float &dt,
    constant const float &diffusivity,
    device const metric_t * const z_center_metrics,
    device const float * const ux,
    device const float * const uy,
    device const float * const uz,
    device const float * const temperature,
    device float * const dtemperature,
    device float * const dtemperature_explicit_previous,
    uint3 id
) {
  const uint32_t i = id.x + 1;
  const uint32_t j = id.y + 1;
  const uint32_t k = id.z + 1;
  if (NX < i || NY < j || NZ < k) {
    return;
  }
  const metric_t z_center_metric = z_center_metrics[k];
  const float hz_c = z_center_metric.scale_factor;
  const float j_cc = HX * HY * hz_c;
  const float ux_xm = HY * hz_c * UX(i    , j    , k    );
  const float ux_xp = HY * hz_c * UX(i + 1, j    , k    );
  const float uy_ym = HX * hz_c * UY(i    , j    , k    );
  const float uy_yp = HX * hz_c * UY(i    , j + 1, k    );
  const float uz_zm = HX * HY * UZ(i    , j    , k    );
  const float uz_zp = HX * HY * UZ(i    , j    , k + 1);
  const float dtemperature_xm = - TEMPERATURE(i - 1, j    , k    )
                                + TEMPERATURE(i    , j    , k    );
  const float dtemperature_xp = - TEMPERATURE(i    , j    , k    )
                                + TEMPERATURE(i + 1, j    , k    );
  const float dtemperature_ym = - TEMPERATURE(i    , j - 1, k    )
                                + TEMPERATURE(i    , j    , k    );
  const float dtemperature_yp = - TEMPERATURE(i    , j    , k    )
                                + TEMPERATURE(i    , j + 1, k    );
  const float dtemperature_zm = - TEMPERATURE(i    , j    , k - 1)
                                + TEMPERATURE(i    , j    , k    );
  const float dtemperature_zp = - TEMPERATURE(i    , j    , k    )
                                + TEMPERATURE(i    , j    , k + 1);
  const float advection_x = - 1.f / j_cc * (
      + 0.5f * ux_xm * dtemperature_xm
      + 0.5f * ux_xp * dtemperature_xp
  );
  const float advection_y = - 1.f / j_cc * (
      + 0.5f * uy_ym * dtemperature_ym
      + 0.5f * uy_yp * dtemperature_yp
  );
  const float advection_z = - 1.f / j_cc * (
      + 0.5f * uz_zm * dtemperature_zm
      + 0.5f * uz_zp * dtemperature_zp
  );
  const float diffusion_x = diffusivity * (
      + (1.f / HX / HX) * TEMPERATURE(i - 1, j    , k    )
      - (2.f / HX / HX) * TEMPERATURE(i    , j    , k    )
      + (1.f / HX / HX) * TEMPERATURE(i + 1, j    , k    )
  );
  const float diffusion_y = diffusivity * (
      + (1.f / HY / HY) * TEMPERATURE(i    , j - 1, k    )
      - (2.f / HY / HY) * TEMPERATURE(i    , j    , k    )
      + (1.f / HY / HY) * TEMPERATURE(i    , j + 1, k    )
  );
  const float diffusion_z = diffusivity * (
      + z_center_metric.lower_diagonal * TEMPERATURE(i    , j    , k - 1)
      + z_center_metric.main_diagonal  * TEMPERATURE(i    , j    , k    )
      + z_center_metric.upper_diagonal * TEMPERATURE(i    , j    , k + 1)
  );
  const uint32_t index = ((k - 1) * NY + (j - 1)) * NX + (i - 1);
  dtemperature[index] = dt * (
      + runge_kutta_coefficient.explicit_current * (
          + advection_x
          + advection_y
          + advection_z
          + diffusion_x
          + diffusion_y
      )
      + runge_kutta_coefficient.explicit_previous * dtemperature_explicit_previous[index]
      + runge_kutta_coefficient.implicit * (
          + diffusion_z
      )
  );
  dtemperature_explicit_previous[index] =
      + advection_x
      + advection_y
      + advection_z
      + diffusion_x
      + diffusion_y;
}

kernel void integrate_field(
    constant const runge_kutta_coefficient_t &runge_kutta_coefficient [[buffer(0)]],
    constant const float &dt [[buffer(1)]],
    constant const float &diffusivity [[buffer(2)]],
    device const metric_t * const z_center_metrics [[buffer(3)]],
    device const metric_t * const z_face_metrics [[buffer(4)]],
    device const float * const ux [[buffer(5)]],
    device const float * const uy [[buffer(6)]],
    device const float * const uz [[buffer(7)]],
    device const float * const p [[buffer(8)]],
    device const float * const temperature [[buffer(9)]],
    device float * const dux [[buffer(10)]],
    device float * const duy [[buffer(11)]],
    device float * const duz [[buffer(12)]],
    device float * const dtemperature [[buffer(13)]],
    device float * const dux_explicit_previous [[buffer(14)]],
    device float * const duy_explicit_previous [[buffer(15)]],
    device float * const duz_explicit_previous [[buffer(16)]],
    device float * const dtemperature_explicit_previous [[buffer(17)]],
    uint3 id [[thread_position_in_grid]]
) {
  integrate_ux(
      runge_kutta_coefficient,
      dt,
      diffusivity,
      z_center_metrics,
      ux,
      uy,
      uz,
      p,
      dux,
      dux_explicit_previous,
      id
  );
  integrate_uy(
      runge_kutta_coefficient,
      dt,
      diffusivity,
      z_center_metrics,
      ux,
      uy,
      uz,
      p,
      duy,
      duy_explicit_previous,
      id
  );
  integrate_uz(
      runge_kutta_coefficient,
      dt,
      diffusivity,
      z_center_metrics,
      z_face_metrics,
      ux,
      uy,
      uz,
      p,
      temperature,
      duz,
      duz_explicit_previous,
      id
  );
  integrate_temperature(
      runge_kutta_coefficient,
      dt,
      diffusivity,
      z_center_metrics,
      ux,
      uy,
      uz,
      temperature,
      dtemperature,
      dtemperature_explicit_previous,
      id
  );
}

