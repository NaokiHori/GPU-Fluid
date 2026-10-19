#include <metal_stdlib>

using namespace metal;

#define UX(i, j, k) ux[((k) * (NY + 2) + (j)) * (NX + 2) + (i)]
#define UY(i, j, k) uy[((k) * (NY + 2) + (j)) * (NX + 2) + (i)]
#define UZ(i, j, k) uz[((k) * (NY + 2) + (j)) * (NX + 2) + (i)]
#define TEMPERATURE(i, j, k) temperature[((k) * (NY + 2) + (j)) * (NX + 2) + (i)]

#define THREADGROUP_SIZE (THREADGROUP_WIDTH * THREADGROUP_HEIGHT * THREADGROUP_DEPTH)

kernel void check_quadratic_quantity(
    device const float * const z_center_scale_factors [[buffer(0)]],
    device const float * const z_face_scale_factors [[buffer(1)]],
    device const float * const ux [[buffer(2)]],
    device const float * const uy [[buffer(3)]],
    device const float * const uz [[buffer(4)]],
    device const float * const temperature [[buffer(5)]],
    device float * const quadratic_quantities [[buffer(6)]],
    uint3 id [[thread_position_in_grid]],
    uint thread_index_in_threadgroup [[thread_index_in_threadgroup]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint3 threadgroups_per_grid [[threadgroups_per_grid]]
) {
  const uint i = id.x + 1;
  const uint j = id.y + 1;
  const uint k = id.z + 1;
  // [quantity][thread]: ux, uy, uz, temperature
  threadgroup float local_quadratic_quantities[NQUANTITIES][THREADGROUP_SIZE];
  float locals[NQUANTITIES] = {0.f, 0.f, 0.f, 0.f};
  if (i <= NX && j <= NY && k <= NZ) {
    const float volume_center = HX * HY * z_center_scale_factors[k];
    const float ux_value = UX(i, j, k);
    const float uy_value = UY(i, j, k);
    const float temperature_value = TEMPERATURE(i, j, k);
    locals[0] = volume_center * 0.5f * ux_value * ux_value;
    locals[1] = volume_center * 0.5f * uy_value * uy_value;
    locals[3] = volume_center * 0.5f * temperature_value * temperature_value;
    // wall faces (k = 1) carry no unknown
    if (2 <= k) {
      const float volume_face = HX * HY * z_face_scale_factors[k];
      const float uz_value = UZ(i, j, k);
      locals[2] = volume_face * 0.5f * uz_value * uz_value;
    }
  }
  for (uint q = 0; q < NQUANTITIES; q++) {
    local_quadratic_quantities[q][thread_index_in_threadgroup] = locals[q];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = THREADGROUP_SIZE / 2; 0 < s; s >>= 1) {
    if (thread_index_in_threadgroup < s) {
      for (uint q = 0; q < NQUANTITIES; q++) {
        local_quadratic_quantities[q][thread_index_in_threadgroup] +=
          local_quadratic_quantities[q][thread_index_in_threadgroup + s];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (0 == thread_index_in_threadgroup) {
    const uint num_groups = threadgroups_per_grid.x * threadgroups_per_grid.y * threadgroups_per_grid.z;
    const uint group_index
      = (threadgroup_position_in_grid.z * threadgroups_per_grid.y + threadgroup_position_in_grid.y)
      * threadgroups_per_grid.x
      + threadgroup_position_in_grid.x;
    for (uint q = 0; q < NQUANTITIES; q++) {
      quadratic_quantities[q * num_groups + group_index] = local_quadratic_quantities[q][0];
    }
  }
}

