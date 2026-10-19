#include <metal_stdlib>

using namespace metal;

#define UX(i, j, k) ux[((k) * (NY + 2) + (j)) * (NX + 2) + (i)]
#define UY(i, j, k) uy[((k) * (NY + 2) + (j)) * (NX + 2) + (i)]
#define UZ(i, j, k) uz[((k) * (NY + 2) + (j)) * (NX + 2) + (i)]

kernel void compute_convective_rate(
    device const float * const z_center_scale_factors [[buffer(0)]],
    device const float * const ux [[buffer(1)]],
    device const float * const uy [[buffer(2)]],
    device const float * const uz [[buffer(3)]],
    device float * const convective_rate_maxs [[buffer(4)]],
    uint3 id [[thread_position_in_grid]],
    uint thread_index_in_threadgroup [[thread_index_in_threadgroup]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint3 threadgroups_per_grid [[threadgroups_per_grid]]
) {
  const uint i = id.x + 1;
  const uint j = id.y + 1;
  const uint k = id.z + 1;
  //
  threadgroup float local_convective_rate_maxs[THREADGROUP_WIDTH * THREADGROUP_HEIGHT * THREADGROUP_DEPTH];
  threadgroup float * const local_convective_rate_max = local_convective_rate_maxs + thread_index_in_threadgroup;
  if (NX < i || NY < j || NZ < k) {
    *local_convective_rate_max = 0.f;
  } else {
    const float hz = z_center_scale_factors[k];
    const float ux_max = max(
        fabs(UX(i    , j, k)),
        fabs(UX(i + 1, j, k))
    );
    const float uy_max = max(
        fabs(UY(i, j    , k)),
        fabs(UY(i, j + 1, k))
    );
    const float uz_max = max(
        fabs(UZ(i, j, k    )),
        fabs(UZ(i, j, k + 1))
    );
    *local_convective_rate_max = ux_max / HX + uy_max / HY + uz_max / hz;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = THREADGROUP_WIDTH * THREADGROUP_HEIGHT * THREADGROUP_DEPTH / 2; 0 < s; s >>= 1) {
    if (thread_index_in_threadgroup < s) {
      local_convective_rate_maxs[thread_index_in_threadgroup] = max(
        local_convective_rate_maxs[thread_index_in_threadgroup],
        local_convective_rate_maxs[thread_index_in_threadgroup + s]
      );
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (0 == thread_index_in_threadgroup) {
    const uint group_index
      = (threadgroup_position_in_grid.z * threadgroups_per_grid.y + threadgroup_position_in_grid.y)
      * threadgroups_per_grid.x
      + threadgroup_position_in_grid.x;
    convective_rate_maxs[group_index] = local_convective_rate_maxs[0];
  }
}

