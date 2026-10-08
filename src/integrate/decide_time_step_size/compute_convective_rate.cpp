#include <metal_stdlib>

using namespace metal;

kernel void compute_convective_rate(
    device const float * const x_center_scale_factors [[buffer(0)]],
    device const float * const y_center_scale_factors [[buffer(1)]],
    device const float * const ux [[buffer(2)]],
    device const float * const uy [[buffer(3)]],
    device float * const convective_rate_maxs [[buffer(4)]],
    uint2 id [[thread_position_in_grid]],
    uint thread_index_in_threadgroup [[thread_index_in_threadgroup]],
    uint2 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint2 threadgroups_per_grid [[threadgroups_per_grid]]
) {
  const uint i = id.x + 1;
  const uint j = id.y + 1;
  //
  threadgroup float local_convective_rate_maxs[THREADGROUP_WIDTH * THREADGROUP_HEIGHT];
  threadgroup float * const local_convective_rate_max = local_convective_rate_maxs + thread_index_in_threadgroup;
  if (NX < i || NY < j) {
    *local_convective_rate_max = 0.f;
  } else {
    const float hx = x_center_scale_factors[i];
    const float hy = y_center_scale_factors[j];
    const float ux_max = max(
        fabs(ux[(j    ) * (NX + 2) + (i    )]),
        fabs(ux[(j    ) * (NX + 2) + (i + 1)])
    );
    const float uy_max = max(
        fabs(uy[(j    ) * (NX + 2) + (i    )]),
        fabs(uy[(j + 1) * (NX + 2) + (i    )])
    );
    *local_convective_rate_max = ux_max / hx + uy_max / hy;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = THREADGROUP_WIDTH * THREADGROUP_HEIGHT / 2; 0 < s; s >>= 1) {
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
      = threadgroup_position_in_grid.y
      * threadgroups_per_grid.x
      + threadgroup_position_in_grid.x;
    convective_rate_maxs[group_index] = local_convective_rate_maxs[0];
  }
}

