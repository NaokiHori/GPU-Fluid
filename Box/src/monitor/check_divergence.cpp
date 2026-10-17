#include <metal_stdlib>

using namespace metal;

kernel void check_divergence(
    device const float * const x_center_scale_factors [[buffer(0)]],
    device const float * const y_center_scale_factors [[buffer(1)]],
    device const float * const ux [[buffer(2)]],
    device const float * const uy [[buffer(3)]],
    device float * const divergence_sums [[buffer(4)]],
    device float * const divergence_maxs [[buffer(5)]],
    uint2 id [[thread_position_in_grid]],
    uint thread_index_in_threadgroup [[thread_index_in_threadgroup]],
    uint2 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint2 threadgroups_per_grid [[threadgroups_per_grid]]
) {
  const uint i = id.x + 1;
  const uint j = id.y + 1;
  threadgroup float local_divergence_sums[THREADGROUP_WIDTH * THREADGROUP_HEIGHT];
  threadgroup float local_divergence_maxs[THREADGROUP_WIDTH * THREADGROUP_HEIGHT];
  threadgroup float * const local_divergence_sum = local_divergence_sums + thread_index_in_threadgroup;
  threadgroup float * const local_divergence_max = local_divergence_maxs + thread_index_in_threadgroup;
  if (NX < i || NY < j) {
    *local_divergence_sum = 0.f;
    *local_divergence_max = 0.f;
  } else {
    const float hx = x_center_scale_factors[i];
    const float hy = y_center_scale_factors[j];
    const float duxdx = 1.f / hx * (
        - ux[(j    ) * (NX + 2) + (i    )]
        + ux[(j    ) * (NX + 2) + (i + 1)]
    );
    const float duydy = 1.f / hy * (
        - uy[(j    ) * (NX + 2) + (i    )]
        + uy[(j + 1) * (NX + 2) + (i    )]
    );
    const float divergence = duxdx + duydy;
    *local_divergence_sum = divergence * hx * hy;
    *local_divergence_max = divergence;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = THREADGROUP_WIDTH * THREADGROUP_HEIGHT / 2; 0 < s; s >>= 1) {
    if (thread_index_in_threadgroup < s) {
      local_divergence_sums[thread_index_in_threadgroup] =
        + local_divergence_sums[thread_index_in_threadgroup]
        + local_divergence_sums[thread_index_in_threadgroup + s];
      local_divergence_maxs[thread_index_in_threadgroup] = max(
        local_divergence_maxs[thread_index_in_threadgroup],
        local_divergence_maxs[thread_index_in_threadgroup + s]
      );
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (0 == thread_index_in_threadgroup) {
    const uint group_index
      = threadgroup_position_in_grid.y
      * threadgroups_per_grid.x
      + threadgroup_position_in_grid.x;
    divergence_sums[group_index] = local_divergence_sums[0];
    divergence_maxs[group_index] = local_divergence_maxs[0];
  }
}

