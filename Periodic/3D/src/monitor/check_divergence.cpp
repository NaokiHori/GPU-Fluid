#include <metal_stdlib>

using namespace metal;

#define UX(i, j, k) ux[((k) * (NY + 2) + (j)) * (NX + 2) + (i)]
#define UY(i, j, k) uy[((k) * (NY + 2) + (j)) * (NX + 2) + (i)]
#define UZ(i, j, k) uz[((k) * (NY + 2) + (j)) * (NX + 2) + (i)]

kernel void check_divergence(
    device const float * const z_center_scale_factors [[buffer(0)]],
    device const float * const ux [[buffer(1)]],
    device const float * const uy [[buffer(2)]],
    device const float * const uz [[buffer(3)]],
    device float * const divergence_sums [[buffer(4)]],
    device float * const divergence_maxs [[buffer(5)]],
    uint3 id [[thread_position_in_grid]],
    uint thread_index_in_threadgroup [[thread_index_in_threadgroup]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint3 threadgroups_per_grid [[threadgroups_per_grid]]
) {
  const uint i = id.x + 1;
  const uint j = id.y + 1;
  const uint k = id.z + 1;
  threadgroup float local_divergence_sums[THREADGROUP_WIDTH * THREADGROUP_HEIGHT * THREADGROUP_DEPTH];
  threadgroup float local_divergence_maxs[THREADGROUP_WIDTH * THREADGROUP_HEIGHT * THREADGROUP_DEPTH];
  threadgroup float * const local_divergence_sum = local_divergence_sums + thread_index_in_threadgroup;
  threadgroup float * const local_divergence_max = local_divergence_maxs + thread_index_in_threadgroup;
  if (NX < i || NY < j || NZ < k) {
    *local_divergence_sum = 0.f;
    *local_divergence_max = 0.f;
  } else {
    const float hz = z_center_scale_factors[k];
    const float duxdx = 1.f / HX * (
        - UX(i    , j, k)
        + UX(i + 1, j, k)
    );
    const float duydy = 1.f / HY * (
        - UY(i, j    , k)
        + UY(i, j + 1, k)
    );
    const float duzdz = 1.f / hz * (
        - UZ(i, j, k    )
        + UZ(i, j, k + 1)
    );
    const float divergence = duxdx + duydy + duzdz;
    *local_divergence_sum = divergence * HX * HY * hz;
    *local_divergence_max = fabs(divergence);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = THREADGROUP_WIDTH * THREADGROUP_HEIGHT * THREADGROUP_DEPTH / 2; 0 < s; s >>= 1) {
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
      = (threadgroup_position_in_grid.z * threadgroups_per_grid.y + threadgroup_position_in_grid.y)
      * threadgroups_per_grid.x
      + threadgroup_position_in_grid.x;
    divergence_sums[group_index] = local_divergence_sums[0];
    divergence_maxs[group_index] = local_divergence_maxs[0];
  }
}

