#include <metal_stdlib>

using namespace metal;

static_assert(0 == (THREADGROUP_SIZE & (THREADGROUP_SIZE - 1)));

kernel void finalize_time_step_size(
    constant const float &safety_factor [[buffer(0)]],
    constant const float &maximum_dt [[buffer(1)]],
    constant const uint &batch_index [[buffer(2)]],
    device const float * const convective_rate_maxs [[buffer(3)]],
    device float * const dt [[buffer(4)]],
    device float * const dt_history [[buffer(5)]],
    uint thread_index_in_threadgroup [[thread_index_in_threadgroup]]
) {
  threadgroup float local_convective_rate_maxs[THREADGROUP_SIZE];
  float local_max = 0.f;
  for (uint g = thread_index_in_threadgroup; g < NUM_GROUPS; g += THREADGROUP_SIZE) {
    local_max = max(local_max, convective_rate_maxs[g]);
  }
  local_convective_rate_maxs[thread_index_in_threadgroup] = local_max;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = THREADGROUP_SIZE / 2; 0 < s; s >>= 1) {
    if (thread_index_in_threadgroup < s) {
      local_convective_rate_maxs[thread_index_in_threadgroup] = max(
        local_convective_rate_maxs[thread_index_in_threadgroup],
        local_convective_rate_maxs[thread_index_in_threadgroup + s]
      );
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (0 == thread_index_in_threadgroup) {
    const float max_rate = local_convective_rate_maxs[0];
    // zero velocity: fall back to the upper bound
    const float value = 0.f < max_rate ? min(safety_factor / max_rate, maximum_dt) : maximum_dt;
    *dt = value;
    dt_history[batch_index] = value;
  }
}

