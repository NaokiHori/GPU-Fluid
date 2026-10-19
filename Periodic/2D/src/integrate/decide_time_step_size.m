#include "shared_buffer.h"
#include "./decide_time_step_size.h"

int decide_time_step_size_data_initialize(
    id<MTLDevice> device,
    const domain_t * const domain,
    const size_t batch_size,
    decide_time_step_size_data_t * const decide_time_step_size_data
) {
  compute_convective_rate_data_t * const compute_convective_rate_data = &decide_time_step_size_data->compute_convective_rate_data;
  finalize_time_step_size_data_t * const finalize_time_step_size_data = &decide_time_step_size_data->finalize_time_step_size_data;
  if (0 != compute_convective_rate_data_initialize(
      device,
      domain,
      compute_convective_rate_data
  )) {
    return 1;
  }
  if (0 != finalize_time_step_size_data_initialize(
      device,
      domain,
      compute_convective_rate_data->num_groups_x,
      compute_convective_rate_data->num_groups_y,
      &decide_time_step_size_data->mtl_buffer_convective_rate_maxs,
      finalize_time_step_size_data
  )) {
    return 1;
  }
  decide_time_step_size_data->mtl_buffer_dt = create_shared_buffer(device, sizeof(float));
  decide_time_step_size_data->mtl_buffer_dt_history = create_shared_buffer(
      device,
      batch_size * sizeof(float)
  );
  return 0;
}

int decide_time_step_size(
    const domain_t * const domain,
    decide_time_step_size_data_t * const decide_time_step_size_data,
    id<MTLComputeCommandEncoder> encoder,
    flow_field_t * const flow_field,
    const uint32_t batch
) {
  id<MTLBuffer> mtl_buffer_convective_rate_maxs = decide_time_step_size_data->mtl_buffer_convective_rate_maxs;
  compute_convective_rate(
      domain,
      &decide_time_step_size_data->compute_convective_rate_data,
      mtl_buffer_convective_rate_maxs,
      encoder,
      flow_field
  );
  finalize_time_step_size(
      &decide_time_step_size_data->finalize_time_step_size_data,
      encoder,
      mtl_buffer_convective_rate_maxs,
      decide_time_step_size_data->mtl_buffer_dt,
      decide_time_step_size_data->mtl_buffer_dt_history,
      batch
  );
  return 0;
}

int update_time(
    decide_time_step_size_data_t * const decide_time_step_size_data,
    const size_t batch_size,
    size_t * const step,
    double * const time
) {
  *step += batch_size;
  const float * const dt_history = [decide_time_step_size_data->mtl_buffer_dt_history contents];
  for (size_t n = 0; n < batch_size; n++) {
    *time += dt_history[n];
  }
  return 0;
}

