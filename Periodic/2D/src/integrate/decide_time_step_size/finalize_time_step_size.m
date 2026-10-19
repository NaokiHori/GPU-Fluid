#include "shared_buffer.h"
#include "./finalize_time_step_size.h"

int finalize_time_step_size_data_initialize(
    id<MTLDevice> device,
    const domain_t * const domain,
    const NSUInteger num_groups_x,
    const NSUInteger num_groups_y,
    id<MTLBuffer> * const mtl_buffer_convective_rate_maxs,
    finalize_time_step_size_data_t * const finalize_time_step_size_data
) {
  const NSUInteger max_num_threadgroups = num_groups_x * num_groups_y;
  *mtl_buffer_convective_rate_maxs = create_shared_buffer(
      device,
      max_num_threadgroups * sizeof(float)
  );
  const NSUInteger finalize_threadgroup_size = 256;
  if (0 != compute_pipeline_initialize(
      device,
      "src/integrate/decide_time_step_size/finalize_time_step_size.cpp",
      "finalize_time_step_size",
      @{
        @"NX": @(domain->x_coordinate.nitems),
        @"NY": @(domain->y_coordinate.nitems),
        @"THREADGROUP_SIZE": @(finalize_threadgroup_size),
        @"NUM_GROUPS": @(max_num_threadgroups),
      },
      &finalize_time_step_size_data->compute_pipeline
  )) {
    return 1;
  }
  return 0;
}

int finalize_time_step_size(
    finalize_time_step_size_data_t * const finalize_time_step_size_data,
    id<MTLComputeCommandEncoder> encoder,
    id<MTLBuffer> mtl_buffer_convective_rate_maxs,
    id<MTLBuffer> mtl_buffer_dt,
    id<MTLBuffer> mtl_buffer_dt_history,
    const uint32_t batch
) {
  const float safety_factor = 0.5f;
  const float maximum_dt = 1.e-2f;
  [encoder setComputePipelineState:finalize_time_step_size_data->compute_pipeline.pipeline_state];
  [encoder setBytes:&safety_factor length:sizeof(float) atIndex:0];
  [encoder setBytes:&maximum_dt length:sizeof(float) atIndex:1];
  [encoder setBytes:&batch length:sizeof(uint32_t) atIndex:2];
  [encoder setBuffer:mtl_buffer_convective_rate_maxs offset:0 atIndex:3];
  [encoder setBuffer:mtl_buffer_dt offset:0 atIndex:4];
  [encoder setBuffer:mtl_buffer_dt_history offset:0 atIndex:5];
  [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  return 0;
}

