#include "./compute_convective_rate.h"

int compute_convective_rate_data_initialize(
    id<MTLDevice> device,
    const domain_t * const domain,
    compute_convective_rate_data_t * const compute_convective_rate_data
) {
  const size_t nx = domain->x_coordinate.nitems;
  const size_t ny = domain->y_coordinate.nitems;
  NSUInteger * const threadgroup_width = &compute_convective_rate_data->threadgroup_width;
  NSUInteger * const threadgroup_height = &compute_convective_rate_data->threadgroup_height;
  *threadgroup_width = 16;
  *threadgroup_height = 16;
  if (0 != compute_pipeline_initialize(
      device,
      "src/integrate/decide_time_step_size/compute_convective_rate.cpp",
      "compute_convective_rate",
      @{
        @"NX": @(nx),
        @"NY": @(ny),
        @"THREADGROUP_WIDTH": @(*threadgroup_width),
        @"THREADGROUP_HEIGHT": @(*threadgroup_height),
      },
      &compute_convective_rate_data->compute_pipeline
  )) {
    return 1;
  }
  NSUInteger * const num_groups_x = &compute_convective_rate_data->num_groups_x;
  NSUInteger * const num_groups_y = &compute_convective_rate_data->num_groups_y;
  *num_groups_x = (nx + *threadgroup_width - 1) / *threadgroup_width;
  *num_groups_y = (ny + *threadgroup_height - 1) / *threadgroup_height;
  return 0;
}

int compute_convective_rate(
    const domain_t * const domain,
    compute_convective_rate_data_t * const compute_convective_rate_data,
    id<MTLBuffer> mtl_buffer_convective_rate_maxs,
    id<MTLComputeCommandEncoder> encoder,
    flow_field_t * const flow_field
) {
  [encoder setComputePipelineState:compute_convective_rate_data->compute_pipeline.pipeline_state];
  [encoder setBuffer:domain->x_coordinate.mtl_buffer_center_scale_factors offset:0 atIndex:0];
  [encoder setBuffer:domain->y_coordinate.mtl_buffer_center_scale_factors offset:0 atIndex:1];
  [encoder setBuffer:flow_field->ux offset:0 atIndex:2];
  [encoder setBuffer:flow_field->uy offset:0 atIndex:3];
  [encoder setBuffer:mtl_buffer_convective_rate_maxs offset:0 atIndex:4];
  MTLSize thread_group_size = MTLSizeMake(
      compute_convective_rate_data->threadgroup_width,
      compute_convective_rate_data->threadgroup_height,
      1
  );
  MTLSize groupsSize = MTLSizeMake(
      compute_convective_rate_data->num_groups_x,
      compute_convective_rate_data->num_groups_y,
      1
  );
  [encoder dispatchThreadgroups:groupsSize threadsPerThreadgroup:thread_group_size];
  return 0;
}

