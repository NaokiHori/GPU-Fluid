#include "./check_divergence.h"

int check_divergence_data_initialize(
    id<MTLDevice> device,
    const domain_t * const domain,
    check_divergence_data_t * const check_divergence_data
) {
  const size_t nx = domain->x_coordinate.nitems;
  const size_t ny = domain->y_coordinate.nitems;
  NSUInteger * const threadgroup_width = &check_divergence_data->threadgroup_width;
  NSUInteger * const threadgroup_height = &check_divergence_data->threadgroup_height;
  *threadgroup_width = 16;
  *threadgroup_height = 16;
  if (0 != compute_pipeline_initialize(
      device,
      "src/monitor/check_divergence.cpp",
      "check_divergence",
      @{
        @"NX": @(nx),
        @"NY": @(ny),
        @"THREADGROUP_WIDTH": @(*threadgroup_width),
        @"THREADGROUP_HEIGHT": @(*threadgroup_height),
      },
      &check_divergence_data->compute_pipeline
  )) {
    return 1;
  }
  NSUInteger * const num_groups_x = &check_divergence_data->num_groups_x;
  NSUInteger * const num_groups_y = &check_divergence_data->num_groups_y;
  *num_groups_x = (nx + *threadgroup_width - 1) / *threadgroup_width;
  *num_groups_y = (ny + *threadgroup_height - 1) / *threadgroup_height;
  check_divergence_data->max_num_threadgroups = *num_groups_x * *num_groups_y;
  check_divergence_data->mtl_buffer_divergence_sums = [
    device
      newBufferWithLength:check_divergence_data->max_num_threadgroups * sizeof(float)
      options:MTLResourceStorageModeShared
  ];
  check_divergence_data->mtl_buffer_divergence_maxs = [
    device
      newBufferWithLength:check_divergence_data->max_num_threadgroups * sizeof(float)
      options:MTLResourceStorageModeShared
  ];
  return 0;
}

int process_on_gpu_check_divergence(
    id<MTLComputeCommandEncoder> encoder,
    check_divergence_data_t * const check_divergence_data,
    const domain_t * const domain,
    const flow_field_t * const flow_field
) {
  [encoder setComputePipelineState:check_divergence_data->compute_pipeline.pipeline_state];
  [encoder setBuffer:domain->x_coordinate.mtl_buffer_center_scale_factors offset:0 atIndex:0];
  [encoder setBuffer:domain->y_coordinate.mtl_buffer_center_scale_factors offset:0 atIndex:1];
  [encoder setBuffer:flow_field->ux offset:0 atIndex:2];
  [encoder setBuffer:flow_field->uy offset:0 atIndex:3];
  [encoder setBuffer:check_divergence_data->mtl_buffer_divergence_sums offset:0 atIndex:4];
  [encoder setBuffer:check_divergence_data->mtl_buffer_divergence_maxs offset:0 atIndex:5];
  MTLSize thread_group_size = MTLSizeMake(
      check_divergence_data->threadgroup_width,
      check_divergence_data->threadgroup_height,
      1
  );
  MTLSize groupsSize = MTLSizeMake(
      check_divergence_data->num_groups_x,
      check_divergence_data->num_groups_y,
      1
  );
  [encoder dispatchThreadgroups:groupsSize threadsPerThreadgroup:thread_group_size];
  return 0;
}

int process_on_cpu_check_divergence(
    check_divergence_data_t * const check_divergence_data,
    const size_t step,
    const double time
) {
  double total_divergence = 0.;
  double max_divergence = 0.;
  const float * const divergence_sums = [check_divergence_data->mtl_buffer_divergence_sums contents];
  for (NSUInteger j = 0; j < check_divergence_data->num_groups_y; j++) {
    for (NSUInteger i = 0; i < check_divergence_data->num_groups_x; i++) {
      total_divergence += divergence_sums[j * check_divergence_data->num_groups_x + i];
    }
  }
  const float * const divergence_maxs = [check_divergence_data->mtl_buffer_divergence_maxs contents];
  for (NSUInteger j = 0; j < check_divergence_data->num_groups_y; j++) {
    for (NSUInteger i = 0; i < check_divergence_data->num_groups_x; i++) {
      max_divergence = fmaxf(
          max_divergence,
          divergence_maxs[j * check_divergence_data->num_groups_x + i]
      );
    }
  }
  printf("%zu % .7e % .7e % .7e\n", step, time, total_divergence, max_divergence);
  return 0;
}

