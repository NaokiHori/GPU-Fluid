#include "shared_buffer.h"
#include "./check_divergence.h"

int check_divergence_data_initialize(
    id<MTLDevice> device,
    const domain_t * const domain,
    check_divergence_data_t * const check_divergence_data
) {
  const size_t nx = domain->x_coordinate.nitems;
  const size_t ny = domain->y_coordinate.nitems;
  const size_t nz = domain->z_coordinate.nitems;
  const double lx = domain->x_coordinate.length;
  const double ly = domain->y_coordinate.length;
  NSUInteger * const threadgroup_width = &check_divergence_data->threadgroup_width;
  NSUInteger * const threadgroup_height = &check_divergence_data->threadgroup_height;
  NSUInteger * const threadgroup_depth = &check_divergence_data->threadgroup_depth;
  *threadgroup_width = 8;
  *threadgroup_height = 8;
  *threadgroup_depth = 4;
  if (0 != compute_pipeline_initialize(
      device,
      "src/monitor/check_divergence.cpp",
      "check_divergence",
      @{
        @"NX": @(nx),
        @"NY": @(ny),
        @"NZ": @(nz),
        @"HX": @(lx / nx),
        @"HY": @(ly / ny),
        @"THREADGROUP_WIDTH": @(*threadgroup_width),
        @"THREADGROUP_HEIGHT": @(*threadgroup_height),
        @"THREADGROUP_DEPTH": @(*threadgroup_depth),
      },
      &check_divergence_data->compute_pipeline
  )) {
    return 1;
  }
  NSUInteger * const num_groups_x = &check_divergence_data->num_groups_x;
  NSUInteger * const num_groups_y = &check_divergence_data->num_groups_y;
  NSUInteger * const num_groups_z = &check_divergence_data->num_groups_z;
  *num_groups_x = (nx + *threadgroup_width - 1) / *threadgroup_width;
  *num_groups_y = (ny + *threadgroup_height - 1) / *threadgroup_height;
  *num_groups_z = (nz + *threadgroup_depth - 1) / *threadgroup_depth;
  check_divergence_data->max_num_threadgroups = *num_groups_x * *num_groups_y * *num_groups_z;
  check_divergence_data->mtl_buffer_divergence_sums = create_shared_buffer(
      device,
      check_divergence_data->max_num_threadgroups * sizeof(float)
  );
  check_divergence_data->mtl_buffer_divergence_maxs = create_shared_buffer(
      device,
      check_divergence_data->max_num_threadgroups * sizeof(float)
  );
  return 0;
}

int process_on_gpu_check_divergence(
    id<MTLComputeCommandEncoder> encoder,
    check_divergence_data_t * const check_divergence_data,
    const domain_t * const domain,
    const flow_field_t * const flow_field
) {
  [encoder setComputePipelineState:check_divergence_data->compute_pipeline.pipeline_state];
  [encoder setBuffer:domain->z_coordinate.mtl_buffer_center_scale_factors offset:0 atIndex:0];
  [encoder setBuffer:flow_field->ux offset:0 atIndex:1];
  [encoder setBuffer:flow_field->uy offset:0 atIndex:2];
  [encoder setBuffer:flow_field->uz offset:0 atIndex:3];
  [encoder setBuffer:check_divergence_data->mtl_buffer_divergence_sums offset:0 atIndex:4];
  [encoder setBuffer:check_divergence_data->mtl_buffer_divergence_maxs offset:0 atIndex:5];
  MTLSize thread_group_size = MTLSizeMake(
      check_divergence_data->threadgroup_width,
      check_divergence_data->threadgroup_height,
      check_divergence_data->threadgroup_depth
  );
  MTLSize groupsSize = MTLSizeMake(
      check_divergence_data->num_groups_x,
      check_divergence_data->num_groups_y,
      check_divergence_data->num_groups_z
  );
  [encoder dispatchThreadgroups:groupsSize threadsPerThreadgroup:thread_group_size];
  return 0;
}

int process_on_cpu_check_divergence(
    check_divergence_data_t * const check_divergence_data,
    double * const total_divergence,
    double * const max_divergence
) {
  *total_divergence = 0.;
  *max_divergence = 0.;
  const float * const divergence_sums = [check_divergence_data->mtl_buffer_divergence_sums contents];
  const float * const divergence_maxs = [check_divergence_data->mtl_buffer_divergence_maxs contents];
  for (NSUInteger n = 0; n < check_divergence_data->max_num_threadgroups; n++) {
    *total_divergence += divergence_sums[n];
    *max_divergence = fmax(*max_divergence, divergence_maxs[n]);
  }
  return 0;
}

