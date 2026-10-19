#include "shared_buffer.h"
#include "./check_quadratic_quantity.h"

int check_quadratic_quantity_data_initialize(
    id<MTLDevice> device,
    const domain_t * const domain,
    check_quadratic_quantity_data_t * const check_quadratic_quantity_data
) {
  const size_t nx = domain->x_coordinate.nitems;
  const size_t ny = domain->y_coordinate.nitems;
  const size_t nz = domain->z_coordinate.nitems;
  const double lx = domain->x_coordinate.length;
  const double ly = domain->y_coordinate.length;
  NSUInteger * const threadgroup_width = &check_quadratic_quantity_data->threadgroup_width;
  NSUInteger * const threadgroup_height = &check_quadratic_quantity_data->threadgroup_height;
  NSUInteger * const threadgroup_depth = &check_quadratic_quantity_data->threadgroup_depth;
  *threadgroup_width = 8;
  *threadgroup_height = 8;
  *threadgroup_depth = 4;
  if (0 != compute_pipeline_initialize(
      device,
      "src/monitor/check_quadratic_quantity.cpp",
      "check_quadratic_quantity",
      @{
        @"NX": @(nx),
        @"NY": @(ny),
        @"NZ": @(nz),
        @"HX": @(lx / nx),
        @"HY": @(ly / ny),
        @"THREADGROUP_WIDTH": @(*threadgroup_width),
        @"THREADGROUP_HEIGHT": @(*threadgroup_height),
        @"THREADGROUP_DEPTH": @(*threadgroup_depth),
        @"NQUANTITIES": @(NQUANTITIES),
      },
      &check_quadratic_quantity_data->compute_pipeline
  )) {
    return 1;
  }
  NSUInteger * const num_groups_x = &check_quadratic_quantity_data->num_groups_x;
  NSUInteger * const num_groups_y = &check_quadratic_quantity_data->num_groups_y;
  NSUInteger * const num_groups_z = &check_quadratic_quantity_data->num_groups_z;
  *num_groups_x = (nx + *threadgroup_width - 1) / *threadgroup_width;
  *num_groups_y = (ny + *threadgroup_height - 1) / *threadgroup_height;
  *num_groups_z = (nz + *threadgroup_depth - 1) / *threadgroup_depth;
  check_quadratic_quantity_data->max_num_threadgroups = *num_groups_x * *num_groups_y * *num_groups_z;
  check_quadratic_quantity_data->mtl_buffer_quadratic_quantities = create_shared_buffer(
      device,
      NQUANTITIES * check_quadratic_quantity_data->max_num_threadgroups * sizeof(float)
  );
  return 0;
}

int process_on_gpu_check_quadratic_quantity(
    id<MTLComputeCommandEncoder> encoder,
    check_quadratic_quantity_data_t * const check_quadratic_quantity_data,
    const domain_t * const domain,
    const flow_field_t * const flow_field
) {
  [encoder setComputePipelineState:check_quadratic_quantity_data->compute_pipeline.pipeline_state];
  [encoder setBuffer:domain->z_coordinate.mtl_buffer_center_scale_factors offset:0 atIndex:0];
  [encoder setBuffer:domain->z_coordinate.mtl_buffer_face_scale_factors offset:0 atIndex:1];
  [encoder setBuffer:flow_field->ux offset:0 atIndex:2];
  [encoder setBuffer:flow_field->uy offset:0 atIndex:3];
  [encoder setBuffer:flow_field->uz offset:0 atIndex:4];
  [encoder setBuffer:flow_field->temperature offset:0 atIndex:5];
  [encoder setBuffer:check_quadratic_quantity_data->mtl_buffer_quadratic_quantities offset:0 atIndex:6];
  MTLSize thread_group_size = MTLSizeMake(
      check_quadratic_quantity_data->threadgroup_width,
      check_quadratic_quantity_data->threadgroup_height,
      check_quadratic_quantity_data->threadgroup_depth
  );
  MTLSize groupsSize = MTLSizeMake(
      check_quadratic_quantity_data->num_groups_x,
      check_quadratic_quantity_data->num_groups_y,
      check_quadratic_quantity_data->num_groups_z
  );
  [encoder dispatchThreadgroups:groupsSize threadsPerThreadgroup:thread_group_size];
  return 0;
}

int process_on_cpu_check_quadratic_quantity(
    check_quadratic_quantity_data_t * const check_quadratic_quantity_data,
    double * const quantities
) {
  const float * const quadratic_quantities = [check_quadratic_quantity_data->mtl_buffer_quadratic_quantities contents];
  for (size_t n = 0; n < NQUANTITIES; n++) {
    double * const quantity = quantities + n;
    *quantity = 0.;
    for (NSUInteger m = 0; m < check_quadratic_quantity_data->max_num_threadgroups; m++) {
      *quantity += quadratic_quantities[n * check_quadratic_quantity_data->max_num_threadgroups + m];
    }
  }
  return 0;
}

