#include "impose_boundary_condition.h"

int impose_boundary_condition_data_initialize(
    id<MTLDevice> device,
    const domain_t * const domain,
    impose_boundary_condition_data_t * const impose_boundary_condition_data
) {
  const size_t nx = domain->x_coordinate.nitems;
  const size_t ny = domain->y_coordinate.nitems;
  if (0 != compute_pipeline_initialize(
      device,
      "src/impose_boundary_condition.cpp",
      "impose_boundary_condition_x",
      @{
        @"NX": @(nx),
        @"NY": @(ny),
      },
      &impose_boundary_condition_data->compute_pipeline_x
  )) {
    return 1;
  }
  if (0 != compute_pipeline_initialize(
      device,
      "src/impose_boundary_condition.cpp",
      "impose_boundary_condition_y",
      @{
        @"NX": @(nx),
        @"NY": @(ny),
      },
      &impose_boundary_condition_data->compute_pipeline_y
  )) {
    return 1;
  }
  return 0;
}

static int impose_boundary_condition_x(
    const domain_t * const domain,
    compute_pipeline_t * const compute_pipeline,
    id<MTLComputeCommandEncoder> encoder,
    flow_field_t * const flow_field
) {
  const NSUInteger ny = domain->y_coordinate.nitems;
  [encoder setComputePipelineState:compute_pipeline->pipeline_state];
  [encoder setBuffer:flow_field->ux offset:0 atIndex:0];
  [encoder setBuffer:flow_field->uy offset:0 atIndex:1];
  [encoder setBuffer:flow_field->temperature offset:0 atIndex:2];
  //
  MTLSize grid_size = MTLSizeMake(ny, 1, 1);
  NSUInteger max_threads = [compute_pipeline->pipeline_state maxTotalThreadsPerThreadgroup];
  NSUInteger thread_group_size_value = MIN(ny, max_threads);
  MTLSize thread_group_size = MTLSizeMake(thread_group_size_value, 1, 1);
  [encoder dispatchThreads:grid_size threadsPerThreadgroup:thread_group_size];
  return 0;
}

static int impose_boundary_condition_y(
    const domain_t * const domain,
    compute_pipeline_t * const compute_pipeline,
    id<MTLComputeCommandEncoder> encoder,
    flow_field_t * const flow_field
) {
  const NSUInteger nx = domain->x_coordinate.nitems;
  [encoder setComputePipelineState:compute_pipeline->pipeline_state];
  [encoder setBuffer:flow_field->ux offset:0 atIndex:0];
  [encoder setBuffer:flow_field->uy offset:0 atIndex:1];
  [encoder setBuffer:flow_field->temperature offset:0 atIndex:2];
  //
  MTLSize grid_size = MTLSizeMake(nx, 1, 1);
  NSUInteger max_threads = [compute_pipeline->pipeline_state maxTotalThreadsPerThreadgroup];
  NSUInteger thread_group_size_value = MIN(nx, max_threads);
  MTLSize thread_group_size = MTLSizeMake(thread_group_size_value, 1, 1);
  [encoder dispatchThreads:grid_size threadsPerThreadgroup:thread_group_size];
  return 0;
}

int impose_boundary_condition(
    const domain_t * const domain,
    impose_boundary_condition_data_t * const impose_boundary_condition_data,
    id<MTLComputeCommandEncoder> encoder,
    flow_field_t * const flow_field
) {
  impose_boundary_condition_x(
      domain,
      &impose_boundary_condition_data->compute_pipeline_x,
      encoder,
      flow_field
  );
  impose_boundary_condition_y(
      domain,
      &impose_boundary_condition_data->compute_pipeline_y,
      encoder,
      flow_field
  );
  return 0;
}

