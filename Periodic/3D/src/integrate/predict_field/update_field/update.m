#include "./update.h"

int update_data_initialize(
    id<MTLDevice> device,
    const domain_t * const domain,
    update_data_t * const update_data
) {
  const size_t nx = domain->x_coordinate.nitems;
  const size_t ny = domain->y_coordinate.nitems;
  const size_t nz = domain->z_coordinate.nitems;
  if (0 != compute_pipeline_initialize(
      device,
      "src/integrate/predict_field/update_field/update.cpp",
      "update",
      @{
        @"NX": @(nx),
        @"NY": @(ny),
        @"NZ": @(nz),
      },
      &update_data->compute_pipeline
  )) {
    return 1;
  }
  return 0;
}

int update(
    const domain_t * const domain,
    update_data_t * const update_data,
    id<MTLComputeCommandEncoder> encoder,
    flow_field_t * const flow_field,
    flow_solver_t * const flow_solver
) {
  [encoder setComputePipelineState:update_data->compute_pipeline.pipeline_state];
  [encoder setBuffer:flow_solver->dux offset:0 atIndex:0];
  [encoder setBuffer:flow_solver->duy offset:0 atIndex:1];
  [encoder setBuffer:flow_solver->duz offset:0 atIndex:2];
  [encoder setBuffer:flow_solver->dtemperature offset:0 atIndex:3];
  [encoder setBuffer:flow_field->ux offset:0 atIndex:4];
  [encoder setBuffer:flow_field->uy offset:0 atIndex:5];
  [encoder setBuffer:flow_field->uz offset:0 atIndex:6];
  [encoder setBuffer:flow_field->temperature offset:0 atIndex:7];
  //
  MTLSize grid_size = MTLSizeMake(
      domain->x_coordinate.nitems,
      domain->y_coordinate.nitems,
      domain->z_coordinate.nitems
  );
  MTLSize thread_group_size = MTLSizeMake(8, 8, 4);
  [encoder dispatchThreads:grid_size threadsPerThreadgroup:thread_group_size];
  return 0;
}

