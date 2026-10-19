#include "./correct_field.h"

int correct_field_data_initialize(
    id<MTLDevice> device,
    const domain_t * const domain,
    correct_field_data_t * const correct_field_data
) {
  const NSUInteger nx = domain->x_coordinate.nitems;
  const NSUInteger ny = domain->y_coordinate.nitems;
  const double lx = domain->x_coordinate.length;
  if (0 != compute_pipeline_initialize(
      device,
      "src/integrate/correct_field.cpp",
      "correct_field",
      @{
        @"NX": @(nx),
        @"NY": @(ny),
        @"HX": @(lx / nx),
        @"SIZE_OF_METRIC_T": @(sizeof(metric_t)),
        @"OFFSET_OF_METRIC_T_SCALE_FACTOR": @(offsetof(metric_t, scale_factor)),
        @"OFFSET_OF_METRIC_T_LOWER_DIAGONAL": @(offsetof(metric_t, lower_diagonal)),
        @"OFFSET_OF_METRIC_T_MAIN_DIAGONAL": @(offsetof(metric_t, main_diagonal)),
        @"OFFSET_OF_METRIC_T_UPPER_DIAGONAL": @(offsetof(metric_t, upper_diagonal)),
      },
      &correct_field_data->compute_pipeline
  )) {
    return 1;
  }
  return 0;
}

int correct_field(
    const domain_t * const domain,
    correct_field_data_t * const correct_field_data,
    id<MTLComputeCommandEncoder> encoder,
    const runge_kutta_coefficient_t * const runge_kutta_coefficient,
    id<MTLBuffer> mtl_buffer_dt,
    const float diffusivity,
    flow_field_t * const flow_field,
    flow_solver_t * const flow_solver
) {
  [encoder setComputePipelineState:correct_field_data->compute_pipeline.pipeline_state];
  [encoder setBytes:&runge_kutta_coefficient->implicit length:sizeof(float) atIndex:0];
  [encoder setBytes:&diffusivity length:sizeof(float) atIndex:1];
  [encoder setBuffer:mtl_buffer_dt offset:0 atIndex:2];
  [encoder setBuffer:domain->y_coordinate.mtl_buffer_face_scale_factors offset:0 atIndex:3];
  [encoder setBuffer:domain->y_coordinate.mtl_buffer_center_metrics offset:0 atIndex:4];
  [encoder setBuffer:flow_solver->scalar_potential offset:0 atIndex:5];
  [encoder setBuffer:flow_field->ux offset:0 atIndex:6];
  [encoder setBuffer:flow_field->uy offset:0 atIndex:7];
  [encoder setBuffer:flow_field->p offset:0 atIndex:8];
  //
  MTLSize grid_size = MTLSizeMake(domain->x_coordinate.nitems + 2, domain->y_coordinate.nitems + 2, 1);
  MTLSize thread_group_size = MTLSizeMake(16, 16, 1);
  [encoder dispatchThreads:grid_size threadsPerThreadgroup:thread_group_size];
  return 0;
}

