#include "./integrate_field.h"

int integrate_field_data_initialize(
    id<MTLDevice> device,
    const domain_t * const domain,
    integrate_field_data_t * const integrate_field_data
) {
  const NSUInteger nx = domain->x_coordinate.nitems;
  const NSUInteger ny = domain->y_coordinate.nitems;
  const double lx = domain->x_coordinate.length;
  if (0 != compute_pipeline_initialize(
      device,
      "src/integrate/predict_field/integrate_field.cpp",
      "integrate_field",
      @{
        @"NX": @(nx),
        @"NY": @(ny),
        @"HX": @(lx / nx),
        @"SIZE_OF_RUNGE_KUTTA_COEFFICIENT_T": @(sizeof(runge_kutta_coefficient_t)),
        @"OFFSET_OF_RUNGE_KUTTA_COEFFICIENT_T_EXPLICIT_PREVIOUS": @(
            offsetof(runge_kutta_coefficient_t, explicit_previous)
        ),
        @"OFFSET_OF_RUNGE_KUTTA_COEFFICIENT_T_EXPLICIT_CURRENT": @(
            offsetof(runge_kutta_coefficient_t, explicit_current)
        ),
        @"OFFSET_OF_RUNGE_KUTTA_COEFFICIENT_T_IMPLICIT": @(
            offsetof(runge_kutta_coefficient_t, implicit)
        ),
        @"SIZE_OF_METRIC_T": @(sizeof(metric_t)),
        @"OFFSET_OF_METRIC_T_SCALE_FACTOR": @(offsetof(metric_t, scale_factor)),
        @"OFFSET_OF_METRIC_T_LOWER_DIAGONAL": @(offsetof(metric_t, lower_diagonal)),
        @"OFFSET_OF_METRIC_T_MAIN_DIAGONAL": @(offsetof(metric_t, main_diagonal)),
        @"OFFSET_OF_METRIC_T_UPPER_DIAGONAL": @(offsetof(metric_t, upper_diagonal)),
      },
      &integrate_field_data->compute_pipeline
  )) {
    return 1;
  }
  return 0;
}

int integrate_field(
    const domain_t * const domain,
    integrate_field_data_t * const integrate_field_data,
    id<MTLComputeCommandEncoder> encoder,
    const runge_kutta_coefficient_t * const runge_kutta_coefficient,
    id<MTLBuffer> mtl_buffer_dt,
    const float diffusivity,
    flow_field_t * const flow_field,
    flow_solver_t * const flow_solver
) {
  [encoder setComputePipelineState:integrate_field_data->compute_pipeline.pipeline_state];
  [encoder setBytes:runge_kutta_coefficient length:sizeof(runge_kutta_coefficient_t) atIndex:0];
  [encoder setBuffer:mtl_buffer_dt offset:0 atIndex:1];
  [encoder setBytes:&diffusivity length:sizeof(float) atIndex:2];
  [encoder setBuffer:domain->y_coordinate.mtl_buffer_center_metrics offset:0 atIndex:3];
  [encoder setBuffer:domain->y_coordinate.mtl_buffer_face_metrics offset:0 atIndex:4];
  [encoder setBuffer:flow_field->ux offset:0 atIndex:5];
  [encoder setBuffer:flow_field->uy offset:0 atIndex:6];
  [encoder setBuffer:flow_field->p offset:0 atIndex:7];
  [encoder setBuffer:flow_field->temperature offset:0 atIndex:8];
  [encoder setBuffer:flow_solver->dux offset:0 atIndex:9];
  [encoder setBuffer:flow_solver->duy offset:0 atIndex:10];
  [encoder setBuffer:flow_solver->dtemperature offset:0 atIndex:11];
  [encoder setBuffer:flow_solver->dux_explicit_previous offset:0 atIndex:12];
  [encoder setBuffer:flow_solver->duy_explicit_previous offset:0 atIndex:13];
  [encoder setBuffer:flow_solver->dtemperature_explicit_previous offset:0 atIndex:14];
  //
  MTLSize grid_size = MTLSizeMake(domain->x_coordinate.nitems + 2, domain->y_coordinate.nitems + 2, 1);
  MTLSize thread_group_size = MTLSizeMake(16, 16, 1);
  [encoder dispatchThreads:grid_size threadsPerThreadgroup:thread_group_size];
  return 0;
}

