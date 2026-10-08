#include "shared_buffer.h"
#include "./solve_implicit_y.h"

int solve_implicit_y_data_initialize(
    id<MTLDevice> device,
    const domain_t * const domain,
    solve_implicit_y_data_t * const solve_implicit_y_data
) {
  const size_t nx = domain->x_coordinate.nitems;
  const size_t ny = domain->y_coordinate.nitems;
  if (0 != compute_pipeline_initialize(
      device,
      "src/integrate/predict_field/update_field/solve_implicit_y.cpp",
      "solve_implicit_y",
      @{
        @"NX": @(nx),
        @"NY": @(ny),
      },
      &solve_implicit_y_data->compute_pipeline
  )) {
    return 1;
  }
  id<MTLBuffer> * const ux_lower_diagonals = &solve_implicit_y_data->ux_lower_diagonals;
  id<MTLBuffer> * const ux_upper_diagonals = &solve_implicit_y_data->ux_upper_diagonals;
  id<MTLBuffer> * const uy_lower_diagonals = &solve_implicit_y_data->uy_lower_diagonals;
  id<MTLBuffer> * const uy_upper_diagonals = &solve_implicit_y_data->uy_upper_diagonals;
  id<MTLBuffer> * const temperature_lower_diagonals = &solve_implicit_y_data->temperature_lower_diagonals;
  id<MTLBuffer> * const temperature_upper_diagonals = &solve_implicit_y_data->temperature_upper_diagonals;
  *ux_lower_diagonals = create_shared_buffer(device, ny * sizeof(float));
  *ux_upper_diagonals = create_shared_buffer(device, ny * sizeof(float));
  *uy_lower_diagonals = create_shared_buffer(device, (ny - 1) * sizeof(float));
  *uy_upper_diagonals = create_shared_buffer(device, (ny - 1) * sizeof(float));
  *temperature_lower_diagonals = create_shared_buffer(device, ny * sizeof(float));
  *temperature_upper_diagonals = create_shared_buffer(device, ny * sizeof(float));
  const tridiagonal_t * const center_laplace_operators = &domain->y_coordinate.center_laplace_operators;
  const tridiagonal_t * const face_laplace_operators = &domain->y_coordinate.face_laplace_operators;
  // dirichlet
  float * const ptr_ux_lower_diagonals = [*ux_lower_diagonals contents];
  float * const ptr_ux_upper_diagonals = [*ux_upper_diagonals contents];
  for (size_t j = 1; j <= ny; j++) {
    ptr_ux_lower_diagonals[j - 1] = center_laplace_operators->lower_diagonals[j];
    ptr_ux_upper_diagonals[j - 1] = center_laplace_operators->upper_diagonals[j];
  }
  // dirichlet
  float * const ptr_uy_lower_diagonals = [*uy_lower_diagonals contents];
  float * const ptr_uy_upper_diagonals = [*uy_upper_diagonals contents];
  for (size_t j = 2; j <= ny; j++) {
    ptr_uy_lower_diagonals[j - 2] = face_laplace_operators->lower_diagonals[j];
    ptr_uy_upper_diagonals[j - 2] = face_laplace_operators->upper_diagonals[j];
  }
  // dirichlet
  float * const ptr_temperature_lower_diagonals = [*temperature_lower_diagonals contents];
  float * const ptr_temperature_upper_diagonals = [*temperature_upper_diagonals contents];
  for (size_t j = 1; j <= ny; j++) {
    ptr_temperature_lower_diagonals[j - 1] = center_laplace_operators->lower_diagonals[j];
    ptr_temperature_upper_diagonals[j - 1] = center_laplace_operators->upper_diagonals[j];
  }
  return 0;
}

int solve_implicit_y(
    const domain_t * const domain,
    solve_implicit_y_data_t * const solve_implicit_y_data,
    id<MTLComputeCommandEncoder> encoder,
    id<MTLBuffer> mtl_buffer_dt,
    const float off_diagonal_prefactor,
    flow_solver_t * const flow_solver
) {
  const NSUInteger nx = domain->x_coordinate.nitems;
  [encoder setComputePipelineState:solve_implicit_y_data->compute_pipeline.pipeline_state];
  [encoder setBytes:&off_diagonal_prefactor length:sizeof(float) atIndex:0];
  [encoder setBuffer:mtl_buffer_dt offset:0 atIndex:1];
  [encoder setBuffer:solve_implicit_y_data->ux_lower_diagonals offset:0 atIndex:2];
  [encoder setBuffer:solve_implicit_y_data->ux_upper_diagonals offset:0 atIndex:3];
  [encoder setBuffer:solve_implicit_y_data->uy_lower_diagonals offset:0 atIndex:4];
  [encoder setBuffer:solve_implicit_y_data->uy_upper_diagonals offset:0 atIndex:5];
  [encoder setBuffer:solve_implicit_y_data->temperature_lower_diagonals offset:0 atIndex:6];
  [encoder setBuffer:solve_implicit_y_data->temperature_upper_diagonals offset:0 atIndex:7];
  [encoder setBuffer:flow_solver->dux offset:0 atIndex:8];
  [encoder setBuffer:flow_solver->duy offset:0 atIndex:9];
  [encoder setBuffer:flow_solver->dtemperature offset:0 atIndex:10];
  [encoder setBuffer:flow_solver->scalar_potential offset:0 atIndex:11]; // NOTE: used as a scratchpad
  //
  MTLSize grid_size = MTLSizeMake(nx, 1, 1);
  NSUInteger max_threads = [solve_implicit_y_data->compute_pipeline.pipeline_state maxTotalThreadsPerThreadgroup];
  NSUInteger thread_group_size_value = MIN(nx, max_threads);
  MTLSize thread_group_size = MTLSizeMake(thread_group_size_value, 1, 1);
  [encoder dispatchThreads:grid_size threadsPerThreadgroup:thread_group_size];
  return 0;
}

