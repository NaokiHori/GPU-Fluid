#include "shared_buffer.h"
#include "./solve_implicit_x.h"

int solve_implicit_x_data_initialize(
    id<MTLDevice> device,
    const domain_t * const domain,
    solve_implicit_x_data_t * const solve_implicit_x_data
) {
  const size_t nx = domain->x_coordinate.nitems;
  const size_t ny = domain->y_coordinate.nitems;
  if (0 != compute_pipeline_initialize(
      device,
      "src/integrate/predict_field/update_field/solve_implicit_x.cpp",
      "solve_implicit_x",
      @{
        @"NX": @(nx),
        @"NY": @(ny),
      },
      &solve_implicit_x_data->compute_pipeline
  )) {
    return 1;
  }
  id<MTLBuffer> * const ux_lower_diagonals = &solve_implicit_x_data->ux_lower_diagonals;
  id<MTLBuffer> * const ux_upper_diagonals = &solve_implicit_x_data->ux_upper_diagonals;
  id<MTLBuffer> * const uy_lower_diagonals = &solve_implicit_x_data->uy_lower_diagonals;
  id<MTLBuffer> * const uy_upper_diagonals = &solve_implicit_x_data->uy_upper_diagonals;
  id<MTLBuffer> * const temperature_lower_diagonals = &solve_implicit_x_data->temperature_lower_diagonals;
  id<MTLBuffer> * const temperature_upper_diagonals = &solve_implicit_x_data->temperature_upper_diagonals;
  *ux_lower_diagonals = create_shared_buffer(device, (nx - 1) * sizeof(float));
  *ux_upper_diagonals = create_shared_buffer(device, (nx - 1) * sizeof(float));
  *uy_lower_diagonals = create_shared_buffer(device, nx * sizeof(float));
  *uy_upper_diagonals = create_shared_buffer(device, nx * sizeof(float));
  *temperature_lower_diagonals = create_shared_buffer(device, nx * sizeof(float));
  *temperature_upper_diagonals = create_shared_buffer(device, nx * sizeof(float));
  const tridiagonal_t * const center_laplace_operators = &domain->x_coordinate.center_laplace_operators;
  const tridiagonal_t * const face_laplace_operators = &domain->x_coordinate.face_laplace_operators;
  // dirichlet
  float * const ptr_ux_lower_diagonals = [*ux_lower_diagonals contents];
  float * const ptr_ux_upper_diagonals = [*ux_upper_diagonals contents];
  for (size_t i = 2; i <= nx; i++) {
    ptr_ux_lower_diagonals[i - 2] = face_laplace_operators->lower_diagonals[i];
    ptr_ux_upper_diagonals[i - 2] = face_laplace_operators->upper_diagonals[i];
  }
  // dirichlet
  float * const ptr_uy_lower_diagonals = [*uy_lower_diagonals contents];
  float * const ptr_uy_upper_diagonals = [*uy_upper_diagonals contents];
  for (size_t i = 1; i <= nx; i++) {
    ptr_uy_lower_diagonals[i - 1] = center_laplace_operators->lower_diagonals[i];
    ptr_uy_upper_diagonals[i - 1] = center_laplace_operators->upper_diagonals[i];
  }
  // neumann
  float * const ptr_temperature_lower_diagonals = [*temperature_lower_diagonals contents];
  float * const ptr_temperature_upper_diagonals = [*temperature_upper_diagonals contents];
  for (size_t i = 1; i <= nx; i++) {
    ptr_temperature_lower_diagonals[i - 1] = 1 == i ? 0. : center_laplace_operators->lower_diagonals[i];
    ptr_temperature_upper_diagonals[i - 1] = nx == i ? 0. : center_laplace_operators->upper_diagonals[i];
  }
  return 0;
}

int solve_implicit_x(
    const domain_t * const domain,
    solve_implicit_x_data_t * const solve_implicit_x_data,
    id<MTLComputeCommandEncoder> encoder,
    id<MTLBuffer> mtl_buffer_dt,
    const float off_diagonal_prefactor,
    flow_solver_t * const flow_solver
) {
  const NSUInteger ny = domain->y_coordinate.nitems;
  [encoder setComputePipelineState:solve_implicit_x_data->compute_pipeline.pipeline_state];
  [encoder setBytes:&off_diagonal_prefactor length:sizeof(float) atIndex:0];
  [encoder setBuffer:mtl_buffer_dt offset:0 atIndex:1];
  [encoder setBuffer:solve_implicit_x_data->ux_lower_diagonals offset:0 atIndex:2];
  [encoder setBuffer:solve_implicit_x_data->ux_upper_diagonals offset:0 atIndex:3];
  [encoder setBuffer:solve_implicit_x_data->uy_lower_diagonals offset:0 atIndex:4];
  [encoder setBuffer:solve_implicit_x_data->uy_upper_diagonals offset:0 atIndex:5];
  [encoder setBuffer:solve_implicit_x_data->temperature_lower_diagonals offset:0 atIndex:6];
  [encoder setBuffer:solve_implicit_x_data->temperature_upper_diagonals offset:0 atIndex:7];
  [encoder setBuffer:flow_solver->dux offset:0 atIndex:8];
  [encoder setBuffer:flow_solver->duy offset:0 atIndex:9];
  [encoder setBuffer:flow_solver->dtemperature offset:0 atIndex:10];
  [encoder setBuffer:flow_solver->scalar_potential offset:0 atIndex:11]; // NOTE: used as a scratchpad
  //
  MTLSize grid_size = MTLSizeMake(ny, 1, 1);
  NSUInteger max_threads = [solve_implicit_x_data->compute_pipeline.pipeline_state maxTotalThreadsPerThreadgroup];
  NSUInteger thread_group_size_value = MIN(ny, max_threads);
  MTLSize thread_group_size = MTLSizeMake(thread_group_size_value, 1, 1);
  [encoder dispatchThreads:grid_size threadsPerThreadgroup:thread_group_size];
  return 0;
}
