#include "memory.h"
#include "shared_buffer.h"
#include "./solve_poisson_y.h"

int solve_poisson_y_data_initialize(
    id<MTLDevice> device,
    const domain_t * const domain,
    solve_poisson_y_data_t * const solve_poisson_y_data
) {
  const size_t nx = domain->x_coordinate.nitems;
  const size_t ny = domain->y_coordinate.nitems;
  if (0 != compute_pipeline_initialize(
      device,
      "src/integrate/solve_poisson/solve_poisson_y.cpp",
      "solve_poisson_y",
      @{
        @"NX": @(nx),
        @"NY": @(ny),
      },
      &solve_poisson_y_data->compute_pipeline
  )) {
    return 1;
  }
  id<MTLBuffer> * const mtl_buffer_lower_diagonals = &solve_poisson_y_data->mtl_buffer_lower_diagonals;
  id<MTLBuffer> * const mtl_buffer_upper_diagonals = &solve_poisson_y_data->mtl_buffer_upper_diagonals;
  *mtl_buffer_lower_diagonals = create_shared_buffer(device, ny * sizeof(float));
  *mtl_buffer_upper_diagonals = create_shared_buffer(device, ny * sizeof(float));
  float * const ptr_mtl_buffer_lower_diagonals = [*mtl_buffer_lower_diagonals contents];
  float * const ptr_mtl_buffer_upper_diagonals = [*mtl_buffer_upper_diagonals contents];
  double ** const lower_diagonals = &solve_poisson_y_data->lower_diagonals;
  double ** const upper_diagonals = &solve_poisson_y_data->upper_diagonals;
  *lower_diagonals = memory_alloc(ny, sizeof(double));
  *upper_diagonals = memory_alloc(ny, sizeof(double));
  for (size_t j = 1; j <= ny; j++) {
    double * const lower_diagonal = *lower_diagonals + j - 1;
    double * const upper_diagonal = *upper_diagonals + j - 1;
    // impose neumann
    if (1 == j) {
      *lower_diagonal = 0.;
    } else {
      *lower_diagonal = domain->y_coordinate.center_laplace_operators.lower_diagonals[j];
    }
    if (ny == j) {
      *upper_diagonal = 0.;
    } else {
      *upper_diagonal = domain->y_coordinate.center_laplace_operators.upper_diagonals[j];
    }
    ptr_mtl_buffer_lower_diagonals[j - 1] = *lower_diagonal;
    ptr_mtl_buffer_upper_diagonals[j - 1] = *upper_diagonal;
  }
  return 0;
}

int solve_poisson_y(
    const domain_t * const domain,
    solve_poisson_y_data_t * const solve_poisson_y_data,
    MPSCommandBuffer * const command_buffer,
    id<MTLBuffer> mtl_buffer_inverse_values,
    id<MTLBuffer> mtl_buffer_values
) {
  const size_t nx = domain->x_coordinate.nitems;
  const size_t nxh = nx / 2 + 1;
  id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
  [encoder setComputePipelineState:solve_poisson_y_data->compute_pipeline.pipeline_state];
  [encoder setBuffer:solve_poisson_y_data->mtl_buffer_lower_diagonals offset:0 atIndex:0];
  [encoder setBuffer:solve_poisson_y_data->mtl_buffer_upper_diagonals offset:0 atIndex:1];
  [encoder setBuffer:mtl_buffer_inverse_values offset:0 atIndex:2];
  [encoder setBuffer:mtl_buffer_values offset:0 atIndex:3];
  //
  MTLSize grid_size = MTLSizeMake(nxh, 1, 1);
  MTLSize thread_group_size = MTLSizeMake(128, 1, 1);
  [encoder dispatchThreads:grid_size threadsPerThreadgroup:thread_group_size];
  [encoder endEncoding];
  return 0;
}

