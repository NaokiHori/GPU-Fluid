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
  id<MTLBuffer> * const lower_diagonals = &solve_poisson_y_data->lower_diagonals;
  id<MTLBuffer> * const upper_diagonals = &solve_poisson_y_data->upper_diagonals;
  *lower_diagonals = create_shared_buffer(device, ny * sizeof(float));
  *upper_diagonals = create_shared_buffer(device, ny * sizeof(float));
  for (size_t j = 1; j <= ny; j++) {
    float * const lower_diagonal = (float * const)[*lower_diagonals contents] + j - 1;
    float * const upper_diagonal = (float * const)[*upper_diagonals contents] + j - 1;
    // impose neumann
    if (1 == j) {
      *lower_diagonal = 0.f;
    } else {
      *lower_diagonal = domain->y_coordinate.center_laplace_operators.lower_diagonals[j];
    }
    if (ny == j) {
      *upper_diagonal = 0.f;
    } else {
      *upper_diagonal = domain->y_coordinate.center_laplace_operators.upper_diagonals[j];
    }
  }
  return 0;
}

int solve_poisson_y(
    const domain_t * const domain,
    solve_poisson_y_data_t * const solve_poisson_y_data,
    MPSCommandBuffer * const command_buffer,
    id<MTLBuffer> buf_eigenvalues,
    id<MTLBuffer> buf_rhs,
    id<MTLBuffer> buf_tmp
) {
  const size_t nx = domain->x_coordinate.nitems;
  const size_t nxh = nx / 2 + 1;
  id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
  [encoder setComputePipelineState:solve_poisson_y_data->compute_pipeline.pipeline_state];
  [encoder setBuffer:solve_poisson_y_data->lower_diagonals offset:0 atIndex:0];
  [encoder setBuffer:solve_poisson_y_data->upper_diagonals offset:0 atIndex:1];
  [encoder setBuffer:buf_eigenvalues offset:0 atIndex:2];
  [encoder setBuffer:buf_rhs offset:0 atIndex:3];
  [encoder setBuffer:buf_tmp offset:0 atIndex:4];
  //
  MTLSize grid_size = MTLSizeMake(nxh, 1, 1);
  MTLSize thread_group_size = MTLSizeMake(128, 1, 1);
  [encoder dispatchThreads:grid_size threadsPerThreadgroup:thread_group_size];
  [encoder endEncoding];
  return 0;
}

