#include "shared_buffer.h"
#include "./solve_poisson_z.h"

int solve_poisson_z_data_initialize(
    id<MTLDevice> device,
    const domain_t * const domain,
    solve_poisson_z_data_t * const solve_poisson_z_data
) {
  const size_t nx = domain->x_coordinate.nitems;
  const size_t ny = domain->y_coordinate.nitems;
  const size_t nz = domain->z_coordinate.nitems;
  if (0 != compute_pipeline_initialize(
      device,
      "src/integrate/solve_poisson/solve_poisson_z.cpp",
      "solve_poisson_z",
      @{
        @"NX": @(nx),
        @"NY": @(ny),
        @"NZ": @(nz),
      },
      &solve_poisson_z_data->compute_pipeline
  )) {
    return 1;
  }
  id<MTLBuffer> * const lower_diagonals = &solve_poisson_z_data->lower_diagonals;
  id<MTLBuffer> * const upper_diagonals = &solve_poisson_z_data->upper_diagonals;
  *lower_diagonals = create_shared_buffer(device, nz * sizeof(float));
  *upper_diagonals = create_shared_buffer(device, nz * sizeof(float));
  for (size_t k = 1; k <= nz; k++) {
    float * const lower_diagonal = (float * const)[*lower_diagonals contents] + k - 1;
    float * const upper_diagonal = (float * const)[*upper_diagonals contents] + k - 1;
    // impose neumann
    if (1 == k) {
      *lower_diagonal = 0.f;
    } else {
      *lower_diagonal = domain->z_coordinate.center_laplace_operators.lower_diagonals[k];
    }
    if (nz == k) {
      *upper_diagonal = 0.f;
    } else {
      *upper_diagonal = domain->z_coordinate.center_laplace_operators.upper_diagonals[k];
    }
  }
  return 0;
}

int solve_poisson_z(
    const domain_t * const domain,
    solve_poisson_z_data_t * const solve_poisson_z_data,
    MPSCommandBuffer * const command_buffer,
    id<MTLBuffer> buf_x_eigenvalues,
    id<MTLBuffer> buf_y_eigenvalues,
    id<MTLBuffer> buf_rhs,
    id<MTLBuffer> buf_tmp
) {
  const size_t nx = domain->x_coordinate.nitems;
  const size_t ny = domain->y_coordinate.nitems;
  id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
  [encoder setComputePipelineState:solve_poisson_z_data->compute_pipeline.pipeline_state];
  [encoder setBuffer:solve_poisson_z_data->lower_diagonals offset:0 atIndex:0];
  [encoder setBuffer:solve_poisson_z_data->upper_diagonals offset:0 atIndex:1];
  [encoder setBuffer:buf_x_eigenvalues offset:0 atIndex:2];
  [encoder setBuffer:buf_y_eigenvalues offset:0 atIndex:3];
  [encoder setBuffer:buf_rhs offset:0 atIndex:4];
  [encoder setBuffer:buf_tmp offset:0 atIndex:5];
  //
  MTLSize grid_size = MTLSizeMake(nx / 2 + 1, ny, 1);
  MTLSize thread_group_size = MTLSizeMake(16, 16, 1);
  [encoder dispatchThreads:grid_size threadsPerThreadgroup:thread_group_size];
  [encoder endEncoding];
  return 0;
}

