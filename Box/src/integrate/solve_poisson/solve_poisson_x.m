#include "memory.h"
#include "shared_buffer.h"
#include "./solve_poisson_x.h"
#include "./solve_poisson_x/eigendecomposition.h"

static int compare_eigen_pairs(
    const void * const a,
    const void * const b
) {
  const eigen_pair_t * const eigen_pair_a = a;
  const eigen_pair_t * const eigen_pair_b = b;
  if (eigen_pair_b->eigenvalue < eigen_pair_a->eigenvalue) {
    return -1;
  } else {
    return 1;
  }
  return 0;
}

int solve_poisson_x_data_initialize(
    id<MTLDevice> device,
    const domain_t * const domain,
    solve_poisson_x_data_t * const solve_poisson_x_data
) {
  const NSUInteger nx = domain->x_coordinate.nitems;
  const NSUInteger ny = domain->y_coordinate.nitems;
  if (0 != compute_pipeline_initialize(
      device,
      "src/integrate/solve_poisson/solve_poisson_x.cpp",
      "solve_poisson_x",
      @{
        @"NX": @(nx),
        @"NY": @(ny),
      },
      &solve_poisson_x_data->compute_pipeline
  )) {
    return 1;
  }
  id<MTLBuffer> * const forward_operator = &solve_poisson_x_data->forward_operator;
  id<MTLBuffer> * const backward_operator = &solve_poisson_x_data->backward_operator;
  id<MTLBuffer> * const eigenvalues = &solve_poisson_x_data->eigenvalues;
  *forward_operator = create_shared_buffer(device, nx * nx * sizeof(float));
  *backward_operator = create_shared_buffer(device, nx * nx * sizeof(float));
  *eigenvalues = create_shared_buffer(device, nx * sizeof(float));
  // apply symmetrization and pass main/sub diagonal components to the eigendecomposition engine
  const coordinate_t * const x_coordinate = &domain->x_coordinate;
  const tridiagonal_t * const center_laplace_operators = &x_coordinate->center_laplace_operators;
  const double * const face_scale_factors = x_coordinate->face_scale_factors;
  const double * const center_scale_factors = x_coordinate->center_scale_factors;
  double * const main_diagonals = memory_alloc(nx, sizeof(double));
  double * const sub_diagonals = memory_alloc(nx - 1, sizeof(double));
  double * const eigenvectors_transposed = memory_alloc(nx * nx, sizeof(double));
  for (size_t n = 1; n <= nx; n++) {
    double * const main_diagonal = main_diagonals + n - 1;
    *main_diagonal = center_laplace_operators->main_diagonals[n];
    // impose neumann boundary conditions
    if (1 == n) {
      *main_diagonal += center_laplace_operators->lower_diagonals[n];
    }
    if (nx == n) {
      *main_diagonal += center_laplace_operators->upper_diagonals[n];
    }
  }
  for (size_t n = 2; n < nx + 1; n++) {
    sub_diagonals[n - 2] = 1.
      / sqrt(center_scale_factors[n - 1])
      / sqrt(center_scale_factors[n])
      / face_scale_factors[n];
  }
  // find eigenvalues / eigenvectors
  double * const d_eigenvalues = memory_alloc(nx, sizeof(double));
  if (0 != compute_eigendecomposition(
      8,
      nx,
      main_diagonals,
      sub_diagonals,
      d_eigenvalues,
      eigenvectors_transposed
  )) {
    return 1;
  }
  if (0 != sort_eigendecomposition(nx, compare_eigen_pairs, d_eigenvalues, eigenvectors_transposed)) {
  }
  float * const ptr_eigenvalues = [*eigenvalues contents];
  for (size_t n = 0; n < nx; n++) {
    ptr_eigenvalues[n] = d_eigenvalues[n];
  }
  memory_free(d_eigenvalues);
  // apply symmetrization operator to find forward / backward operators (between physical and spectral domains)
  // NOTE: notice the index difference (scale factors are defined on the domain whereas operators are pure matrices not recognizing the domain)
  float * const ptr_forward_operator = [*forward_operator contents];
  float * const ptr_backward_operator = [*backward_operator contents];
  for (size_t i = 0; i < nx; i++) {
    for (size_t j = 0; j < nx; j++) {
      ptr_forward_operator[i * nx + j] = eigenvectors_transposed[i * nx + j] * sqrt(center_scale_factors[j + 1]);
    }
  }
  for (size_t i = 0; i < nx; i++) {
    for (size_t j = 0; j < nx; j++) {
      ptr_backward_operator[i * nx + j] = 1. / sqrt(center_scale_factors[i + 1]) * eigenvectors_transposed[j * nx + i];
    }
  }
  // clean-up
  memory_free(main_diagonals);
  memory_free(sub_diagonals);
  memory_free(eigenvectors_transposed);
  return 0;
}

int solve_poisson_x(
    const domain_t * const domain,
    solve_poisson_x_data_t * const solve_poisson_x_data,
    id<MTLComputeCommandEncoder> encoder,
    id<MTLBuffer> operator,
    id<MTLBuffer> in,
    id<MTLBuffer> out
) {
  const size_t nx = domain->x_coordinate.nitems;
  const size_t ny = domain->y_coordinate.nitems;
  [encoder setComputePipelineState:solve_poisson_x_data->compute_pipeline.pipeline_state];
  [encoder setBuffer:operator offset:0 atIndex:0];
  [encoder setBuffer:in offset:0 atIndex:1];
  [encoder setBuffer:out offset:0 atIndex:2];
  //
  MTLSize grid_size = MTLSizeMake(nx, ny, 1);
  MTLSize thread_group_size = MTLSizeMake(16, 16, 1);
  [encoder dispatchThreads:grid_size threadsPerThreadgroup:thread_group_size];
  return 0;
}

