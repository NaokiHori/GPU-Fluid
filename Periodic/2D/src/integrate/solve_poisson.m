#include <complex.h>
#include "memory.h"
#include "shared_buffer.h"
#include "./solve_poisson.h"

int solve_poisson_data_initialize(
    id<MTLDevice> device,
    const domain_t * const domain,
    id<MTLBuffer> physical_buffer,
    solve_poisson_data_t * const solve_poisson_data
) {
  const size_t nxh = domain->x_coordinate.nitems / 2 + 1;
  const size_t ny = domain->y_coordinate.nitems;
  solve_poisson_data->values = create_shared_buffer(
      device,
      nxh * ny * sizeof(float complex)
  );
  if (0 != compute_right_hand_side_initialize(
      device,
      domain,
      &solve_poisson_data->compute_right_hand_side_data
  )) {
    return 1;
  }
  if (0 != solve_poisson_x_data_initialize(
      domain,
      physical_buffer,
      solve_poisson_data->values,
      &solve_poisson_data->solve_poisson_x_data
  )) {
    return 1;
  }
  if (0 != solve_poisson_y_data_initialize(
      device,
      domain,
      &solve_poisson_data->solve_poisson_y_data
  )) {
    return 1;
  }
  // pre-compute quantities used by the kernel tridiagonal solver
  id<MTLBuffer> * const inverse_values = &solve_poisson_data->inverse_values;
  *inverse_values = create_shared_buffer(
      device,
      nxh * ny * sizeof(float)
  );
  float * const ptr_inverse_values = [*inverse_values contents];
  const double * const eigenvalues = solve_poisson_data->solve_poisson_x_data.eigenvalues;
  const double * const lower_diagonals = solve_poisson_data->solve_poisson_y_data.lower_diagonals;
  const double * const upper_diagonals = solve_poisson_data->solve_poisson_y_data.upper_diagonals;
  for (size_t i = 0; i < nxh; i++) {
    const double eigenvalue = eigenvalues[i];
    // compute the following while mitigating rounding errors
    // pivots[j] =
    //   (lowers[j] + uppers[j] + eigenvalue) - lowers[j] * uppers[j - 1] / pivots[j - 1]
    double slack = lower_diagonals[0] - eigenvalue;
    double pivot = upper_diagonals[0] + slack;
    ptr_inverse_values[0 * nxh + i] = - 1. / pivot;
    for (size_t j = 1; j < ny; j++) {
      slack = - eigenvalue + lower_diagonals[j] * (slack / pivot);
      pivot = upper_diagonals[j] + slack;
      ptr_inverse_values[j * nxh + i] = 0 == i && ny - 1 == j
        ? 0.
        : - 1. / pivot;
    }
  }
  return 0;
}

int solve_poisson(
    const domain_t * const domain,
    solve_poisson_data_t * const solve_poisson_data,
    MPSCommandBuffer * const command_buffer,
    const runge_kutta_coefficient_t * const runge_kutta_coefficient,
    id<MTLBuffer> mtl_buffer_dt,
    const flow_field_t * const flow_field,
    flow_solver_t * const flow_solver
) {
  if (0 != compute_right_hand_side(
      domain,
      &solve_poisson_data->compute_right_hand_side_data,
      command_buffer,
      runge_kutta_coefficient->implicit,
      mtl_buffer_dt,
      flow_field->ux,
      flow_field->uy,
      flow_solver->scalar_potential // use scalar_potential to store divergence
  )) {
    return 1;
  }
  if (0 != solve_poisson_x(
      &solve_poisson_data->solve_poisson_x_data,
      command_buffer,
      true
  )) {
    return 1;
  }
  if (0 != solve_poisson_y(
      domain,
      &solve_poisson_data->solve_poisson_y_data,
      command_buffer,
      solve_poisson_data->inverse_values,
      solve_poisson_data->values
  )) {
    return 1;
  }
  if (0 != solve_poisson_x(
      &solve_poisson_data->solve_poisson_x_data,
      command_buffer,
      false
  )) {
    return 1;
  }
  return 0;
}

