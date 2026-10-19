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
  solve_poisson_data->buf = create_shared_buffer(
      device,
      (domain->x_coordinate.nitems / 2 + 1) * domain->y_coordinate.nitems * sizeof(float complex)
  );
  if (0 != compute_right_hand_side_initialize(
      device,
      domain,
      &solve_poisson_data->compute_right_hand_side_data
  )) {
    return 1;
  }
  if (0 != solve_poisson_x_data_initialize(
      device,
      domain,
      physical_buffer,
      solve_poisson_data->buf,
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
  return 0;
}

int solve_poisson(
    const domain_t * const domain,
    solve_poisson_data_t * const solve_poisson_data,
    MPSCommandBuffer * const command_buffer,
    const runge_kutta_coefficient_t * const runge_kutta_coefficient,
    id<MTLBuffer> mtl_buffer_dt,
    flow_field_t * const flow_field,
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
      solve_poisson_data->solve_poisson_x_data.eigenvalues,
      solve_poisson_data->buf,
      flow_solver->scalar_potential // use scalar_potential to store TDMA working buffer
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

