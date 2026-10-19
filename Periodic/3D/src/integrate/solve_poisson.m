#include <complex.h>
#include "memory.h"
#include "shared_buffer.h"
#include "./solve_poisson.h"

int solve_poisson_data_initialize(
    id<MTLDevice> device,
    const domain_t * const domain,
    id<MTLBuffer> buffer,
    solve_poisson_data_t * const solve_poisson_data
) {
  const size_t nx = domain->x_coordinate.nitems;
  const size_t ny = domain->y_coordinate.nitems;
  const size_t nz = domain->z_coordinate.nitems;
  MPSGraph ** const graph = &solve_poisson_data->graph;
  *graph = [MPSGraph new];
  solve_poisson_data->buf = create_shared_buffer(
      device,
      (nx / 2 + 1) * ny * nz * sizeof(float complex)
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
      *graph,
      domain,
      buffer,
      solve_poisson_data->buf,
      &solve_poisson_data->solve_poisson_x_data
  )) {
    return 1;
  }
  if (0 != solve_poisson_y_data_initialize(
      device,
      *graph,
      domain,
      solve_poisson_data->buf,
      buffer,
      &solve_poisson_data->solve_poisson_y_data
  )) {
    return 1;
  }
  if (0 != solve_poisson_z_data_initialize(
      device,
      domain,
      &solve_poisson_data->solve_poisson_z_data
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
      flow_field->uz,
      flow_solver->scalar_potential // use scalar_potential to store divergence
  )) {
    return 1;
  }
  if (0 != solve_poisson_x(
      solve_poisson_data->graph,
      &solve_poisson_data->solve_poisson_x_data,
      command_buffer,
      true
  )) {
    return 1;
  }
  if (0 != solve_poisson_y(
      solve_poisson_data->graph,
      &solve_poisson_data->solve_poisson_y_data,
      command_buffer,
      true
  )) {
    return 1;
  }
  if (0 != solve_poisson_z(
      domain,
      &solve_poisson_data->solve_poisson_z_data,
      command_buffer,
      solve_poisson_data->solve_poisson_x_data.eigenvalues,
      solve_poisson_data->solve_poisson_y_data.eigenvalues,
      flow_solver->scalar_potential,
      solve_poisson_data->buf
  )) {
    return 1;
  }
  if (0 != solve_poisson_y(
      solve_poisson_data->graph,
      &solve_poisson_data->solve_poisson_y_data,
      command_buffer,
      false
  )) {
    return 1;
  }
  if (0 != solve_poisson_x(
      solve_poisson_data->graph,
      &solve_poisson_data->solve_poisson_x_data,
      command_buffer,
      false
  )) {
    return 1;
  }
  return 0;
}

