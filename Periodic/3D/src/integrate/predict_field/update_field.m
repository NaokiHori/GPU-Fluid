#include "shared_buffer.h"
#include "./update_field.h"

int update_field_data_initialize(
    id<MTLDevice> device,
    const domain_t * const domain,
    update_field_data_t * const update_field_data
) {
  if (0 != solve_implicit_z_data_initialize(
      device,
      domain,
      &update_field_data->solve_implicit_z_data
  )) {
    return 1;
  }
  if (0 != update_data_initialize(
      device,
      domain,
      &update_field_data->update_data
  )) {
    return 1;
  }
  return 0;
}

int update_field(
    const domain_t * const domain,
    update_field_data_t * const update_field_data,
    id<MTLComputeCommandEncoder> encoder,
    const float runge_kutta_coefficient_implicit,
    id<MTLBuffer> mtl_buffer_dt,
    const float diffusivity,
    flow_field_t * const flow_field,
    flow_solver_t * const flow_solver
) {
  const float off_diagonal_prefactor = - 0.5f * diffusivity * runge_kutta_coefficient_implicit;
  solve_implicit_z(
      domain,
      &update_field_data->solve_implicit_z_data,
      encoder,
      mtl_buffer_dt,
      off_diagonal_prefactor,
      flow_solver
  );
  update(
      domain,
      &update_field_data->update_data,
      encoder,
      flow_field,
      flow_solver
  );
  return 0;
}

