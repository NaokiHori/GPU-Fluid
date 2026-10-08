#include "./predict_field.h"

int predict_field_data_initialize(
    id<MTLDevice> device,
    const domain_t * const domain,
    predict_field_data_t * const predict_field_data
) {
  if (0 != integrate_field_data_initialize(
      device,
      domain,
      &predict_field_data->integrate_field_data
  )) {
    return 1;
  }
  if (0 != update_field_data_initialize(
      device,
      domain,
      &predict_field_data->update_field_data
  )) {
    return 1;
  }
  return 0;
}

int predict_field(
    const domain_t * const domain,
    impose_boundary_condition_data_t * const impose_boundary_condition_data,
    predict_field_data_t * const predict_field_data,
    id<MTLComputeCommandEncoder> encoder,
    const runge_kutta_coefficient_t * const runge_kutta_coefficient,
    id<MTLBuffer> mtl_buffer_dt,
    const float diffusivity,
    flow_field_t * const flow_field,
    flow_solver_t * const flow_solver
) {
  integrate_field(
      domain,
      &predict_field_data->integrate_field_data,
      encoder,
      runge_kutta_coefficient,
      mtl_buffer_dt,
      diffusivity,
      flow_field,
      flow_solver
  );
  update_field(
      domain,
      &predict_field_data->update_field_data,
      encoder,
      runge_kutta_coefficient->implicit,
      mtl_buffer_dt,
      diffusivity,
      flow_field,
      flow_solver
  );
  impose_boundary_condition(
      domain,
      impose_boundary_condition_data,
      encoder,
      flow_field
  );
  return 0;
}

