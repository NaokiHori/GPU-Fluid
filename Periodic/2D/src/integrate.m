#include "runge_kutta.h"
#include "impose_boundary_condition.h"
#include "./integrate.h"

int integrate_data_initialize(
    id<MTLDevice> device,
    const domain_t * const domain,
    id<MTLBuffer> physical_buffer,
    const size_t max_batch_size,
    integrate_data_t * const integrate_data
) {
  correct_field_data_t * const correct_field_data = &integrate_data->correct_field_data;
  decide_time_step_size_data_t * const decide_time_step_size_data = &integrate_data->decide_time_step_size_data;
  predict_field_data_t * const predict_field_data = &integrate_data->predict_field_data;
  solve_poisson_data_t * const solve_poisson_data = &integrate_data->solve_poisson_data;
  if (0 != correct_field_data_initialize(device, domain, correct_field_data)) {
    return 1;
  }
  if (0 != decide_time_step_size_data_initialize(
      device,
      domain,
      max_batch_size,
      decide_time_step_size_data
  )) {
    return 1;
  }
  if (0 != predict_field_data_initialize(device, domain, predict_field_data)) {
    return 1;
  }
  if (0 != solve_poisson_data_initialize(device, domain, physical_buffer, solve_poisson_data)) {
    return 1;
  }
  return 0;
}

int integrate(
    const size_t batch_size,
    const domain_t * const domain,
    impose_boundary_condition_data_t * const impose_boundary_condition_data,
    integrate_data_t * const integrate_data,
    id<MTLCommandQueue> queue,
    const float diffusivity,
    flow_field_t * const flow_field,
    flow_solver_t * const flow_solver,
    size_t * const step,
    double * const time
) {
  const runge_kutta_coefficient_t runge_kutta_coefficients[N_RUNGE_KUTTA_STEPS] = {
    {
      .explicit_previous = 0. / 60.,
      .explicit_current = 32. / 60.,
      .implicit = 32. / 60.,
    },
    {
      .explicit_previous = - 17. / 60.,
      .explicit_current = 25. / 60.,
      .implicit = 8. / 60.,
    },
    {
      .explicit_previous = - 25. / 60.,
      .explicit_current = 45. / 60.,
      .implicit = 20. / 60.,
    },
  };
  MPSCommandBuffer * const command_buffer = [MPSCommandBuffer commandBufferFromCommandQueue:queue];
  for (size_t batch = 0; batch < batch_size; batch++) {
    for (size_t runge_kutta_step = 0; runge_kutta_step < N_RUNGE_KUTTA_STEPS; runge_kutta_step++) {
      id<MTLBuffer> mtl_buffer_dt = integrate_data->decide_time_step_size_data.mtl_buffer_dt;
      {
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (0 == runge_kutta_step) {
          decide_time_step_size(
              domain,
              &integrate_data->decide_time_step_size_data,
              encoder,
              flow_field,
              batch
          );
        }
        predict_field(
            domain,
            impose_boundary_condition_data,
            &integrate_data->predict_field_data,
            encoder,
            runge_kutta_coefficients + runge_kutta_step,
            mtl_buffer_dt,
            diffusivity,
            flow_field,
            flow_solver
        );
        [encoder endEncoding];
      }
      solve_poisson(
          domain,
          &integrate_data->solve_poisson_data,
          command_buffer,
          runge_kutta_coefficients + runge_kutta_step,
          mtl_buffer_dt,
          flow_field,
          flow_solver
      );
      {
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        correct_field(
            domain,
            &integrate_data->correct_field_data,
            encoder,
            runge_kutta_coefficients + runge_kutta_step,
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
        [encoder endEncoding];
      }
    }
  }
  [command_buffer commit];
  [command_buffer waitUntilCompleted];
  update_time(&integrate_data->decide_time_step_size_data, batch_size, step, time);
  return 0;
}

