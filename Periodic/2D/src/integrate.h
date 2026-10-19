#if !defined(INTEGRATE_H)
#define INTEGRATE_H

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "compute_pipeline.h"
#include "domain.h"
#include "flow_field.h"
#include "flow_solver.h"
#include "impose_boundary_condition.h"
#include "./integrate/correct_field.h"
#include "./integrate/decide_time_step_size.h"
#include "./integrate/predict_field.h"
#include "./integrate/solve_poisson.h"

typedef struct {
  correct_field_data_t correct_field_data;
  decide_time_step_size_data_t decide_time_step_size_data;
  predict_field_data_t predict_field_data;
  solve_poisson_data_t solve_poisson_data;
} integrate_data_t;

extern int integrate_data_initialize(
    id<MTLDevice> device,
    const domain_t * const domain,
    id<MTLBuffer> physical_buffer,
    const size_t max_batch_size,
    integrate_data_t * const integrate_data
);

extern int integrate(
    const size_t batch_size,
    const domain_t * const domain,
    impose_boundary_condition_data_t * const impose_boundary_condition_data,
    integrate_data_t * const integrate_data,
    id<MTLCommandQueue> queue,
    const float diffusivity,
    flow_field_t * const flow_field,
    flow_solver_t * const flow_solver,
    size_t * const size,
    double * const time
);

#endif // INTEGRATE_H
