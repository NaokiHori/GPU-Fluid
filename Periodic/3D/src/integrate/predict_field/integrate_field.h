#if !defined(INTEGRATE_FIELD_H)
#define INTEGRATE_FIELD_H

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "compute_pipeline.h"
#include "domain.h"
#include "flow_field.h"
#include "flow_solver.h"
#include "runge_kutta.h"

typedef struct {
  compute_pipeline_t compute_pipeline;
} integrate_field_data_t;

extern int integrate_field_data_initialize(
    id<MTLDevice> device,
    const domain_t * const domain,
    integrate_field_data_t * const integrate_field_data
);

extern int integrate_field(
    const domain_t * const domain,
    integrate_field_data_t * const integrate_field_data,
    id<MTLComputeCommandEncoder> encoder,
    const runge_kutta_coefficient_t * const runge_kutta_coefficient,
    id<MTLBuffer> mtl_buffer_dt,
    const float diffusivity,
    flow_field_t * const flow_field,
    flow_solver_t * const flow_solver
);

#endif // INTEGRATE_FIELD_H
