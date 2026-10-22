#if !defined(PREDICT_FIELD_H)
#define PREDICT_FIELD_H

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "compute_pipeline.h"
#include "domain.h"
#include "flow_field.h"
#include "flow_solver.h"
#include "./runge_kutta.h"
#include "./predict_field/integrate_field.h"
#include "./predict_field/update_field.h"

typedef struct {
  integrate_field_data_t integrate_field_data;
  update_field_data_t update_field_data;
} predict_field_data_t;

extern int predict_field_data_initialize(
    id<MTLDevice> device,
    const domain_t * const domain,
    predict_field_data_t * const predict_field_data
);

extern int predict_field(
    const domain_t * const domain,
    predict_field_data_t * const predict_field_data,
    id<MTLComputeCommandEncoder> encoder,
    const runge_kutta_coefficient_t * const runge_kutta_coefficient,
    id<MTLBuffer> mtl_buffer_dt,
    const float diffusivity,
    flow_field_t * const flow_field,
    flow_solver_t * const flow_solver
);

#endif // PREDICT_FIELD_H
