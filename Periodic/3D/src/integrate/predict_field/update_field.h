#if !defined(UPDATE_FIELD_H)
#define UPDATE_FIELD_H

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "compute_pipeline.h"
#include "domain.h"
#include "flow_field.h"
#include "flow_solver.h"
#include "./update_field/solve_implicit_z.h"
#include "./update_field/update.h"

typedef struct {
  solve_implicit_z_data_t solve_implicit_z_data;
  update_data_t update_data;
} update_field_data_t;

extern int update_field_data_initialize(
    id<MTLDevice> device,
    const domain_t * const domain,
    update_field_data_t * const update_field_data
);

extern int update_field(
    const domain_t * const domain,
    update_field_data_t * const update_field_data,
    id<MTLComputeCommandEncoder> encoder,
    const float runge_kutta_coefficient_implicit,
    id<MTLBuffer> mtl_buffer_dt,
    const float diffusivity,
    flow_field_t * const flow_field,
    flow_solver_t * const flow_solver
);

#endif // UPDATE_FIELD_H
