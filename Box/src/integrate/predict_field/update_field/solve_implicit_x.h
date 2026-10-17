#if !defined(SOLVE_IMPLICIT_X_H)
#define SOLVE_IMPLICIT_X_H

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "compute_pipeline.h"
#include "domain.h"
#include "flow_solver.h"

typedef struct {
  compute_pipeline_t compute_pipeline;
  id<MTLBuffer> ux_lower_diagonals;
  id<MTLBuffer> ux_upper_diagonals;
  id<MTLBuffer> uy_lower_diagonals;
  id<MTLBuffer> uy_upper_diagonals;
  id<MTLBuffer> temperature_lower_diagonals;
  id<MTLBuffer> temperature_upper_diagonals;
} solve_implicit_x_data_t;

extern int solve_implicit_x_data_initialize(
    id<MTLDevice> device,
    const domain_t * const domain,
    solve_implicit_x_data_t * const solve_implicit_x_data
);

extern int solve_implicit_x(
    const domain_t * const domain,
    solve_implicit_x_data_t * const solve_implicit_x_data,
    id<MTLComputeCommandEncoder> encoder,
    id<MTLBuffer> mtl_buffer_dt,
    const float off_diagonal_prefactor,
    flow_solver_t * const flow_solver
);

#endif // SOLVE_IMPLICIT_X_H
