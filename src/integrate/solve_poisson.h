#if !defined(SOLVE_POISSON_H)
#define SOLVE_POISSON_H

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "compute_pipeline.h"
#include "domain.h"
#include "flow_field.h"
#include "flow_solver.h"
#include "./runge_kutta.h"

typedef struct {
  // common
  id<MTLBuffer> eigenvalues;
  id<MTLBuffer> buf;
  // x: gemm
  compute_pipeline_t compute_pipeline_x;
  id<MTLBuffer> forward_operator;
  id<MTLBuffer> backward_operator;
  // y: tdma
  compute_pipeline_t compute_pipeline_y;
  id<MTLBuffer> lower_diagonals;
  id<MTLBuffer> upper_diagonals;
  // setup
  compute_pipeline_t compute_pipeline_rhs;
} solve_poisson_data_t;

extern int solve_poisson_data_initialize(
    id<MTLDevice> device,
    const domain_t * const domain,
    solve_poisson_data_t * const solve_poisson_data
);

extern int solve_poisson(
    const domain_t * const domain,
    solve_poisson_data_t * const solve_poisson_data,
    id<MTLComputeCommandEncoder> encoder,
    const runge_kutta_coefficient_t * const runge_kutta_coefficient,
    id<MTLBuffer> mtl_buffer_dt,
    flow_field_t * const flow_field,
    flow_solver_t * const flow_solver
);

#endif // SOLVE_POISSON_H
