#if !defined(SOLVE_POISSON_H)
#define SOLVE_POISSON_H

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "compute_pipeline.h"
#include "domain.h"
#include "flow_field.h"
#include "flow_solver.h"
#include "./runge_kutta.h"
#include "./solve_poisson/compute_right_hand_side.h"
#include "./solve_poisson/solve_poisson_x.h"
#include "./solve_poisson/solve_poisson_y.h"

typedef struct {
  // common
  id<MTLBuffer> buf;
  // x: gemm
  solve_poisson_x_data_t solve_poisson_x_data;
  // y: tdma
  solve_poisson_y_data_t solve_poisson_y_data;
  // setup
  compute_right_hand_side_data_t compute_right_hand_side_data;
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
