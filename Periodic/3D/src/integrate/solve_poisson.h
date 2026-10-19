#if !defined(SOLVE_POISSON_H)
#define SOLVE_POISSON_H

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include "compute_pipeline.h"
#include "domain.h"
#include "flow_field.h"
#include "flow_solver.h"
#include "./runge_kutta.h"
#include "./solve_poisson/compute_right_hand_side.h"
#include "./solve_poisson/solve_poisson_x.h"
#include "./solve_poisson/solve_poisson_y.h"
#include "./solve_poisson/solve_poisson_z.h"

typedef struct {
  // common
  id<MTLBuffer> buf;
  MPSGraph * graph;
  // x: fft
  solve_poisson_x_data_t solve_poisson_x_data;
  // y: fft
  solve_poisson_y_data_t solve_poisson_y_data;
  // z: tdma
  solve_poisson_z_data_t solve_poisson_z_data;
  // setup
  compute_right_hand_side_data_t compute_right_hand_side_data;
} solve_poisson_data_t;

extern int solve_poisson_data_initialize(
    id<MTLDevice> device,
    const domain_t * const domain,
    id<MTLBuffer> physical_buffer,
    solve_poisson_data_t * const solve_poisson_data
);

extern int solve_poisson(
    const domain_t * const domain,
    solve_poisson_data_t * const solve_poisson_data,
    MPSCommandBuffer * const command_buffer,
    const runge_kutta_coefficient_t * const runge_kutta_coefficient,
    id<MTLBuffer> mtl_buffer_dt,
    flow_field_t * const flow_field,
    flow_solver_t * const flow_solver
);

#endif // SOLVE_POISSON_H
