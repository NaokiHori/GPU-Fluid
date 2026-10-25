#if !defined(SOLVE_POISSON_Y_H)
#define SOLVE_POISSON_Y_H

#import <Foundation/Foundation.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <Metal/Metal.h>

#include "compute_pipeline.h"
#include "domain.h"

typedef struct {
  compute_pipeline_t compute_pipeline;
  id<MTLBuffer> mtl_buffer_lower_diagonals;
  id<MTLBuffer> mtl_buffer_upper_diagonals;
  double * lower_diagonals;
  double * upper_diagonals;
} solve_poisson_y_data_t;

extern int solve_poisson_y_data_initialize(
    id<MTLDevice> device,
    const domain_t * const domain,
    solve_poisson_y_data_t * const solve_poisson_y_data
);

extern int solve_poisson_y(
    const domain_t * const domain,
    solve_poisson_y_data_t * const solve_poisson_y_data,
    MPSCommandBuffer * const command_buffer,
    id<MTLBuffer> mtl_buffer_inverse_values,
    id<MTLBuffer> mtl_buffer_values
);

#endif // SOLVE_POISSON_Y_H
