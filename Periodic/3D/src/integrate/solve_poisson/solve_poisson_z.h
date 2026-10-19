#if !defined(SOLVE_POISSON_Z_H)
#define SOLVE_POISSON_Z_H

#import <Foundation/Foundation.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <Metal/Metal.h>

#include "compute_pipeline.h"
#include "domain.h"

typedef struct {
  compute_pipeline_t compute_pipeline;
  id<MTLBuffer> lower_diagonals;
  id<MTLBuffer> upper_diagonals;
} solve_poisson_z_data_t;

extern int solve_poisson_z_data_initialize(
    id<MTLDevice> device,
    const domain_t * const domain,
    solve_poisson_z_data_t * const solve_poisson_z_data
);

extern int solve_poisson_z(
    const domain_t * const domain,
    solve_poisson_z_data_t * const solve_poisson_z_data,
    MPSCommandBuffer * const command_buffer,
    id<MTLBuffer> buf_x_eigenvalues,
    id<MTLBuffer> buf_y_eigenvalues,
    id<MTLBuffer> buf_rhs,
    id<MTLBuffer> buf_tmp
);

#endif // SOLVE_POISSON_Z_H
