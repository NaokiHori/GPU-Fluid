#if !defined(SOLVE_POISSON_Y_H)
#define SOLVE_POISSON_Y_H

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "compute_pipeline.h"
#include "domain.h"

typedef struct {
  compute_pipeline_t compute_pipeline;
  id<MTLBuffer> lower_diagonals;
  id<MTLBuffer> upper_diagonals;
} solve_poisson_y_data_t;

extern int solve_poisson_y_data_initialize(
    id<MTLDevice> device,
    const domain_t * const domain,
    solve_poisson_y_data_t * const solve_poisson_y_data
);

extern int solve_poisson_y(
    const domain_t * const domain,
    solve_poisson_y_data_t * const solve_poisson_y_data,
    id<MTLComputeCommandEncoder> encoder,
    id<MTLBuffer> buf_eigenvalues,
    id<MTLBuffer> buf_rhs,
    id<MTLBuffer> buf_tmp
);

#endif // SOLVE_POISSON_Y_H
