#if !defined(SOLVE_POISSON_X_H)
#define SOLVE_POISSON_X_H

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "compute_pipeline.h"
#include "domain.h"

typedef struct {
  compute_pipeline_t compute_pipeline;
  id<MTLBuffer> forward_operator;
  id<MTLBuffer> backward_operator;
  id<MTLBuffer> eigenvalues;
} solve_poisson_x_data_t;

extern int solve_poisson_x_data_initialize(
    id<MTLDevice> device,
    const domain_t * const domain,
    solve_poisson_x_data_t * const solve_poisson_x_data
);

extern int solve_poisson_x(
    const domain_t * const domain,
    solve_poisson_x_data_t * const solve_poisson_x_data,
    id<MTLComputeCommandEncoder> encoder,
    id<MTLBuffer> operator,
    id<MTLBuffer> in,
    id<MTLBuffer> out
);

#endif // SOLVE_POISSON_X_H
