#if !defined(SOLVE_POISSON_X_H)
#define SOLVE_POISSON_X_H

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <Foundation/Foundation.h>

#include "compute_pipeline.h"
#include "domain.h"

typedef struct {
  MPSGraphTensorDataDictionary * forward_feeds;
  MPSGraphTensorDataDictionary * forward_results_dictionary;
  MPSGraphTensorDataDictionary * backward_feeds;
  MPSGraphTensorDataDictionary * backward_results_dictionary;
} x_fft_plan_t;

typedef struct {
  x_fft_plan_t fft_plan;
  id<MTLBuffer> eigenvalues;
} solve_poisson_x_data_t;

extern int solve_poisson_x_data_initialize(
    id<MTLDevice> device,
    MPSGraph * const graph,
    const domain_t * const domain,
    id<MTLBuffer> buffer_physical,
    id<MTLBuffer> buffer_intermediate,
    solve_poisson_x_data_t * const solve_poisson_x_data
);

extern int solve_poisson_x(
    MPSGraph * const graph,
    solve_poisson_x_data_t * const solve_poisson_x_data,
    MPSCommandBuffer * const command_buffer,
    const bool is_forward
);

#endif // SOLVE_POISSON_X_H
