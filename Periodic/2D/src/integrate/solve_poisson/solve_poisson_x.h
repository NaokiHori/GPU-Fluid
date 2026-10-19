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
} fft_plan_t;

typedef struct {
  MPSGraph * graph;
  fft_plan_t fft_plan;
  id<MTLBuffer> eigenvalues;
} solve_poisson_x_data_t;

extern int solve_poisson_x_data_initialize(
    id<MTLDevice> device,
    const domain_t * const domain,
    id<MTLBuffer> physical_buffer,
    id<MTLBuffer> spectral_buffer,
    solve_poisson_x_data_t * const solve_poisson_x_data
);

extern int solve_poisson_x(
    solve_poisson_x_data_t * const solve_poisson_x_data,
    MPSCommandBuffer * const command_buffer,
    const bool is_forward
);

#endif // SOLVE_POISSON_X_H
