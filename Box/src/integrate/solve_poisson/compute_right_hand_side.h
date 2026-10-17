#if !defined(COMPUTE_RIGHT_HAND_SIDE_H)
#define COMPUTE_RIGHT_HAND_SIDE_H

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "compute_pipeline.h"
#include "domain.h"

typedef struct {
  compute_pipeline_t compute_pipeline;
} compute_right_hand_side_data_t;

extern int compute_right_hand_side_initialize(
    id<MTLDevice> device,
    const domain_t * const domain,
    compute_right_hand_side_data_t * const compute_right_hand_side_data
);

extern int compute_right_hand_side(
    const domain_t * const domain,
    compute_right_hand_side_data_t * const compute_right_hand_side_data,
    id<MTLComputeCommandEncoder> encoder,
    const float runge_kutta_coefficient_implicit,
    id<MTLBuffer> mtl_buffer_dt,
    id<MTLBuffer> ux,
    id<MTLBuffer> uy,
    id<MTLBuffer> out
);

#endif // COMPUTE_RIGHT_HAND_SIDE_H
