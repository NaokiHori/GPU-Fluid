#if !defined(COMPUTE_CONVECTIVE_RATE_H)
#define COMPUTE_CONVECTIVE_RATE_H

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "compute_pipeline.h"
#include "domain.h"
#include "flow_field.h"

typedef struct {
  compute_pipeline_t compute_pipeline;
  NSUInteger threadgroup_width;
  NSUInteger threadgroup_height;
  NSUInteger num_groups_x;
  NSUInteger num_groups_y;
} compute_convective_rate_data_t;

extern int compute_convective_rate_data_initialize(
    id<MTLDevice> device,
    const domain_t * const domain,
    compute_convective_rate_data_t * const compute_convective_rate_data
);

extern int compute_convective_rate(
    const domain_t * const domain,
    compute_convective_rate_data_t * const compute_convective_rate_data,
    id<MTLBuffer> mtl_buffer_convective_rate_maxs,
    id<MTLComputeCommandEncoder> encoder,
    flow_field_t * const flow_field
);

#endif // COMPUTE_CONVECTIVE_RATE_H
