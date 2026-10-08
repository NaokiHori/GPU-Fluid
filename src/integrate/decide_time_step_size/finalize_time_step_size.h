#if !defined(FINALIZE_TIME_STEP_SIZE_H)
#define FINALIZE_TIME_STEP_SIZE_H

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "compute_pipeline.h"
#include "domain.h"
#include "flow_field.h"

typedef struct {
  compute_pipeline_t compute_pipeline;
} finalize_time_step_size_data_t;

extern int finalize_time_step_size_data_initialize(
    id<MTLDevice> device,
    const domain_t * const domain,
    const NSUInteger num_groups_x,
    const NSUInteger num_groups_y,
    id<MTLBuffer> * const mtl_buffer_convective_rate_maxs,
    finalize_time_step_size_data_t * const finalize_time_step_size_data
);

extern int finalize_time_step_size(
    finalize_time_step_size_data_t * const finalize_time_step_size_data,
    id<MTLComputeCommandEncoder> encoder,
    id<MTLBuffer> mtl_buffer_convective_rate_maxs,
    id<MTLBuffer> mtl_buffer_dt,
    id<MTLBuffer> mtl_buffer_dt_history,
    const uint32_t batch
);

#endif // FINALIZE_TIME_STEP_SIZE_H
