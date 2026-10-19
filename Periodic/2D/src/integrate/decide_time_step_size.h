#if !defined(DECIDE_TIME_STEP_SIZE_H)
#define DECIDE_TIME_STEP_SIZE_H

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "compute_pipeline.h"
#include "domain.h"
#include "flow_field.h"
#include "./decide_time_step_size/compute_convective_rate.h"
#include "./decide_time_step_size/finalize_time_step_size.h"

typedef struct {
  compute_convective_rate_data_t compute_convective_rate_data;
  finalize_time_step_size_data_t finalize_time_step_size_data;
  id<MTLBuffer> mtl_buffer_convective_rate_maxs;
  id<MTLBuffer> mtl_buffer_dt;
  id<MTLBuffer> mtl_buffer_dt_history;
} decide_time_step_size_data_t;

extern int decide_time_step_size_data_initialize(
    id<MTLDevice> device,
    const domain_t * const domain,
    const size_t batch_size,
    decide_time_step_size_data_t * const decide_time_step_size_data
);

extern int decide_time_step_size(
    const domain_t * const domain,
    decide_time_step_size_data_t * const decide_time_step_size_data,
    id<MTLComputeCommandEncoder> encoder,
    flow_field_t * const flow_field,
    const uint32_t batch
);

extern int update_time(
    decide_time_step_size_data_t * const decide_time_step_size_data,
    const size_t batch_size,
    size_t * const step,
    double * const time
);

#endif // DECIDE_TIME_STEP_SIZE_H
