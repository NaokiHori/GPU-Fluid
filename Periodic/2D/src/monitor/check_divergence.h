#if !defined(CHECK_DIVERGENCE_H)
#define CHECK_DIVERGENCE_H

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "compute_pipeline.h"
#include "domain.h"
#include "flow_field.h"

typedef struct {
  compute_pipeline_t compute_pipeline;
  NSUInteger max_num_threadgroups;
  NSUInteger num_groups_x;
  NSUInteger num_groups_y;
  NSUInteger threadgroup_width;
  NSUInteger threadgroup_height;
  //
  id<MTLBuffer> mtl_buffer_divergence_sums;
  id<MTLBuffer> mtl_buffer_divergence_maxs;
} check_divergence_data_t;

extern int check_divergence_data_initialize(
    id<MTLDevice> device,
    const domain_t * const domain,
    check_divergence_data_t * const check_divergence_data
);

extern int process_on_gpu_check_divergence(
    id<MTLComputeCommandEncoder> encoder,
    check_divergence_data_t * const check_divergence_data,
    const domain_t * const domain,
    const flow_field_t * const flow_field
);

extern int process_on_cpu_check_divergence(
    check_divergence_data_t * const check_divergence_data,
    double * const total_divergence,
    double * const max_divergence
);

#endif // CHECK_DIVERGENCE_H
