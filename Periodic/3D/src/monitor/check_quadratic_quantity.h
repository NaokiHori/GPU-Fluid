#if !defined(CHECK_QUADRATIC_QUANTITY_H)
#define CHECK_QUADRATIC_QUANTITY_H

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "compute_pipeline.h"
#include "domain.h"
#include "flow_field.h"

#define NQUANTITIES 4

typedef struct {
  compute_pipeline_t compute_pipeline;
  NSUInteger max_num_threadgroups;
  NSUInteger num_groups_x;
  NSUInteger num_groups_y;
  NSUInteger num_groups_z;
  NSUInteger threadgroup_width;
  NSUInteger threadgroup_height;
  NSUInteger threadgroup_depth;
  //
  id<MTLBuffer> mtl_buffer_quadratic_quantities;
} check_quadratic_quantity_data_t;

extern int check_quadratic_quantity_data_initialize(
    id<MTLDevice> device,
    const domain_t * const domain,
    check_quadratic_quantity_data_t * const check_quadratic_quantity_data
);

extern int process_on_gpu_check_quadratic_quantity(
    id<MTLComputeCommandEncoder> encoder,
    check_quadratic_quantity_data_t * const check_quadratic_quantity_data,
    const domain_t * const domain,
    const flow_field_t * const flow_field
);

extern int process_on_cpu_check_quadratic_quantity(
    check_quadratic_quantity_data_t * const check_quadratic_quantity_data,
    double * const quadratic_quantities
);

#endif // CHECK_QUADRATIC_QUANTITY_H
