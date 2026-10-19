#if !defined(IMPOSE_BOUNDARY_CONDITION_H)
#define IMPOSE_BOUNDARY_CONDITION_H

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "compute_pipeline.h"
#include "domain.h"
#include "flow_field.h"

typedef struct {
  compute_pipeline_t compute_pipeline_x;
  compute_pipeline_t compute_pipeline_y;
} impose_boundary_condition_data_t;

extern int impose_boundary_condition_data_initialize(
    id<MTLDevice> device,
    const domain_t * const domain,
    impose_boundary_condition_data_t * const impose_boundary_condition_data
);

extern int impose_boundary_condition(
    const domain_t * const domain,
    impose_boundary_condition_data_t * const impose_boundary_condition_data,
    id<MTLComputeCommandEncoder> encoder,
    flow_field_t * const flow_field
);

#endif //IMPOSE_BOUNDARY_CONDITION_H
