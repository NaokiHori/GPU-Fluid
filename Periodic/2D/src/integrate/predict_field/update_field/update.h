#if !defined(UPDATE_H)
#define UPDATE_H

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "compute_pipeline.h"
#include "domain.h"
#include "flow_field.h"
#include "flow_solver.h"

typedef struct {
  compute_pipeline_t compute_pipeline;
} update_data_t;

extern int update_data_initialize(
    id<MTLDevice> device,
    const domain_t * const domain,
    update_data_t * const update_data
);

extern int update(
    const domain_t * const domain,
    update_data_t * const update_data,
    id<MTLComputeCommandEncoder> encoder,
    flow_field_t * const flow_field,
    flow_solver_t * const flow_solver
);

#endif // UPDATE_H
