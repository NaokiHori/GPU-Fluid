#if !defined(MONITOR_H)
#define MONITOR_H

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stddef.h>
#include "compute_pipeline.h"
#include "domain.h"
#include "flow_field.h"
#include "./monitor/check_divergence.h"

typedef struct {
  check_divergence_data_t check_divergence_data;
} monitor_data_t;

extern int monitor_data_initialize(
    id<MTLDevice> device,
    const domain_t * const domain,
    monitor_data_t * const monitor_data
);

extern int monitor(
    id<MTLCommandQueue> queue,
    monitor_data_t * const monitor_data,
    const size_t step,
    const double time,
    const domain_t * const domain,
    const flow_field_t * const flow_field
);

#endif // MONITOR_H
