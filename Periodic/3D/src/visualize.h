#if !defined(VISUALIZE_H)
#define VISUALIZE_H

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stddef.h>
#include "compute_pipeline.h"
#include "domain.h"
#include "flow_field.h"

typedef struct {
  compute_pipeline_t compute_pipeline;
  size_t width;
  size_t height;
  id<MTLBuffer> mtl_buffer_camera;
  id<MTLBuffer> mtl_buffer_z_centers;
  id<MTLBuffer> mtl_buffer_image;
} visualize_data_t;

extern int visualize_data_initialize(
    id<MTLDevice> device,
    const domain_t * const domain,
    const size_t width,
    const size_t height,
    visualize_data_t * const visualize_data
);

extern int visualize(
    id<MTLCommandQueue> queue,
    const domain_t * const domain,
    visualize_data_t * const visualize_data,
    const size_t index,
    const flow_field_t * const flow_field
);

#endif // VISUALIZE_H
