#if !defined(COMPUTE_PIPELINE_H)
#define COMPUTE_PIPELINE_H

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

typedef struct {
  id<MTLComputePipelineState> pipeline_state;
} compute_pipeline_t;

extern int compute_pipeline_initialize(
    id<MTLDevice> device,
    const char * const shader_path,
    const char * const function_name,
    const NSDictionary<NSString *, id> * const macros,
    compute_pipeline_t * const compute_pipeline
);

#endif // COMPUTE_PIPELINE_H
