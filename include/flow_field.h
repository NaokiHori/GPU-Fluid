#if !defined(FLOW_FIELD_H)
#define FLOW_FIELD_H

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "domain.h" // domain_t

typedef struct {
  id<MTLBuffer> ux;
  id<MTLBuffer> uy;
  id<MTLBuffer> p;
  id<MTLBuffer> temperature;
} flow_field_t;

extern int flow_field_initialize(
    id<MTLDevice> device,
    const domain_t * const domain,
    flow_field_t * const flow_field
);

#endif // FLOW_FIELD_H
