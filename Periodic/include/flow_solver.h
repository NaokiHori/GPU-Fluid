#if !defined(FLOW_SOLVER_H)
#define FLOW_SOLVER_H

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "domain.h" // domain_t

typedef struct {
  id<MTLBuffer> scalar_potential;
  // increment of each field
  id<MTLBuffer> dux;
  id<MTLBuffer> duy;
  id<MTLBuffer> dtemperature;
  // buffers to store previous information of runge-kutta integrator
  id<MTLBuffer> dux_explicit_previous;
  id<MTLBuffer> duy_explicit_previous;
  id<MTLBuffer> dtemperature_explicit_previous;
} flow_solver_t;

extern int flow_solver_initialize(
    id<MTLDevice> device,
    const domain_t * const domain,
    flow_solver_t * const flow_solver
);

#endif // FLOW_SOLVER_H
