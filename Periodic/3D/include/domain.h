#if !defined(DOMAIN_H)
#define DOMAIN_H

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stddef.h>

#include "tridiagonal.h"

#define NDIMS 3

// TODO: use shared definition between host and device
typedef struct {
  float scale_factor;
  float lower_diagonal;
  float main_diagonal;
  float upper_diagonal;
} metric_t;

typedef struct {
  double length;
  size_t nitems;
  double * centers;
  double * faces;
} x_coordinate_t;

typedef struct {
  double length;
  size_t nitems;
  double * centers;
  double * faces;
} y_coordinate_t;

typedef struct {
  double length;
  size_t nitems;
  // computed and used by CPU
  double * centers;
  double * faces;
  double * center_scale_factors;
  double * face_scale_factors;
  tridiagonal_t center_laplace_operators;
  tridiagonal_t face_laplace_operators;
  // used by GPU
  id<MTLBuffer> mtl_buffer_center_scale_factors;
  id<MTLBuffer> mtl_buffer_face_scale_factors;
  id<MTLBuffer> mtl_buffer_center_metrics;
  id<MTLBuffer> mtl_buffer_face_metrics;
} z_coordinate_t;

typedef struct {
  x_coordinate_t x_coordinate;
  y_coordinate_t y_coordinate;
  z_coordinate_t z_coordinate;
} domain_t;

extern int domain_initialize(
    id<MTLDevice> device,
    domain_t * const domain
);

extern int domain_finalize(
    domain_t * const domain
);

#endif // DOMAIN_H
