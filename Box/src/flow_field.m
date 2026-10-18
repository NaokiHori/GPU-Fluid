#include <stddef.h>
#include "flow_field.h"
#include "shared_buffer.h"

static int init_ux(
    const domain_t * const domain,
    id<MTLBuffer> ux
) {
  const size_t nx = domain->x_coordinate.nitems;
  const size_t ny = domain->y_coordinate.nitems;
  float * const ptr_ux = [ux contents];
  for (size_t j = 1; j <= ny; j++) {
    for (size_t i = 2; i <= nx; i++) {
      ptr_ux[j * (nx + 2) + i] = 0.f;
    }
  }
  return 0;
}

static int init_uy(
    const domain_t * const domain,
    id<MTLBuffer> uy
) {
  const size_t nx = domain->x_coordinate.nitems;
  const size_t ny = domain->y_coordinate.nitems;
  float * const ptr_uy = (float *)[uy contents];
  for (size_t j = 2; j <= ny; j++) {
    for (size_t i = 1; i <= nx; i++) {
      ptr_uy[j * (nx + 2) + i] = 0.f;
    }
  }
  return 0;
}

static int init_p(
    const domain_t * const domain,
    id<MTLBuffer> p
) {
  const size_t nx = domain->x_coordinate.nitems;
  const size_t ny = domain->y_coordinate.nitems;
  float * const ptr_p = (float *)[p contents];
  for (size_t j = 1; j <= ny; j++) {
    for (size_t i = 1; i <= nx; i++) {
      ptr_p[j * (nx + 2) + i] = 0.f;
    }
  }
  return 0;
}

static int init_temperature(
    const domain_t * const domain,
    id<MTLBuffer> temperature
) {
  const size_t nx = domain->x_coordinate.nitems;
  const size_t ny = domain->y_coordinate.nitems;
  float * const ptr_temperature = (float *)[temperature contents];
  const float amplitude = 1.e-2f;
  for (size_t j = 1; j <= ny; j++) {
    for (size_t i = 1; i <= nx; i++) {
      ptr_temperature[j * (nx + 2) + i] = amplitude * (- 0.5f + 1.f * rand() / RAND_MAX);
    }
  }
  return 0;
}

int flow_field_initialize(
    id<MTLDevice> device,
    const domain_t * const domain,
    flow_field_t * const flow_field
) {
  const size_t nx = domain->x_coordinate.nitems;
  const size_t ny = domain->y_coordinate.nitems;
  flow_field->ux = create_shared_buffer(device, (nx + 2) * (ny + 2) * sizeof(float));
  flow_field->uy = create_shared_buffer(device, (nx + 2) * (ny + 2) * sizeof(float));
  flow_field->p = create_shared_buffer(device, (nx + 2) * (ny + 2) * sizeof(float));
  flow_field->temperature = create_shared_buffer(device, (nx + 2) * (ny + 2) * sizeof(float));
  if (0 != init_ux(domain, flow_field->ux)) {
    return 1;
  }
  if (0 != init_uy(domain, flow_field->uy)) {
    return 1;
  }
  if (0 != init_p(domain, flow_field->p)) {
    return 1;
  }
  if (0 != init_temperature(domain, flow_field->temperature)) {
    return 1;
  }
  return 0;
}

