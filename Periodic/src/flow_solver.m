#include "flow_solver.h"
#include "shared_buffer.h"

int flow_solver_initialize(
    id<MTLDevice> device,
    const domain_t * const domain,
    flow_solver_t * const flow_solver
) {
  const size_t nx = domain->x_coordinate.nitems;
  const size_t ny = domain->y_coordinate.nitems;
  flow_solver->scalar_potential = create_shared_buffer(device, nx * ny * sizeof(float));
  flow_solver->dux = create_shared_buffer(device, nx * ny * sizeof(float));
  flow_solver->duy = create_shared_buffer(device, nx * (ny - 1) * sizeof(float));
  flow_solver->dtemperature = create_shared_buffer(device, nx * ny * sizeof(float));
  flow_solver->dux_explicit_previous = create_shared_buffer(device, nx * ny * sizeof(float));
  flow_solver->duy_explicit_previous = create_shared_buffer(device, nx * (ny - 1) * sizeof(float));
  flow_solver->dtemperature_explicit_previous = create_shared_buffer(device, nx * ny * sizeof(float));
  return 0;
}

