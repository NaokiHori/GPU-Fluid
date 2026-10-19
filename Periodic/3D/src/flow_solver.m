#include <complex.h>
#include "flow_solver.h"
#include "shared_buffer.h"

int flow_solver_initialize(
    id<MTLDevice> device,
    const domain_t * const domain,
    flow_solver_t * const flow_solver
) {
  const size_t nx = domain->x_coordinate.nitems;
  const size_t ny = domain->y_coordinate.nitems;
  const size_t nz = domain->z_coordinate.nitems;
  // NOTE: allocate larger buffer to accommodate intermediate buffer of Poisson solver
  flow_solver->scalar_potential = create_shared_buffer(
      device,
      (nx / 2 + 1) * ny * nz * sizeof(float complex)
  );
  flow_solver->dux = create_shared_buffer(device, nx * ny * nz * sizeof(float));
  flow_solver->duy = create_shared_buffer(device, nx * ny * nz * sizeof(float));
  flow_solver->duz = create_shared_buffer(device, nx * ny * (nz - 1) * sizeof(float));
  flow_solver->dtemperature = create_shared_buffer(device, nx * ny * nz * sizeof(float));
  flow_solver->dux_explicit_previous = create_shared_buffer(device, nx * ny * nz * sizeof(float));
  flow_solver->duy_explicit_previous = create_shared_buffer(device, nx * ny * nz * sizeof(float));
  flow_solver->duz_explicit_previous = create_shared_buffer(device, nx * ny * (nz - 1) * sizeof(float));
  flow_solver->dtemperature_explicit_previous = create_shared_buffer(device, nx * ny * nz * sizeof(float));
  return 0;
}

