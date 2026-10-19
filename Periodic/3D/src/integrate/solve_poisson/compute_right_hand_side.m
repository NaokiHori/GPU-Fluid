#include "./compute_right_hand_side.h"

int compute_right_hand_side_initialize(
    id<MTLDevice> device,
    const domain_t * const domain,
    compute_right_hand_side_data_t * const compute_right_hand_side_data
) {
  const size_t nx = domain->x_coordinate.nitems;
  const size_t ny = domain->y_coordinate.nitems;
  const size_t nz = domain->z_coordinate.nitems;
  const double lx = domain->x_coordinate.length;
  const double ly = domain->y_coordinate.length;
  if (0 != compute_pipeline_initialize(
      device,
      "src/integrate/solve_poisson/compute_right_hand_side.cpp",
      "compute_right_hand_side",
      @{
        @"NX": @(nx),
        @"NY": @(ny),
        @"NZ": @(nz),
        @"HX": @(lx / nx),
        @"HY": @(ly / ny),
      },
      &compute_right_hand_side_data->compute_pipeline
  )) {
    return 1;
  }
  return 0;
}

int compute_right_hand_side(
    const domain_t * const domain,
    compute_right_hand_side_data_t * const compute_right_hand_side_data,
    MPSCommandBuffer * const command_buffer,
    const float runge_kutta_coefficient_implicit,
    id<MTLBuffer> mtl_buffer_dt,
    id<MTLBuffer> ux,
    id<MTLBuffer> uy,
    id<MTLBuffer> uz,
    id<MTLBuffer> out
) {
  const size_t nx = domain->x_coordinate.nitems;
  const size_t ny = domain->y_coordinate.nitems;
  const size_t nz = domain->z_coordinate.nitems;
  id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
  [encoder setComputePipelineState:compute_right_hand_side_data->compute_pipeline.pipeline_state];
  [encoder setBytes:&runge_kutta_coefficient_implicit length:sizeof(float) atIndex:0];
  [encoder setBuffer:mtl_buffer_dt offset:0 atIndex:1];
  [encoder setBuffer:domain->z_coordinate.mtl_buffer_center_scale_factors offset:0 atIndex:2];
  [encoder setBuffer:ux offset:0 atIndex:3];
  [encoder setBuffer:uy offset:0 atIndex:4];
  [encoder setBuffer:uz offset:0 atIndex:5];
  [encoder setBuffer:out offset:0 atIndex:6];
  //
  MTLSize grid_size = MTLSizeMake(nx, ny, nz);
  MTLSize thread_group_size = MTLSizeMake(8, 8, 4);
  [encoder dispatchThreads:grid_size threadsPerThreadgroup:thread_group_size];
  [encoder endEncoding];
  return 0;
}

