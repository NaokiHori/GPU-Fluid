#include "shared_buffer.h"
#include "./solve_poisson_x.h"

static int x_fft_plan_initialize(
    MPSGraph * const graph,
    const NSUInteger nx,
    const NSUInteger ny,
    const NSUInteger nz,
    id<MTLBuffer> mtl_buffer_physical,
    id<MTLBuffer> mtl_buffer_intermediate,
    x_fft_plan_t * const plan
) {
  const NSUInteger nxh = nx / 2 + 1;
  // tensor data used by forward and backward
  MPSGraphTensorData * const tensor_data_physical = [
    [MPSGraphTensorData alloc]
      initWithMTLBuffer:mtl_buffer_physical
      shape:@[@(nz), @(ny), @(nx)]
      dataType:MPSDataTypeFloat32
  ];
  MPSGraphTensorData * const tensor_data_intermediate = [
    [MPSGraphTensorData alloc]
      initWithMTLBuffer:mtl_buffer_intermediate
      shape:@[@(nz), @(ny), @(nxh)]
      dataType:MPSDataTypeComplexFloat32
  ];
  // forward
  {
    MPSGraphTensor * const tensor_input = [
      graph
        placeholderWithShape:@[@(nz), @(ny), @(nx)]
        dataType:MPSDataTypeFloat32
        name:@"tensor_input_forward_x"
    ];
    MPSGraphFFTDescriptor * const desc = [MPSGraphFFTDescriptor descriptor];
    desc.inverse = NO;
    desc.scalingMode = MPSGraphFFTScalingModeNone;
    MPSGraphTensor * const tensor_output = [
      graph
        realToHermiteanFFTWithTensor:tensor_input
        axes:@[@2]
        descriptor:desc
        name:@"tensor_output_forward_x"
    ];
    plan->forward_feeds = @{tensor_input: tensor_data_physical};
    plan->forward_results_dictionary = @{tensor_output: tensor_data_intermediate};
  }
  // backward
  {
    MPSGraphTensor * const tensor_input = [
      graph
        placeholderWithShape:@[@(nz), @(ny), @(nxh)]
        dataType:MPSDataTypeComplexFloat32
        name:@"tensor_input_backward_x"
    ];
    MPSGraphFFTDescriptor * const desc = [MPSGraphFFTDescriptor descriptor];
    desc.inverse = YES;
    desc.scalingMode = MPSGraphFFTScalingModeSize;
    desc.roundToOddHermitean = 1 == nx % 2;
    MPSGraphTensor * const tensor_output = [
      graph
        HermiteanToRealFFTWithTensor:tensor_input
        axes:@[@2]
        descriptor:desc
        name:@"tensor_output_backward_x"
    ];
    plan->backward_feeds = @{tensor_input: tensor_data_intermediate};
    plan->backward_results_dictionary = @{tensor_output: tensor_data_physical};
  }
  return 0;
}

int solve_poisson_x_data_initialize(
    id<MTLDevice> device,
    MPSGraph * const graph,
    const domain_t * const domain,
    id<MTLBuffer> buffer_physical,
    id<MTLBuffer> buffer_intermediate,
    solve_poisson_x_data_t * const solve_poisson_x_data
) {
  const double lx = domain->x_coordinate.length;
  const size_t nx = domain->x_coordinate.nitems;
  const size_t ny = domain->y_coordinate.nitems;
  const size_t nz = domain->z_coordinate.nitems;
  const size_t nxh = nx / 2 + 1;
  id<MTLBuffer> * const eigenvalues = &solve_poisson_x_data->eigenvalues;
  *eigenvalues = create_shared_buffer(device, nxh * sizeof(float));
  float * const ptr_eigenvalues = [*eigenvalues contents];
  for (size_t n = 0; n < nxh; n++) {
    ptr_eigenvalues[n] = - pow(2. / lx * nx * sin(M_PI * n / nx), 2.);
  }
  if (0 != x_fft_plan_initialize(
      graph,
      nx,
      ny,
      nz,
      buffer_physical,
      buffer_intermediate,
      &solve_poisson_x_data->fft_plan
  )) {
    return 1;
  }
  return 0;
}

int solve_poisson_x(
    MPSGraph * const graph,
    solve_poisson_x_data_t * const solve_poisson_x_data,
    MPSCommandBuffer * const command_buffer,
    const bool is_forward
) {
  x_fft_plan_t * const fft_plan = &solve_poisson_x_data->fft_plan;
  if (is_forward) {
    [
      graph
        encodeToCommandBuffer:command_buffer
        feeds:fft_plan->forward_feeds
        targetOperations:nil
        resultsDictionary:fft_plan->forward_results_dictionary
        executionDescriptor:nil
    ];
  } else {
    [
      graph
        encodeToCommandBuffer:command_buffer
        feeds:fft_plan->backward_feeds
        targetOperations:nil
        resultsDictionary:fft_plan->backward_results_dictionary
        executionDescriptor:nil
    ];
  }
  return 0;
}

