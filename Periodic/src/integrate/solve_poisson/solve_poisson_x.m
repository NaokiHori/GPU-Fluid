#include "memory.h"
#include "shared_buffer.h"
#include "./solve_poisson_x.h"

static int fft_plan_initialize(
    MPSGraph * const graph,
    const NSUInteger nx,
    const NSUInteger ny,
    id<MTLBuffer> mtl_buffer_physical,
    id<MTLBuffer> mtl_buffer_spectral,
    fft_plan_t * const plan
) {
  const NSUInteger nxh = nx / 2 + 1;
  // tensor data used by forward and backward
  MPSGraphTensorData * const tensor_data_physical = [
    [MPSGraphTensorData alloc]
      initWithMTLBuffer:mtl_buffer_physical
      shape:@[@(ny), @(nx)]
      dataType:MPSDataTypeFloat32
  ];
  MPSGraphTensorData * const tensor_data_spectral = [
    [MPSGraphTensorData alloc]
      initWithMTLBuffer:mtl_buffer_spectral
      shape:@[@(ny), @(nxh)]
      dataType:MPSDataTypeComplexFloat32
  ];
  // forward
  {
    MPSGraphTensor * const tensor_input = [
      graph
        placeholderWithShape:@[@(ny), @(nx)]
        dataType:MPSDataTypeFloat32
        name:@"tensor_input_forward"
    ];
    MPSGraphFFTDescriptor * const desc = [MPSGraphFFTDescriptor descriptor];
    desc.inverse = NO;
    desc.scalingMode = MPSGraphFFTScalingModeNone;
    MPSGraphTensor * const tensor_output = [
      graph
        realToHermiteanFFTWithTensor:tensor_input
        axes:@[@1]
        descriptor:desc
        name:@"tensor_output_forward"
    ];
    plan->forward_feeds = @{tensor_input: tensor_data_physical};
    plan->forward_results_dictionary = @{tensor_output: tensor_data_spectral};
  }
  // backward
  {
    MPSGraphTensor * const tensor_input = [
      graph
        placeholderWithShape:@[@(ny), @(nxh)]
        dataType:MPSDataTypeComplexFloat32
        name:@"tensor_input_backward"
    ];
    MPSGraphFFTDescriptor * const desc = [MPSGraphFFTDescriptor descriptor];
    desc.inverse = YES;
    desc.scalingMode = MPSGraphFFTScalingModeSize;
    desc.roundToOddHermitean = 1 == nx / 2;
    MPSGraphTensor *tensor_output = [
      graph
        HermiteanToRealFFTWithTensor:tensor_input
        axes:@[@1]
        descriptor:desc
        name:@"tensor_output_backward"
    ];
    plan->backward_feeds = @{tensor_input: tensor_data_spectral};
    plan->backward_results_dictionary = @{tensor_output: tensor_data_physical};
  }
  return 0;
}

int solve_poisson_x_data_initialize(
    id<MTLDevice> device,
    const domain_t * const domain,
    id<MTLBuffer> physical_buffer,
    id<MTLBuffer> spectral_buffer,
    solve_poisson_x_data_t * const solve_poisson_x_data
) {
  const double lx = domain->x_coordinate.length;
  const size_t nx = domain->x_coordinate.nitems;
  const size_t ny = domain->y_coordinate.nitems;
  id<MTLBuffer> * const eigenvalues = &solve_poisson_x_data->eigenvalues;
  *eigenvalues = create_shared_buffer(device, nx * sizeof(float));
  float * const ptr_eigenvalues = [*eigenvalues contents];
  for (size_t n = 0; n < nx; n++) {
    ptr_eigenvalues[n] = - pow(2. / lx * nx * sin(M_PI * n / nx), 2.);
  }
  MPSGraph ** const graph = &solve_poisson_x_data->graph;
  *graph = [MPSGraph new];
  if (0 != fft_plan_initialize(
      *graph,
      nx,
      ny,
      physical_buffer,
      spectral_buffer,
      &solve_poisson_x_data->fft_plan
  )) {
    return 1;
  }
  return 0;
}

int solve_poisson_x(
    solve_poisson_x_data_t * const solve_poisson_x_data,
    MPSCommandBuffer * const command_buffer,
    const bool is_forward
) {
  fft_plan_t * const fft_plan = &solve_poisson_x_data->fft_plan;
  if (is_forward) {
    [
      solve_poisson_x_data->graph
        encodeToCommandBuffer:command_buffer
        feeds:fft_plan->forward_feeds
        targetOperations:nil
        resultsDictionary:fft_plan->forward_results_dictionary
        executionDescriptor:nil
    ];
  } else {
    [
      solve_poisson_x_data->graph
        encodeToCommandBuffer:command_buffer
        feeds:fft_plan->backward_feeds
        targetOperations:nil
        resultsDictionary:fft_plan->backward_results_dictionary
        executionDescriptor:nil
    ];
  }
  return 0;
}

