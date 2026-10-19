#include "shared_buffer.h"
#include "./solve_poisson_y.h"

static int y_fft_plan_initialize(
    MPSGraph * const graph,
    const NSUInteger nx,
    const NSUInteger ny,
    const NSUInteger nz,
    id<MTLBuffer> mtl_buffer_intermediate,
    id<MTLBuffer> mtl_buffer_spectral,
    y_fft_plan_t * const plan
) {
  const NSUInteger nxh = nx / 2 + 1;
  // tensor data used by forward and backward
  MPSGraphTensorData * const tensor_data_intermediate = [
    [MPSGraphTensorData alloc]
      initWithMTLBuffer:mtl_buffer_intermediate
      shape:@[@(nz), @(ny), @(nxh)]
      dataType:MPSDataTypeComplexFloat32
  ];
  MPSGraphTensorData * const tensor_data_spectral = [
    [MPSGraphTensorData alloc]
      initWithMTLBuffer:mtl_buffer_spectral
      shape:@[@(nz), @(ny), @(nxh)]
      dataType:MPSDataTypeComplexFloat32
  ];
  // forward
  {
    MPSGraphTensor * const tensor_input = [
      graph
        placeholderWithShape:@[@(nz), @(ny), @(nxh)]
        dataType:MPSDataTypeComplexFloat32
        name:@"tensor_input_forward_y"
    ];
    MPSGraphFFTDescriptor * const desc = [MPSGraphFFTDescriptor descriptor];
    desc.inverse = NO;
    desc.scalingMode = MPSGraphFFTScalingModeNone;
    MPSGraphTensor * const tensor_output = [
      graph
        fastFourierTransformWithTensor:tensor_input
        axes:@[@1]
        descriptor:desc
        name:@"tensor_output_forward_y"
    ];
    plan->forward_feeds = @{tensor_input: tensor_data_intermediate};
    plan->forward_results_dictionary = @{tensor_output: tensor_data_spectral};
  }
  // backward
  {
    MPSGraphTensor * const tensor_input = [
      graph
        placeholderWithShape:@[@(nz), @(ny), @(nxh)]
        dataType:MPSDataTypeComplexFloat32
        name:@"tensor_input_backward_y"
    ];
    MPSGraphFFTDescriptor * const desc = [MPSGraphFFTDescriptor descriptor];
    desc.inverse = YES;
    desc.scalingMode = MPSGraphFFTScalingModeSize;
    MPSGraphTensor * const tensor_output = [
      graph
        fastFourierTransformWithTensor:tensor_input
        axes:@[@1]
        descriptor:desc
        name:@"tensor_output_backward_y"
    ];
    plan->backward_feeds = @{tensor_input: tensor_data_spectral};
    plan->backward_results_dictionary = @{tensor_output: tensor_data_intermediate};
  }
  return 0;
}

int solve_poisson_y_data_initialize(
    id<MTLDevice> device,
    MPSGraph * const graph,
    const domain_t * const domain,
    id<MTLBuffer> buffer_intermediate,
    id<MTLBuffer> buffer_spectral,
    solve_poisson_y_data_t * const solve_poisson_y_data
) {
  const double ly = domain->y_coordinate.length;
  const size_t nx = domain->x_coordinate.nitems;
  const size_t ny = domain->y_coordinate.nitems;
  const size_t nz = domain->z_coordinate.nitems;
  id<MTLBuffer> * const eigenvalues = &solve_poisson_y_data->eigenvalues;
  *eigenvalues = create_shared_buffer(device, ny * sizeof(float));
  float * const ptr_eigenvalues = [*eigenvalues contents];
  for (size_t n = 0; n < ny; n++) {
    ptr_eigenvalues[n] = - pow(2. / ly * ny * sin(M_PI * n / ny), 2.);
  }
  if (0 != y_fft_plan_initialize(
      graph,
      nx,
      ny,
      nz,
      buffer_intermediate,
      buffer_spectral,
      &solve_poisson_y_data->fft_plan
  )) {
    return 1;
  }
  return 0;
}

int solve_poisson_y(
    MPSGraph * const graph,
    solve_poisson_y_data_t * const solve_poisson_y_data,
    MPSCommandBuffer * const command_buffer,
    const bool is_forward
) {
  y_fft_plan_t * const fft_plan = &solve_poisson_y_data->fft_plan;
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

