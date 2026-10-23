#include <errno.h>
#include <math.h>
#include <stdio.h>
#include "shared_buffer.h"
#include "./visualize.h"

#define ROOT_DIRECTORY "output/image/"

static const size_t BYTES_PIXEL = 3;

int visualize_data_initialize(
    id<MTLDevice> device,
    const domain_t * const domain,
    const size_t width,
    const size_t height,
    visualize_data_t * const visualize_data
) {
  const size_t nx = domain->x_coordinate.nitems;
  const size_t ny = domain->y_coordinate.nitems;
  const double lx = domain->x_coordinate.length;
  const double ly = domain->y_coordinate.length;
  visualize_data->width = width;
  visualize_data->height = height;
  if (0 != compute_pipeline_initialize(
      device,
      "src/visualize.cpp",
      "visualize",
      @{
        @"NX": @(nx),
        @"NY": @(ny),
        @"HX": @(lx / nx),
        @"HY": @(ly / ny),
        @"LX": @(lx),
        @"LY": @(ly),
        @"WIDTH": @(width),
        @"HEIGHT": @(height),
      },
      &visualize_data->compute_pipeline
  )) {
    return 1;
  }
  visualize_data->mtl_buffer_x_centers = create_shared_buffer(device, (nx + 2) * sizeof(float));
  visualize_data->mtl_buffer_y_centers = create_shared_buffer(device, (ny + 2) * sizeof(float));
  float * const x_centers = [visualize_data->mtl_buffer_x_centers contents];
  float * const y_centers = [visualize_data->mtl_buffer_y_centers contents];
  for (size_t i = 0; i < nx + 2; i++) {
    x_centers[i] = domain->x_coordinate.centers[i];
  }
  for (size_t j = 0; j < ny + 2; j++) {
    y_centers[j] = domain->y_coordinate.centers[j];
  }
  // ppm buffer
  visualize_data->mtl_buffer_image = create_shared_buffer(
      device,
      width * height * BYTES_PIXEL * sizeof(uint8_t)
  );
  return 0;
}

static int process_on_gpu_visualize(
    id<MTLComputeCommandEncoder> encoder,
    visualize_data_t * const visualize_data,
    const flow_field_t * const flow_field
) {
  [encoder setComputePipelineState:visualize_data->compute_pipeline.pipeline_state];
  [encoder setBuffer:visualize_data->mtl_buffer_x_centers offset:0 atIndex:0];
  [encoder setBuffer:visualize_data->mtl_buffer_y_centers offset:0 atIndex:1];
  [encoder setBuffer:flow_field->temperature offset:0 atIndex:2];
  [encoder setBuffer:visualize_data->mtl_buffer_image offset:0 atIndex:3];
  MTLSize grid_size = MTLSizeMake(visualize_data->width, visualize_data->height, 1);
  MTLSize thread_group_size = MTLSizeMake(16, 16, 1);
  [encoder dispatchThreads:grid_size threadsPerThreadgroup:thread_group_size];
  return 0;
}

static int process_on_cpu_visualize(
    visualize_data_t * const visualize_data,
    const size_t index
) {
  const size_t width = visualize_data->width;
  const size_t height = visualize_data->height;
  char file_name[128] = {'\0'};
  snprintf(file_name, sizeof(file_name), ROOT_DIRECTORY "%08zu.ppm", index);
  errno = 0;
  FILE * const fp = fopen(file_name, "w");
  if (NULL == fp) {
    perror(file_name);
    return 1;
  }
  fprintf(fp, "P6\n%zu %zu\n255\n", width, height);
  const uint8_t * const image = [visualize_data->mtl_buffer_image contents];
  fwrite(image, sizeof(uint8_t), width * height * BYTES_PIXEL, fp);
  fclose(fp);
  return 0;
}

int visualize(
    id<MTLCommandQueue> queue,
    visualize_data_t * const visualize_data,
    const size_t index,
    const flow_field_t * const flow_field
) {
  id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
  process_on_gpu_visualize(encoder, visualize_data, flow_field);
  [encoder endEncoding];
  [command_buffer commit];
  [command_buffer waitUntilCompleted];
  process_on_cpu_visualize(visualize_data, index);
  return 0;
}

