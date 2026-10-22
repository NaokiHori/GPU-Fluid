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
  const size_t nz = domain->z_coordinate.nitems;
  const double lx = domain->x_coordinate.length;
  const double ly = domain->y_coordinate.length;
  const double lz = domain->z_coordinate.length;
  visualize_data->width = width;
  visualize_data->height = height;
  const double fov = 35. * M_PI / 180.;
  if (0 != compute_pipeline_initialize(
      device,
      "src/visualize.cpp",
      "visualize",
      @{
        @"NX": @(nx),
        @"NY": @(ny),
        @"NZ": @(nz),
        @"HX": @(lx / nx),
        @"HY": @(ly / ny),
        @"LX": @(lx),
        @"LY": @(ly),
        @"LZ": @(lz),
        @"WIDTH": @(width),
        @"HEIGHT": @(height),
        @"TAN_HALF_FOV": @(tan(0.5 * fov)),
        @"STEP": @(0.5 * fmin(lx / nx, ly / ny)),
        @"TEMPERATURE_SCALE": @(0.3), // |T| at which the color saturates
        @"OPACITY": @(25.),
      },
      &visualize_data->compute_pipeline
  )) {
    return 1;
  }
  // four xyz components computed and stored afterwards
  // - eye
  // - right
  // - forward
  // - up
  visualize_data->mtl_buffer_camera = create_shared_buffer(device, 12 * sizeof(float));
  // cell-center coordinates in z, including the wall nodes
  visualize_data->mtl_buffer_z_centers = create_shared_buffer(device, (nz + 2) * sizeof(float));
  float * const z_centers = [visualize_data->mtl_buffer_z_centers contents];
  for (size_t k = 0; k < nz + 2; k++) {
    z_centers[k] = domain->z_coordinate.centers[k];
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
    const domain_t * const domain,
    visualize_data_t * const visualize_data,
    const double angle,
    const flow_field_t * const flow_field
) {
  // camera: look at the box center from an oblique angle, z is up
  // NOTE: trial-and-errors and needed to get a "nice" scene
  {
    const double lx = domain->x_coordinate.length;
    const double ly = domain->y_coordinate.length;
    const double lz = domain->z_coordinate.length;
    const double center[3] = {0.5 * lx, 0.5 * ly, 0.4 * lz};
    const double scale = fmax(lx, fmax(ly, lz));
    double eye[3] = {
      center[0] + 1.8 * scale * cos(angle),
      center[1] + 1.8 * scale * sin(angle),
      center[2] + 0.2 * scale,
    };
    double forward[3] = {center[0] - eye[0], center[1] - eye[1], center[2] - eye[2]};
    double norm = sqrt(forward[0] * forward[0] + forward[1] * forward[1] + forward[2] * forward[2]);
    // move the eye farther away so that the whole box fits
    for (size_t n = 0; n < 3; n++) {
      eye[n] = center[n] - 1. * forward[n] / norm * norm / 1.;
      forward[n] /= norm;
    }
    // right = forward x z, up = right x forward
    double right[3] = {forward[1], - forward[0], 0.};
    norm = sqrt(right[0] * right[0] + right[1] * right[1]);
    for (size_t n = 0; n < 3; n++) {
      right[n] /= norm;
    }
    const double up[3] = {
      right[1] * forward[2] - right[2] * forward[1],
      right[2] * forward[0] - right[0] * forward[2],
      right[0] * forward[1] - right[1] * forward[0],
    };
    float * const camera = [visualize_data->mtl_buffer_camera contents];
    for (size_t n = 0; n < 3; n++) {
      camera[0 + n] = eye[n];
      camera[3 + n] = forward[n];
      camera[6 + n] = right[n];
      camera[9 + n] = up[n];
    }
  }
  [encoder setComputePipelineState:visualize_data->compute_pipeline.pipeline_state];
  [encoder setBuffer:visualize_data->mtl_buffer_camera offset:0 atIndex:0];
  [encoder setBuffer:visualize_data->mtl_buffer_z_centers offset:0 atIndex:1];
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
    const domain_t * const domain,
    visualize_data_t * const visualize_data,
    const size_t index,
    const flow_field_t * const flow_field
) {
  id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
  process_on_gpu_visualize(encoder, domain, visualize_data, 2. * M_PI * index / 600, flow_field);
  [encoder endEncoding];
  [command_buffer commit];
  [command_buffer waitUntilCompleted];
  return process_on_cpu_visualize(visualize_data, index);
}

