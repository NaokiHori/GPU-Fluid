#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <stdio.h>

#include "compute_pipeline.h"
#include "domain.h"
#include "flow_field.h"
#include "flow_solver.h"
#include "shared_buffer.h"
#include "./integrate.h"
#include "./monitor.h"
#include "./save.h"
#include "./visualize.h"

typedef struct {
  size_t monitor;
  size_t save;
  size_t visualize;
} schedule_t;

int main(
    void
) {
  @autoreleasepool {
    const size_t max_batch_size = 32;
    // setup gpu
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue = [device newCommandQueue];
    // construct field and solver
    domain_t domain = {0};
    if (0 != domain_initialize(device, &domain)) {
      return 1;
    }
    flow_field_t flow_field = {0};
    if (0 != flow_field_initialize(device, &domain, &flow_field)) {
      return 1;
    }
    flow_solver_t flow_solver = {0};
    if (0 != flow_solver_initialize(device, &domain, &flow_solver)) {
      return 1;
    }
    visualize_data_t visualize_data = {0};
    if (0 != visualize_data_initialize(
        device,
        &domain,
        2160, // image width
        1080, // image height
        &visualize_data
    )) {
      return 1;
    }
    // prepare shared buffer and peripheral stuffs
    integrate_data_t integrate_data = {0};
    if (0 != integrate_data_initialize(
        device,
        &domain,
        flow_solver.scalar_potential,
        max_batch_size,
        &integrate_data
    )) {
      return 1;
    }
    impose_boundary_condition_data_t impose_boundary_condition_data = {0};
    if (0 != impose_boundary_condition_data_initialize(
        device,
        &domain,
        &impose_boundary_condition_data
    )) {
      return 1;
    }
    monitor_data_t monitor_data = {0};
    if (0 != monitor_data_initialize(
        device,
        &domain,
        &monitor_data
    )) {
      return 1;
    }
    // impose boundary condition on the initial field
    {
      MPSCommandBuffer * const command_buffer = [queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
      impose_boundary_condition(
          &domain,
          &impose_boundary_condition_data,
          encoder,
          &flow_field
      );
      [encoder endEncoding];
      [command_buffer commit];
    }
    // integration
    const double time_max = 2.e+1;
    const float diffusivity = 5e-5f;
    const schedule_t rate = {
      .monitor = 100,
      .save = 1000,
      .visualize = 100,
    };
    schedule_t next = rate;
    for (double time = 0.; time < time_max; ) {
      static size_t step = 0;
      integrate(
          max_batch_size,
          &domain,
          &impose_boundary_condition_data,
          &integrate_data,
          queue,
          diffusivity,
          &flow_field,
          &flow_solver,
          &step,
          &time
      );
      // post process
      if (next.monitor < step) {
        monitor(queue, &monitor_data, step, time, &domain, &flow_field);
        next.monitor += rate.monitor;
      }
      if (next.save < step) {
        static size_t directory_index = 0;
        save(directory_index, step, time, &domain, &flow_field);
        directory_index += 1;
        next.save += rate.save;
      }
      if (next.visualize < step) {
        static size_t file_index = 0;
        visualize(queue, &visualize_data, file_index, &flow_field);
        file_index += 1;
      }
    }
  }
  return 0;
}

