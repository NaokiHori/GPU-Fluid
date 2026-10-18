#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "./monitor.h"
#include "./monitor/check_divergence.h"

#define ROOT_DIRECTORY "output/log/"

static int output(
    const size_t step,
    const double time,
    const char file_name[],
    const size_t nitems,
    const double * quantities
) {
  errno = 0;
  FILE * const fp = fopen(file_name, "a");
  if (NULL == fp) {
    perror(file_name);
    return 1;
  }
  fprintf(fp, "%10zu % .15e ", step, time);
  for (size_t n = 0; n < nitems; n++) {
    fprintf(fp, "% .15e%c", quantities[n], nitems - 1 == n ? '\n' : ' ');
  }
  fclose(fp);
  return 0;
}

static int process_on_gpu(
    id<MTLCommandQueue> queue,
    monitor_data_t * const monitor_data,
    const domain_t * const domain,
    const flow_field_t * const flow_field
) {
  MPSCommandBuffer * const command_buffer = [queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
  process_on_gpu_check_divergence(encoder, &monitor_data->check_divergence_data, domain, flow_field);
  [encoder endEncoding];
  [command_buffer commit];
  [command_buffer waitUntilCompleted];
  return 0;
}

static int process_on_cpu(
    monitor_data_t * const monitor_data,
    const size_t step,
    const double time
) {
  printf("step %zu time % .15e\n", step, time);
  const char file_name[] = ROOT_DIRECTORY "divergence.dat";
  double total_divergence = 0.;
  double max_divergence = 0.;
  process_on_cpu_check_divergence(
      &monitor_data->check_divergence_data,
      &total_divergence,
      &max_divergence
  );
  output(step, time, file_name, 2, (double []){total_divergence, max_divergence});
  return 0;
}

int monitor_data_initialize(
    id<MTLDevice> device,
    const domain_t * const domain,
    monitor_data_t * const monitor_data
) {
  check_divergence_data_initialize(device, domain, &monitor_data->check_divergence_data);
  return 0;
}

int monitor(
    id<MTLCommandQueue> queue,
    monitor_data_t * const monitor_data,
    const size_t step,
    const double time,
    const domain_t * const domain,
    const flow_field_t * const flow_field
) {
  process_on_gpu(queue, monitor_data, domain, flow_field);
  process_on_cpu(monitor_data, step, time);
  return 0;
}

