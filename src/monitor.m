#include "./monitor.h"
#include "./monitor/check_divergence.h"

static int process_on_gpu(
    id<MTLCommandQueue> queue,
    monitor_data_t * const monitor_data,
    const domain_t * const domain,
    const flow_field_t * const flow_field
) {
  id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
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
  process_on_cpu_check_divergence(&monitor_data->check_divergence_data, step, time);
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

