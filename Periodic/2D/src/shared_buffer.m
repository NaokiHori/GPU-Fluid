#include "shared_buffer.h"

id<MTLBuffer> create_shared_buffer(
    id<MTLDevice> device,
    const size_t size
) {
  return [device newBufferWithLength:size options:MTLResourceStorageModeShared];
}

