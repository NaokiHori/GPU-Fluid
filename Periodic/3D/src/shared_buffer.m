#include <string.h>
#include "shared_buffer.h"

id<MTLBuffer> create_shared_buffer(
    id<MTLDevice> device,
    const size_t size
) {
  id<MTLBuffer> buffer = [device newBufferWithLength:size options:MTLResourceStorageModeShared];
  void * const ptr = [buffer contents];
  memset(ptr, 0, size);
  return buffer;
}

