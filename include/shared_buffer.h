#if !defined(SHARED_BUFFER_H)
#define SHARED_BUFFER_H

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

extern id<MTLBuffer> create_shared_buffer(
    id<MTLDevice> device,
    const size_t size
);

#endif // SHARED_BUFFER_H
