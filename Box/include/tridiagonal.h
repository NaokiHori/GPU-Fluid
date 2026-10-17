#if !defined(TRIDIAGONAL_H)
#define TRIDIAGONAL_H

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

typedef struct {
  double * lower_diagonals;
  double * main_diagonals;
  double * upper_diagonals;
  id<MTLBuffer> mtl_buffer_lower_diagonals;
  id<MTLBuffer> mtl_buffer_main_diagonals;
  id<MTLBuffer> mtl_buffer_upper_diagonals;
} tridiagonal_t;

#endif // TRIDIAGONAL_H
