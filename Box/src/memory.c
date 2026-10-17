#include <stdio.h> // fprintf, stderr
#include <stdlib.h> // calloc, free, exit, EXIT_FAILURE
#include "memory.h"

void * memory_alloc(
    const size_t nitems,
    const size_t size
) {
  void * const ptr = calloc(nitems, size);
  if (NULL == ptr) {
    goto abort;
  }
  return ptr;
abort:
  fprintf(stderr, "failed to allocate memory: %zu x %zu\n", nitems, size);
  exit(EXIT_FAILURE);
}

void memory_free(
    void * const ptr
) {
  free(ptr);
}

