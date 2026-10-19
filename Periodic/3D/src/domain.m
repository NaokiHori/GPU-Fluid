#include <assert.h>
#include <math.h>
#include <stddef.h>
#include "domain.h"
#include "memory.h"
#include "shared_buffer.h"
#include "tridiagonal.h"

static int allocate_tridiagonal(
    id<MTLDevice> device,
    const size_t nitems,
    tridiagonal_t * const tridiagonal
) {
  double ** const lower_diagonals = &tridiagonal->lower_diagonals;
  double ** const main_diagonals = &tridiagonal->main_diagonals;
  double ** const upper_diagonals = &tridiagonal->upper_diagonals;
  *lower_diagonals = memory_alloc(nitems, sizeof(double));
  *main_diagonals = memory_alloc(nitems, sizeof(double));
  *upper_diagonals = memory_alloc(nitems, sizeof(double));
  id<MTLBuffer> * const mtl_buffer_lower_diagonals = &tridiagonal->mtl_buffer_lower_diagonals;
  id<MTLBuffer> * const mtl_buffer_main_diagonals = &tridiagonal->mtl_buffer_main_diagonals;
  id<MTLBuffer> * const mtl_buffer_upper_diagonals = &tridiagonal->mtl_buffer_upper_diagonals;
  *mtl_buffer_lower_diagonals = create_shared_buffer(device, nitems * sizeof(float));
  *mtl_buffer_main_diagonals = create_shared_buffer(device, nitems * sizeof(float));
  *mtl_buffer_upper_diagonals = create_shared_buffer(device, nitems * sizeof(float));
  return 0;
}

static int free_tridiagonal(
    tridiagonal_t * const tridiagonal
) {
  memory_free(tridiagonal->lower_diagonals);
  memory_free(tridiagonal->main_diagonals);
  memory_free(tridiagonal->upper_diagonals);
  return 0;
}

static int build_coordinate_x(
    const double length,
    const size_t nitems,
    x_coordinate_t * const coordinate
) {
  coordinate->length = length;
  coordinate->nitems = nitems;
  // coordinates
  double ** const faces = &coordinate->faces;
  double ** const centers = &coordinate->centers;
  *faces = memory_alloc(nitems + 2, sizeof(double));
  *centers = memory_alloc(nitems + 2, sizeof(double));
  (*faces)[0] = nan("");
  (*faces)[1] = 0.;
  for (size_t i = 2; i < nitems + 1; i++) {
    double * const face = *faces + i;
    *face = length * (i - 1) / nitems;
  }
  (*faces)[nitems + 1] = length;
  (*centers)[0] = nan("");
  for (size_t i = 1; i < nitems + 1; i++) {
    (*centers)[i] = 0.5 * (*faces)[i] + 0.5 * (*faces)[i + 1];
  }
  (*centers)[nitems + 1] = nan("");
  return 0;
}

static int build_coordinate_y(
    const double length,
    const size_t nitems,
    y_coordinate_t * const coordinate
) {
  coordinate->length = length;
  coordinate->nitems = nitems;
  // coordinates
  double ** const faces = &coordinate->faces;
  double ** const centers = &coordinate->centers;
  *faces = memory_alloc(nitems + 2, sizeof(double));
  *centers = memory_alloc(nitems + 2, sizeof(double));
  (*faces)[0] = nan("");
  (*faces)[1] = 0.;
  for (size_t i = 2; i < nitems + 1; i++) {
    double * const face = *faces + i;
    *face = length * (i - 1) / nitems;
  }
  (*faces)[nitems + 1] = length;
  (*centers)[0] = nan("");
  for (size_t i = 1; i < nitems + 1; i++) {
    (*centers)[i] = 0.5 * (*faces)[i] + 0.5 * (*faces)[i + 1];
  }
  (*centers)[nitems + 1] = nan("");
  return 0;
}

// grid-streching indicator
// 0: extremely stretched
// 1: uniform
static int build_coordinate_z(
    id<MTLDevice> device,
    const double length,
    const size_t nitems,
    const double grad,
    z_coordinate_t * const coordinate
) {
  coordinate->length = length;
  coordinate->nitems = nitems;
  // coordinates
  double ** const faces = &coordinate->faces;
  double ** const centers = &coordinate->centers;
  *faces = memory_alloc(nitems + 2, sizeof(double));
  *centers = memory_alloc(nitems + 2, sizeof(double));
  (*faces)[0] = nan("");
  (*faces)[1] = 0.;
  for (size_t i = 2; i < nitems + 1; i++) {
    double * const face = *faces + i;
    const double x = 1. * (i - 1) / nitems;
    *face =
      + (2. * grad - 2.) * (x * x * x)
      + (- 3. * grad + 3.) * (x * x)
      + grad * x;
    *face *= length;
  }
  (*faces)[nitems + 1] = length;
  (*centers)[0] = (*faces)[1];
  for (size_t i = 1; i < nitems + 1; i++) {
    (*centers)[i] = 0.5 * (*faces)[i] + 0.5 * (*faces)[i + 1];
  }
  (*centers)[nitems + 1] = (*faces)[nitems + 1];
  // scale factors
  double ** const face_scale_factors = &coordinate->face_scale_factors;
  double ** const center_scale_factors = &coordinate->center_scale_factors;
  *face_scale_factors = memory_alloc(nitems + 2, sizeof(double));
  *center_scale_factors = memory_alloc(nitems + 2, sizeof(double));
  (*face_scale_factors)[0] = nan("");
  for (size_t i = 1; i < nitems + 2; i++) {
    (*face_scale_factors)[i] = (*centers)[i] - (*centers)[i - 1];
  }
  (*center_scale_factors)[0] = nan("");
  for (size_t i = 1; i < nitems + 1; i++) {
    (*center_scale_factors)[i] = (*faces)[i + 1] - (*faces)[i];
  }
  (*center_scale_factors)[nitems + 1] = nan("");
  id<MTLBuffer> * const mtl_buffer_center_scale_factors = &coordinate->mtl_buffer_center_scale_factors;
  id<MTLBuffer> * const mtl_buffer_face_scale_factors = &coordinate->mtl_buffer_face_scale_factors;
  *mtl_buffer_center_scale_factors = create_shared_buffer(device, (nitems + 2) * sizeof(float));
  *mtl_buffer_face_scale_factors = create_shared_buffer(device, (nitems + 2) * sizeof(float));
  float * const ptr_center_scale_factors = [*mtl_buffer_center_scale_factors contents];
  float * const ptr_face_scale_factors = [*mtl_buffer_face_scale_factors contents];
  for (size_t i = 0; i < nitems + 2; i++) {
    ptr_center_scale_factors[i] = (*center_scale_factors)[i];
    ptr_face_scale_factors[i] = (*face_scale_factors)[i];
  }
  // metrics at face for gpu
  {
    tridiagonal_t * const laplace_operators = &coordinate->face_laplace_operators;
    allocate_tridiagonal(device, nitems + 2, laplace_operators);
    double * const lower_diagonals = laplace_operators->lower_diagonals;
    double * const main_diagonals = laplace_operators->main_diagonals;
    double * const upper_diagonals = laplace_operators->upper_diagonals;
    lower_diagonals[0] = nan("");
    main_diagonals[0] = nan("");
    upper_diagonals[0] = nan("");
    lower_diagonals[1] = nan("");
    main_diagonals[1] = nan("");
    upper_diagonals[1] = nan("");
    for (size_t i = 2; i < nitems + 1; i++) {
      double * const lower_diagonal = lower_diagonals + i;
      double * const main_diagonal = main_diagonals + i;
      double * const upper_diagonal = upper_diagonals + i;
      *lower_diagonal = 1. / (*face_scale_factors)[i] / (*center_scale_factors)[i - 1];
      *upper_diagonal = 1. / (*face_scale_factors)[i] / (*center_scale_factors)[i    ];
      *main_diagonal = - *lower_diagonal - *upper_diagonal;
    }
    lower_diagonals[nitems + 1] = nan("");
    main_diagonals[nitems + 1] = nan("");
    upper_diagonals[nitems + 1] = nan("");
    float * const ptr_lower_diagonals = [laplace_operators->mtl_buffer_lower_diagonals contents];
    float * const ptr_main_diagonals = [laplace_operators->mtl_buffer_main_diagonals contents];
    float * const ptr_upper_diagonals = [laplace_operators->mtl_buffer_upper_diagonals contents];
    for (size_t i = 0; i < nitems + 2; i++) {
      ptr_lower_diagonals[i] = lower_diagonals[i];
      ptr_main_diagonals[i] = main_diagonals[i];
      ptr_upper_diagonals[i] = upper_diagonals[i];
    }
    id<MTLBuffer> * const mtl_buffer_metrics = &coordinate->mtl_buffer_face_metrics;
    *mtl_buffer_metrics = create_shared_buffer(device, (nitems + 2) * sizeof(metric_t));
    metric_t * const ptr_mtl_buffer_metrics = [*mtl_buffer_metrics contents];
    for (size_t i = 0; i < nitems + 2; i++) {
      metric_t * const metric = ptr_mtl_buffer_metrics + i;
      metric->scale_factor = (*face_scale_factors)[i];
      metric->lower_diagonal = lower_diagonals[i];
      metric->main_diagonal = main_diagonals[i];
      metric->upper_diagonal = upper_diagonals[i];
    }
  }
  // metrics at center for gpu
  {
    tridiagonal_t * const laplace_operators = &coordinate->center_laplace_operators;
    allocate_tridiagonal(device, nitems + 2, laplace_operators);
    double * const lower_diagonals = laplace_operators->lower_diagonals;
    double * const main_diagonals = laplace_operators->main_diagonals;
    double * const upper_diagonals = laplace_operators->upper_diagonals;
    lower_diagonals[0] = nan("");
    main_diagonals[0] = nan("");
    upper_diagonals[0] = nan("");
    for (size_t i = 1; i < nitems + 1; i++) {
      double * const lower_diagonal = lower_diagonals + i;
      double * const main_diagonal = main_diagonals + i;
      double * const upper_diagonal = upper_diagonals + i;
      *lower_diagonal = 1. / (*center_scale_factors)[i] / (*face_scale_factors)[i    ];
      *upper_diagonal = 1. / (*center_scale_factors)[i] / (*face_scale_factors)[i + 1];
      *main_diagonal = - *lower_diagonal - *upper_diagonal;
    }
    lower_diagonals[nitems + 1] = nan("");
    main_diagonals[nitems + 1] = nan("");
    upper_diagonals[nitems + 1] = nan("");
    float * const ptr_lower_diagonals = [laplace_operators->mtl_buffer_lower_diagonals contents];
    float * const ptr_main_diagonals = [laplace_operators->mtl_buffer_main_diagonals contents];
    float * const ptr_upper_diagonals = [laplace_operators->mtl_buffer_upper_diagonals contents];
    for (size_t i = 0; i < nitems + 2; i++) {
      ptr_lower_diagonals[i] = lower_diagonals[i];
      ptr_main_diagonals[i] = main_diagonals[i];
      ptr_upper_diagonals[i] = upper_diagonals[i];
    }
    id<MTLBuffer> * const mtl_buffer_metrics = &coordinate->mtl_buffer_center_metrics;
    *mtl_buffer_metrics = create_shared_buffer(device, (nitems + 2) * sizeof(metric_t));
    metric_t * const ptr_mtl_buffer_metrics = [*mtl_buffer_metrics contents];
    for (size_t i = 0; i < nitems + 2; i++) {
      metric_t * const metric = ptr_mtl_buffer_metrics + i;
      metric->scale_factor = (*center_scale_factors)[i];
      metric->lower_diagonal = lower_diagonals[i];
      metric->main_diagonal = main_diagonals[i];
      metric->upper_diagonal = upper_diagonals[i];
    }
  }
  return 0;
}

static int x_coordinate_finalize(
    x_coordinate_t * const coordinate
) {
  memory_free(coordinate->faces);
  memory_free(coordinate->centers);
  return 0;
}

static int y_coordinate_finalize(
    y_coordinate_t * const coordinate
) {
  memory_free(coordinate->faces);
  memory_free(coordinate->centers);
  return 0;
}

static int z_coordinate_finalize(
    z_coordinate_t * const coordinate
) {
  memory_free(coordinate->faces);
  memory_free(coordinate->centers);
  memory_free(coordinate->face_scale_factors);
  memory_free(coordinate->center_scale_factors);
  free_tridiagonal(&coordinate->face_laplace_operators);
  free_tridiagonal(&coordinate->center_laplace_operators);
  return 0;
}

int domain_initialize(
    id<MTLDevice> device,
    domain_t * const domain
) {
  // memory layout consistency check
  assert(sizeof(metric_t) == 16);
  assert(offsetof(metric_t, scale_factor) == 0);
  assert(offsetof(metric_t, lower_diagonal) == 4);
  assert(offsetof(metric_t, main_diagonal) == 8);
  assert(offsetof(metric_t, upper_diagonal) == 12);
  //
  const double lengths[NDIMS] = {2., 2., 1.};
  const size_t nitems[NDIMS] = {64, 64, 32};
  if (0 != build_coordinate_x(
      lengths[0],
      nitems[0],
      &domain->x_coordinate
  )) {
    return 1;
  }
  if (0 != build_coordinate_y(
      lengths[1],
      nitems[1],
      &domain->y_coordinate
  )) {
    return 1;
  }
  if (0 != build_coordinate_z(
      device,
      lengths[2],
      nitems[2],
      0.125,
      &domain->z_coordinate
  )) {
    return 1;
  }
  return 0;
}

int domain_finalize(
    domain_t * const domain
) {
  x_coordinate_finalize(&domain->x_coordinate);
  y_coordinate_finalize(&domain->y_coordinate);
  z_coordinate_finalize(&domain->z_coordinate);
  return 0;
}

