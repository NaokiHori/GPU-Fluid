#if !defined(EIGENDECOMPOSITION_H)
#define EIGENDECOMPOSITION_H

#include <stddef.h> // size_t

// used by sort_eigendecomposition
typedef struct {
  double eigenvalue;
  const double * eigenvector;
} eigen_pair_t;

extern int compute_eigendecomposition(
    const size_t iter_max,
    const size_t nitems,
    double * const main_diagonals,
    double * const sub_diagonals,
    double * const eigenvalues,
    double * const eigenvectors_transposed
);

extern int sort_eigendecomposition(
    const size_t nitems,
    int (* const compare)(
      const void * const eigen_pair_a,
      const void * const eigen_pair_b
    ),
    double * const eigenvalues,
    double * const eigenvectors_transposed
);

#endif // EIGENDECOMPOSITION_H
