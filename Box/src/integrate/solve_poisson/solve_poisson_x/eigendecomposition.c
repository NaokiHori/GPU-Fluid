#include <float.h>
#include <math.h>
#include <stdbool.h>
#include "memory.h"
#include "./eigendecomposition.h"

static bool is_converged(
    const double sub_diagonal
) {
  return fabs(sub_diagonal) < DBL_EPSILON;
}

// compute Wilkinson's shift
static int compute_shift(
    const double * const main_diagonals,
    const double * const sub_diagonals,
    const size_t n_end,
    double * const shift
) {
  if (n_end < 2) {
    return 1;
  }
  const double diagonals[2] = {
    main_diagonals[n_end - 2],
    main_diagonals[n_end - 1],
  };
  const double sub_diagonal = sub_diagonals[n_end - 2];
  const double half_difference = 0.5 * (diagonals[0] - diagonals[1]);
  const double sign_half_difference = half_difference < 0. ? -1. : 1.;
  *shift =
    diagonals[1]
    -
    pow(sub_diagonal, 2.) / (
        half_difference
        +
        sign_half_difference * hypot(sub_diagonal, half_difference)
    );
  return 0;
}

static int chase_bulge(
    const size_t nitems,
    const size_t n_start,
    const size_t n_end,
    const double shift,
    double * const main_diagonals,
    double * const sub_diagonals,
    double * const eigenvectors_transposed
) {
  // rotation angle is determined by these quantities
  double values[2] = {
    main_diagonals[n_start] - shift,
    sub_diagonals[n_start],
  };
  for (size_t i = n_start; i < n_end - 1; i++) {
    // Givens rotation applied to i-th and (i+1)-th rows
    // BEFORE
    //   (i    )-th row: [ . . s m s . . . ]
    //   (i + 1)-th row: [ . . b s m s . . ]
    // AFTER
    //   (i    )-th row: [ . . s m s b . . ]
    //   (i + 1)-th row: [ . . . s m s . . ]
    // - m: main-diagonal
    // - s: sub-diagonal
    // - b: bulge
    const double r = hypot(values[0], values[1]);
    if (n_start < i) {
      // affected by eliminating the bulge
      sub_diagonals[i - 1] = r;
    }
    const double c = 0. == r ? 1. : + values[0] / r;
    const double s = 0. == r ? 0. : - values[1] / r;
    const double local_main_diagonals[2] = {
      main_diagonals[i    ],
      main_diagonals[i + 1],
    };
    const double local_sub_diagonal = sub_diagonals[i];
    // update the 2x2 diagonal block this rotation acts on
    // Q^T A Q = G A G^T =
    //   [  c -s ] [ d0  e ] [  c  s ]
    //   [  s  c ] [  e d1 ] [ -s  c ]
    const double cc = c * c;
    const double sc = s * c;
    const double ss = s * s;
    main_diagonals[i    ] = cc * local_main_diagonals[0] - 2. * sc * local_sub_diagonal + ss * local_main_diagonals[1];
    main_diagonals[i + 1] = ss * local_main_diagonals[0] + 2. * sc * local_sub_diagonal + cc * local_main_diagonals[1];
    sub_diagonals[i] = sc * local_main_diagonals[0] + (cc - ss) * local_sub_diagonal - sc * local_main_diagonals[1];
    // update (transposed) eigenvectors
    // V := V Q = V G^T
    // -> V^T := (V G^T)^T = G V^T
    for (size_t j = 0; j < nitems; j++) {
      // for each column (j), two rows (i and i + 1) are modulated
      double * const elements[2] = {
        eigenvectors_transposed + (i    ) * nitems + j,
        eigenvectors_transposed + (i + 1) * nitems + j,
      };
      const double values[2] = {
        *elements[0],
        *elements[1],
      };
      *elements[0] = + c * values[0] - s * values[1];
      *elements[1] = + s * values[0] + c * values[1];
    }
    if (n_end - 2 == i) {
      break;
    }
    // prepare for the next bulge chasing
    // [ b ] = [  c -s ] [ 0 ] = [ -s * e ]
    // [ s ] = [  s  c ] [ e ] = [  c * e ]
    values[0] = sub_diagonals[i];
    // new bulge
    values[1] = - s * sub_diagonals[i + 1];
    sub_diagonals[i + 1] *= c;
  }
  return 0;
}

typedef struct {
  size_t n_start;
  size_t n_end;
} range_t;

static bool is_trivial_range(
    const range_t * const range
) {
  return range->n_end - range->n_start < 2;
}

typedef struct {
  size_t count;
  range_t * ranges;
} range_stack_t;

static int initialize_stack(
    const size_t nitems,
    range_stack_t * const stack
) {
  stack->count = 0;
  stack->ranges = memory_alloc(nitems, sizeof(range_t));
  return 0;
}

static int push_stack(
    range_stack_t * const stack,
    const range_t range
) {
  size_t * const count = &stack->count;
  stack->ranges[*count] = range;
  *count += 1;
  return 0;
}

static int pop_stack(
    range_stack_t * const stack,
    range_t * const range
) {
  size_t * const count = &stack->count;
  if (*count < 1) {
    return 1;
  }
  *range = stack->ranges[*count - 1];
  *count -= 1;
  return 0;
}

static int finalize_stack(
    range_stack_t * const stack
) {
  stack->count = 0;
  memory_free(stack->ranges);
  return 0;
}

static int converge_range(
    const size_t iter_max,
    const size_t nitems,
    double * const main_diagonals,
    double * const sub_diagonals,
    double * const eigenvectors_transposed,
    range_stack_t * const stack,
    const range_t * const current_range
) {
  const size_t n_start = current_range->n_start;
  const size_t n_end = current_range->n_end;
  // the maximum number of iterations is defined for safety: should not reach
  for (size_t iter = 0; iter < iter_max * (n_end - n_start); iter++) {
    // process block
    double shift = 0.;
    compute_shift(main_diagonals, sub_diagonals, n_end, &shift);
    chase_bulge(
        nitems,
        n_start,
        n_end,
        shift,
        main_diagonals,
        sub_diagonals,
        eigenvectors_transposed
    );
    // check deflation point
    for (size_t n = n_start; n < n_end - 1; n++) {
      if (!is_converged(sub_diagonals[n])) {
        continue;
      }
      // split the block into two sub blocks at the deflation point,
      //   which are registered to the todo list (stack)
      push_stack(stack, (range_t){.n_start = n_start, .n_end = n + 1});
      push_stack(stack, (range_t){.n_start = n + 1, .n_end = n_end});
      return 0;
    }
  }
  return 1;
}

static int build_identity_matrix(
    const size_t nitems,
    double * const matrix
) {
  for (size_t i = 0; i < nitems; i++) {
    for (size_t j = 0; j < nitems; j++) {
      matrix[i * nitems + j] = i == j ? 1. : 0.;
    }
  }
  return 0;
}

int compute_eigendecomposition(
    const size_t iter_max,
    const size_t nitems,
    double * const main_diagonals,
    double * const sub_diagonals,
    double * const eigenvalues,
    double * const eigenvectors_transposed
) {
  build_identity_matrix(nitems, eigenvectors_transposed);
  int result = 0;
  range_stack_t stack = {0};
  initialize_stack(nitems, &stack);
  // set initial range: the full matrix
  push_stack(
      &stack,
      (range_t){
        .n_start = 0,
        .n_end = nitems,
      }
  );
  // process all stacked ranges
  for (;;) {
    range_t current_range = {0};
    if (0 != pop_stack(&stack, &current_range)) {
      // cannot pull anything from stack: no more ranges to process
      result = 0;
      break;
    }
    if (is_trivial_range(&current_range)) {
      // one-row block: nothing to do
      continue;
    }
    // let the current range converged
    if (0 != converge_range(
        iter_max,
        nitems,
        main_diagonals,
        sub_diagonals,
        eigenvectors_transposed,
        &stack,
        &current_range
    )) {
      // failed to converge
      result = 1;
      break;
    }
  }
  finalize_stack(&stack);
  for (size_t i = 0; i < nitems; i++) {
    eigenvalues[i] = main_diagonals[i];
  }
  return result;
}

int sort_eigendecomposition(
    const size_t nitems,
    int (* const compare)(
      const void * const eigen_pair_a,
      const void * const eigen_pair_b
    ),
    double * const eigenvalues,
    double * const eigenvectors_transposed
) {
  eigen_pair_t * const pairs = memory_alloc(nitems, sizeof(eigen_pair_t));
  double * const sorted_eigenvectors_transposed = memory_alloc(nitems * nitems, sizeof(double));
  for (size_t i = 0; i < nitems; i++) {
    eigen_pair_t * const pair = pairs + i;
    pair->eigenvalue = eigenvalues[i];
    pair->eigenvector = eigenvectors_transposed + i * nitems;
  }
  qsort(pairs, nitems, sizeof(eigen_pair_t), compare);
  for (size_t i = 0; i < nitems; i++) {
    const eigen_pair_t * const pair = pairs + i;
    eigenvalues[i] = pair->eigenvalue;
    for (size_t j = 0; j < nitems; j++) {
      sorted_eigenvectors_transposed[i * nitems + j] = pair->eigenvector[j];
    }
  }
  for (size_t n = 0; n < nitems * nitems; n++) {
    eigenvectors_transposed[n] = sorted_eigenvectors_transposed[n];
  }
  memory_free(pairs);
  memory_free(sorted_eigenvectors_transposed);
  return 0;
}

