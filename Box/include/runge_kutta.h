#if !defined(RUNGE_KUTTA_H)
#define RUNGE_KUTTA_H

#define N_RUNGE_KUTTA_STEPS 3

typedef struct {
  float explicit_previous;
  float explicit_current;
  float implicit;
} runge_kutta_coefficient_t;

#endif // RUNGE_KUTTA_H
