# Periodic

Wall-bounded in one direction, periodic in the other directions.

## Method

- Finite-difference method (second-order accurate central-difference scheme)
- Fractional-step method
- Three-stage Runge-Kutta integrator
- Fully-explicit treatment in the periodic directions, semi-implicit treatment (Crank-Nicolson scheme) in the wall-normal direction
- FFT-based eigendecomposition in the periodic directions, Thomas algorithm in the wall-normal direction to solve Poisson equations

