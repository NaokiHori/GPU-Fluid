# Box

Two-dimensional wall-bounded domains.

<p align="center">
<image width="50%" src="./cover.jpg" />
</p>

## Usage

```bash
make output
make all
./a.out
```

The resulting flow fields are written to `output/save` with `NPY` format.
Contour images are dumped to `output/image` with `PPM` format.

## Method

- Finite-difference method (second-order accurate central-difference scheme)
- Fractional-step method
- Three-stage Runge-Kutta integrator
- Semi-implicit treatment (Crank-Nicolson scheme) in the two directions with approximate factorization to get independent tridiagonal linear systems
- GEMM-based eigendecomposition in the periodic directions, Thomas algorithm in the wall-normal direction to solve Poisson equations

