# GPU Fluid

GPGPU Direct Numerical Simulator on Apple Silicon.

## Features

- Designed exclusively for Apple Silicon leveraging unified memory architecture.
- Zero external dependencies (requires only standard Clang and Metal toolchains available by default on macOS).

## Usage

Visit one of the directories and execute:

```bash
make output
make all
./a.out
```

The resulting flow fields are written to `output/save` with `NPY` format, which can be for example visualized as:

```python
import numpy as np
from matplotlib import pyplot

x_centers = np.load('output/save/x_centers.npy')
y_centers = np.load('output/save/y_centers.npy')

x_centers, y_centers = np.meshgrid(x_centers, y_centers)

temperature = np.load(f'output/save/0000000000/temperature.npy')

figure = pyplot.figure()
ax = figure.add_subplot()
ax.contourf(x_centers, y_centers, temperature)
ax.set_aspect('equal')
pyplot.show()
pyplot.close()
```

## Reference

- [Performing calculations on a GPU | Apple Developer Documentation](https://developer.apple.com/documentation/metal/performing-calculations-on-a-gpu)

