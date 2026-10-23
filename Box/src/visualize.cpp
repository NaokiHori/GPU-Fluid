#include <metal_stdlib>

using namespace metal;

#define TEMPERATURE(i, j) temperature[(j) * (NX + 2) + (i)]

inline int binary_search(
    const int nitems,
    device const float * const centers,
    const float position
) {
  int low = 0;
  int hi = nitems + 1;
  while (1 < hi - low) {
    const int mid = low + (hi - low) / 2;
    if (centers[mid] <= position) {
      low = mid;
    } else {
      hi = mid;
    }
  }
  return low;
}

inline float compute_weight(
    const int i,
    device const float * const centers,
    const float position
) {
  return clamp(
      (position - centers[i]) / (centers[i + 1] - centers[i]),
      0.f,
      1.f
  );
}

inline float sample_temperature(
    device const float * const x_centers,
    device const float * const y_centers,
    device const float * const temperature,
    const float2 position
) {
  const int i = binary_search(NX, x_centers, position.x);
  const int j = binary_search(NY, y_centers, position.y);
  const float wx = compute_weight(i, x_centers, position.x);
  const float wy = compute_weight(j, y_centers, position.y);
  // trilinear interpolation
  return mix(
      mix(TEMPERATURE(i, j    ), TEMPERATURE(i + 1, j    ), wx),
      mix(TEMPERATURE(i, j + 1), TEMPERATURE(i + 1, j + 1), wx),
      wy
  );
}

inline float3 colormap(
    const float value
) {
  const float clamped = clamp(value, -1.f, 1.f);
  return 0.f < clamped
    ? float3(1.f, 0.5f, 0.1f) * clamped
    : - float3(0.1f, 0.5f, 1.f) * clamped;
}

kernel void visualize(
    device const float * const x_centers [[buffer(0)]],
    device const float * const y_centers [[buffer(1)]],
    device const float * const temperature [[buffer(2)]],
    device packed_uchar3 * const image [[buffer(3)]],
    uint2 id [[thread_position_in_grid]]
) {
  const uint i = id.x;
  const uint j = id.y;
  if (WIDTH <= i || HEIGHT <= j) {
    return;
  }
  const float2 position = float2(
      LX * (1. * i + 0.5f) / WIDTH,
      LY * (1. * j + 0.5f) / HEIGHT
  );
  const float s = sample_temperature(x_centers, y_centers, temperature, position);
  const float3 color = colormap(s);
  image[j * WIDTH + i] = uchar3(255.f * saturate(color));
}

