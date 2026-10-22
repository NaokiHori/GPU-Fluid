#include <metal_stdlib>

using namespace metal;

#define TEMPERATURE(i, j, k) temperature[((k) * (NY + 2) + (j)) * (NX + 2) + (i)]

inline float sample_temperature(
    device const float * const z_centers,
    device const float * const temperature,
    const float3 position
) {
  // x, y: cell centers at (i - 0.5) * H, with periodic halos
  const float fx = position.x / HX + 0.5f;
  const float fy = position.y / HY + 0.5f;
  const int i0 = clamp((int)floor(fx), 0, NX);
  const int j0 = clamp((int)floor(fy), 0, NY);
  const float wx = clamp(fx - i0, 0.f, 1.f);
  const float wy = clamp(fy - j0, 0.f, 1.f);
  // z: non-uniform, z_centers[0] and z_centers[NZ + 1] are the wall nodes
  // use binary search
  int low = 0;
  int hi = NZ + 1;
  while (1 < hi - low) {
    const int mid = low + (hi - low) / 2;
    if (z_centers[mid] <= position.z) {
      low = mid;
    } else {
      hi = mid;
    }
  }
  const int k0 = low;
  const float wz = clamp(
      (position.z - z_centers[k0]) / (z_centers[k0 + 1] - z_centers[k0]),
      0.f,
      1.f
  );
  // trilinear interpolation
  const float t00 = mix(TEMPERATURE(i0, j0    , k0    ), TEMPERATURE(i0 + 1, j0    , k0    ), wx);
  const float t10 = mix(TEMPERATURE(i0, j0 + 1, k0    ), TEMPERATURE(i0 + 1, j0 + 1, k0    ), wx);
  const float t01 = mix(TEMPERATURE(i0, j0    , k0 + 1), TEMPERATURE(i0 + 1, j0    , k0 + 1), wx);
  const float t11 = mix(TEMPERATURE(i0, j0 + 1, k0 + 1), TEMPERATURE(i0 + 1, j0 + 1, k0 + 1), wx);
  return mix(mix(t00, t10, wy), mix(t01, t11, wy), wz);
}

inline float3 colormap(
    const float s
) {
  const float3 cold = float3(0.1f, 0.5f, 1.f);
  const float3 hot = float3(1.f, 0.5f, 0.1f);
  return 0.f < s ? hot * s : cold * (- s);
}

kernel void visualize(
    constant const float * const camera [[buffer(0)]], // eye, forward, right, up (3 floats each)
    device const float * const z_centers [[buffer(1)]],
    device const float * const temperature [[buffer(2)]],
    device packed_uchar3 * const image [[buffer(3)]],
    uint2 id [[thread_position_in_grid]]
) {
  const uint i = id.x;
  const uint j = id.y;
  if (WIDTH <= i || HEIGHT <= j) {
    return;
  }
  const float3 eye = float3(camera[0], camera[1], camera[2]);
  const float3 forward = float3(camera[3], camera[4], camera[5]);
  const float3 right = float3(camera[6], camera[7], camera[8]);
  const float3 up = float3(camera[9], camera[10], camera[11]);
  // ray
  const float aspect = (float)WIDTH / (float)HEIGHT;
  const float u = (2.f * (i + 0.5f) / WIDTH - 1.f) * TAN_HALF_FOV * aspect;
  const float v = (1.f - 2.f * (j + 0.5f) / HEIGHT) * TAN_HALF_FOV;
  const float3 direction = normalize(forward + u * right + v * up);
  // ray-box intersection (slab method)
  const float3 box = float3(LX, LY, LZ);
  const float3 inverse = 1.f / direction;
  const float3 t0 = (float3(0.f) - eye) * inverse;
  const float3 t1 = (box - eye) * inverse;
  const float t_near = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), max(min(t0.z, t1.z), 0.f));
  const float t_far = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));
  const float3 background = float3(0.02f, 0.02f, 0.05f);
  float3 color = float3(0.f);
  float alpha = 0.f;
  if (t_near < t_far) {
    // front-to-back compositing
    for (float t = t_near + 0.5f * STEP; t < t_far; t += STEP) {
      const float3 position = eye + t * direction;
      const float s = clamp(sample_temperature(z_centers, temperature, position) / TEMPERATURE_SCALE, -1.f, 1.f);
      const float a = 1.f - exp(- OPACITY * s * s * STEP);
      color += (1.f - alpha) * a * colormap(s);
      alpha += (1.f - alpha) * a;
      if (0.99f < alpha) {
        break;
      }
    }
  }
  color += (1.f - alpha) * background;
  image[j * WIDTH + i] = uchar3(255.f * saturate(color));
}

