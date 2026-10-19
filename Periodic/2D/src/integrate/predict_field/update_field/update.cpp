#include <metal_stdlib>

using namespace metal;

inline void update_ux(
    device const float * const dux,
    device float * const ux,
    uint2 id
) {
  uint i = id.x;
  uint j = id.y;
  if (i < 1 || NX < i || j < 1 || NY < j) {
    return;
  }
  ux[j * (NX + 2) + i] += dux[(j - 1) * NX + (i - 1)];
}

inline void update_uy(
    device const float * const duy,
    device float * const uy,
    uint2 id
) {
  uint i = id.x;
  uint j = id.y;
  if (i < 1 || NX < i || j < 2 || NY < j) {
    return;
  }
  uy[j * (NX + 2) + i] += duy[(j - 2) * NX + (i - 1)];
}

inline void update_temperature(
    device const float * const dtemperature,
    device float * const temperature,
    uint2 id
) {
  uint i = id.x;
  uint j = id.y;
  if (i < 1 || NX < i || j < 1 || NY < j) {
    return;
  }
  temperature[j * (NX + 2) + i] += dtemperature[(j - 1) * NX + (i - 1)];
}

kernel void update(
    const device float * const dux [[buffer(0)]],
    const device float * const duy [[buffer(1)]],
    const device float * const dtemperature [[buffer(2)]],
    device float * const ux [[buffer(3)]],
    device float * const uy [[buffer(4)]],
    device float * const temperature [[buffer(5)]],
    uint2 id [[thread_position_in_grid]]
) {
  update_ux(dux, ux, id);
  update_uy(duy, uy, id);
  update_temperature(dtemperature, temperature, id);
}

