#include <metal_stdlib>

using namespace metal;

#define UX(i, j, k) ux[((k) * (NY + 2) + (j)) * (NX + 2) + (i)]
#define UY(i, j, k) uy[((k) * (NY + 2) + (j)) * (NX + 2) + (i)]
#define UZ(i, j, k) uz[((k) * (NY + 2) + (j)) * (NX + 2) + (i)]
#define TEMPERATURE(i, j, k) temperature[((k) * (NY + 2) + (j)) * (NX + 2) + (i)]
// no halo cells
#define DUX(i, j, k) dux[((k - 1) * (NY) + (j - 1)) * (NX) + (i - 1)]
#define DUY(i, j, k) duy[((k - 1) * (NY) + (j - 1)) * (NX) + (i - 1)]
#define DUZ(i, j, k) duz[((k - 2) * (NY) + (j - 1)) * (NX) + (i - 1)]
#define DTEMPERATURE(i, j, k) dtemperature[((k - 1) * (NY) + (j - 1)) * (NX) + (i - 1)]

inline void update_ux(
    device const float * const dux,
    device float * const ux,
    uint3 id
) {
  const uint i = id.x + 1;
  const uint j = id.y + 1;
  const uint k = id.z + 1;
  if (NX < i || NY < j || NZ < k) {
    return;
  }
  UX(i, j, k) += DUX(i, j, k);
}

inline void update_uy(
    device const float * const duy,
    device float * const uy,
    uint3 id
) {
  const uint i = id.x + 1;
  const uint j = id.y + 1;
  const uint k = id.z + 1;
  if (NX < i || NY < j || NZ < k) {
    return;
  }
  UY(i, j, k) += DUY(i, j, k);
}

inline void update_uz(
    device const float * const duz,
    device float * const uz,
    uint3 id
) {
  const uint i = id.x + 1;
  const uint j = id.y + 1;
  const uint k = id.z + 1;
  if (NX < i || NY < j || k < 2 || NZ < k) {
    return;
  }
  UZ(i, j, k) += DUZ(i, j, k);
}

inline void update_temperature(
    device const float * const dtemperature,
    device float * const temperature,
    uint3 id
) {
  const uint i = id.x + 1;
  const uint j = id.y + 1;
  const uint k = id.z + 1;
  if (NX < i || NY < j || NZ < k) {
    return;
  }
  TEMPERATURE(i, j, k) += DTEMPERATURE(i, j, k);
}

kernel void update(
    const device float * const dux [[buffer(0)]],
    const device float * const duy [[buffer(1)]],
    const device float * const duz [[buffer(2)]],
    const device float * const dtemperature [[buffer(3)]],
    device float * const ux [[buffer(4)]],
    device float * const uy [[buffer(5)]],
    device float * const uz [[buffer(6)]],
    device float * const temperature [[buffer(7)]],
    uint3 id [[thread_position_in_grid]]
) {
  update_ux(dux, ux, id);
  update_uy(duy, uy, id);
  update_uz(duz, uz, id);
  update_temperature(dtemperature, temperature, id);
}

