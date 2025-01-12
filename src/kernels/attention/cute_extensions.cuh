#include <cute/atom/mma_atom.hpp>
#include <cute/swizzle_layout.hpp>
#include <cute/tensor.hpp>

#include "cute/config.hpp"

namespace cute {

template <class... Args, class ATensor>
CUTE_HOST_DEVICE constexpr auto partition_fragment_A(
    const ThrMMA<Args...>& thr_mma,
    const ATensor& atensor) {
  auto atensor_noswizzle =
      make_tensor(atensor.data(), get_nonswizzle_portion(atensor.layout()));
  return thr_mma.partition_fragment_A(atensor_noswizzle);
}

template <class... Args, class BTensor>
CUTE_HOST_DEVICE constexpr auto partition_fragment_B(
    const ThrMMA<Args...>& thr_mma,
    const BTensor& btensor) {
  auto btensor_noswizzle =
      make_tensor(btensor.data(), get_nonswizzle_portion(btensor.layout()));
  return thr_mma.partition_fragment_B(btensor_noswizzle);
}

template <bool Clear_OOB,
          class TiledCopy,
          class TensorS,
          class TensorD,
          class TensorC>
CUTE_HOST_DEVICE void safe_copy(
    const TiledCopy& tiled_copy,
    const TensorS& src,       // (CPY, CPY_M/N, CPY_K)
    TensorD& dst,             // (CPY, CPY_M/N, CPY_K)
    const TensorC& identity,  // (CPY, CPY_M/N, CPY_K) -> (blk_m/n, blk_k)
    const int max_coord       // max_coord_m/n
) {
  CUTE_UNROLL
  for (int m = 0; m < size<1>(src); ++m) {
    if (get<0>(identity(0, m, 0)) < max_coord) {
      cute::copy(tiled_copy, src(_, m, _), dst(_, m, _));
    } else {
      if constexpr (Clear_OOB) {
        cute::clear(dst(_, m, _));
      }
    }
  }
}

}  // namespace cute