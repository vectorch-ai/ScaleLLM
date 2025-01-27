#pragma once

#include <cute/atom/mma_atom.hpp>
#include <cute/tensor.hpp>

#include "cute/config.hpp"
#include "cute/layout.hpp"

namespace cute {

template <size_t I, class IntTupleA, class IntTupleB>
CUTE_HOST_DEVICE constexpr auto elem_less(IntTupleA const& a,
                                          IntTupleB const& b) {
  return elem_less(get<I>(a), get<I>(b));
}

template <class Copy_Atom, class TensorS, class TensorD>
CUTE_HOST_DEVICE void zfill(const Copy_Atom& copy_atom,
                            const TensorS& src,
                            TensorD&& dst) {
  CUTE_STATIC_ASSERT(TensorS::rank == TensorD::rank, "rank-mismatch.");

  auto has_with_bool = cute::is_valid(
      [](auto t) -> void_t<decltype(declval<typename decltype(t)::Traits>()
                                        .with(true))> {},
      copy_atom);
  if constexpr (has_with_bool) {
    constexpr int R = TensorD::rank;
    if constexpr (R == 1) {  // Dispatch the copy
      copy_atom.with(false).call(src, dst);
    } else {  // Loop over all but the first mode
      Tensor src_v = group_modes<1, R>(src);
      Tensor dst_v = group_modes<1, R>(dst);
      CUTE_UNROLL
      for (int i = 0; i < size<1>(dst_v); ++i) {
        copy_atom.with(false).call(src_v(_, i), dst_v(_, i));
      }
    }
  } else {
    // just call clear if no with method
    clear(dst);
  }
}

template <class Copy_Atom, class TensorS, class TensorD>
CUTE_HOST_DEVICE void zfill(const Copy_Atom& copy_atom,
                            const TensorS& src,
                            TensorD& dst) {
  zfill(copy_atom, src, dst);
}

template <bool EVEN_MN,
          bool EVEN_K,
          bool ZFILL_MN,
          bool ZFILL_K,
          class CopyAtom,
          class TV,
          class Tiler,
          class TensorS,
          class TensorD,
          class TensorC,
          class Coord>
CUTE_HOST_DEVICE void safe_copy(
    const TiledCopy<CopyAtom, TV, Tiler>& tiled_copy,
    const TensorS& src,       // (CPY, CPY_M/N, CPY_K)
    TensorD& dst,             // (CPY, CPY_M/N, CPY_K)
    const TensorC& identity,  // (CPY, CPY_M/N, CPY_K) -> (blk_m/n, blk_k)
    const Coord& max_coord    // max_coord(blk_m/n, blk_k)
) {
  CUTE_STATIC_ASSERT(TensorS::rank == TensorD::rank, "rank-mismatch.");
  auto copy_atom = static_cast<const CopyAtom&>(tiled_copy);

  if constexpr (!EVEN_MN && !EVEN_K) {
    // handle both m/n and k oob
    CUTE_UNROLL
    for (int mi = 0; mi < size<1>(src); ++mi) {
      if (elem_less<0>(identity(_0{}, mi, _0{}), max_coord)) {
        CUTE_UNROLL
        for (int ki = 0; ki < size<2>(src); ++ki) {
          if (elem_less<1>(identity(_0{}, _0{}, ki), max_coord)) {
            copy(copy_atom, src(_, mi, ki), dst(_, mi, ki));
          } else if constexpr (ZFILL_K) {
            zfill(copy_atom, src(_, mi, ki), dst(_, mi, ki));
          }
        }
      } else if constexpr (ZFILL_MN) {
        zfill(copy_atom, src(_, mi, _), dst(_, mi, _));
      } else if constexpr (ZFILL_K) {
        // still need to handle k oob even if m/n is not zfilled
        CUTE_UNROLL
        for (int ki = 0; ki < size<2>(src); ++ki) {
          if (!elem_less<1>(identity(_0{}, _0{}, ki), max_coord)) {
            zfill(copy_atom, src(_, mi, ki), dst(_, mi, ki));
          }
        }
      }
    }
  } else if constexpr (!EVEN_MN && EVEN_K) {
    // only handle m/n oob
    CUTE_UNROLL
    for (int mi = 0; mi < size<1>(src); ++mi) {
      if (elem_less<0>(identity(_0{}, mi, _0{}), max_coord)) {
        copy(copy_atom, src(_, mi, _), dst(_, mi, _));
      } else if constexpr (ZFILL_MN) {
        zfill(copy_atom, src(_, mi, _), dst(_, mi, _));
      }
    }
  } else if constexpr (EVEN_MN && !EVEN_K) {
    // only handle k oob
    CUTE_UNROLL
    for (int ki = 0; ki < size<2>(src); ++ki) {
      if (elem_less<1>(identity(_0{}, _0{}, ki), max_coord)) {
        copy(copy_atom, src(_, _, ki), dst(_, _, ki));
      } else if constexpr (ZFILL_K) {
        zfill(copy_atom, src(_, _, ki), dst(_, _, ki));
      }
    }
  } else {
    // no oob, just copy
    copy(copy_atom, src, dst);
  }
}

}  // namespace cute