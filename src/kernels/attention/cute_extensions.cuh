#pragma once

#include <cute/atom/mma_atom.hpp>
#include <cute/tensor.hpp>

#include "cute/config.hpp"
#include "cute/layout.hpp"

namespace cute {

namespace detail {
// Trait that detects if atom's traits has a member function with(bool)
template <class, class Enable = void>
constexpr bool has_with_bool = false;

template <class Copy_Atom>
constexpr bool
    has_with_bool<Copy_Atom,
                  cute::void_t<decltype(declval<typename Copy_Atom::Traits>()
                                            .with(declval<bool>()))>> = true;

}  // namespace detail

template <int... Is, int B, int M, int S, class Offset, class LayoutB>
CUTE_HOST_DEVICE constexpr auto select(
    const ComposedLayout<Swizzle<B, M, S>, Offset, LayoutB>& c) {
  return composition(c.layout_a(), c.offset(), select<Is...>(c.layout_b()));
}

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
  if constexpr (detail::has_with_bool<Copy_Atom>) {
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
          class Coord,
          __CUTE_REQUIRES(TensorS::rank == 3 && TensorD::rank == 3 &&
                          TensorC::rank == 3)>
CUTE_HOST_DEVICE void safe_copy(
    const TiledCopy<CopyAtom, TV, Tiler>& tiled_copy,
    const TensorS& src,       // (CPY, CPY_M/N, CPY_K)
    TensorD& dst,             // (CPY, CPY_M/N, CPY_K)
    const TensorC& identity,  // (CPY, CPY_M/N, CPY_K) -> (blk_m/n, blk_k)
    const Coord& max_coord    // max_coord(blk_m/n, blk_k)
) {
  CUTE_STATIC_ASSERT_V(size<0>(src) == size<0>(dst));       // CPY == CPY
  CUTE_STATIC_ASSERT_V(size<0>(src) == size<0>(identity));  // CPY == CPY
  CUTE_STATIC_ASSERT_V(size<1>(src) == size<1>(dst));       // CPY_M/N
  CUTE_STATIC_ASSERT_V(size<1>(src) == size<1>(identity));  // CPY_M/N
  CUTE_STATIC_ASSERT_V(size<2>(src) == size<2>(dst));       // CPY_K
  CUTE_STATIC_ASSERT_V(size<2>(src) == size<2>(identity));  // CPY_K

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
          class Coord,
          __CUTE_REQUIRES(TensorS::rank == 3 && TensorD::rank == 3 &&
                          TensorC::rank == 3)>
CUTE_HOST_DEVICE void safe_copy(
    const TiledCopy<CopyAtom, TV, Tiler>& tiled_copy,
    const TensorS& src,       // (CPY, CPY_M/N, CPY_K)
    TensorD&& dst,            // (CPY, CPY_M/N, CPY_K)
    const TensorC& identity,  // (CPY, CPY_M/N, CPY_K) -> (blk_m/n, blk_k)
    const Coord& max_coord    // max_coord(blk_m/n, blk_k)
) {
  safe_copy<EVEN_MN, EVEN_K, ZFILL_MN, ZFILL_K>(
      tiled_copy, src, dst, identity, max_coord);
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
          class Coord,
          __CUTE_REQUIRES(TensorS::rank == 4 && TensorD::rank == 4 &&
                          TensorC::rank == 3)>
CUTE_HOST_DEVICE void safe_copy(
    const TiledCopy<CopyAtom, TV, Tiler>& tiled_copy,
    const TensorS& src,       // (CPY, CPY_M/N, CPY_K, k)
    TensorD& dst,             // (CPY, CPY_M/N, CPY_K, k)
    const TensorC& identity,  // (CPY, CPY_M/N, CPY_K) -> (blk_m/n, blk_k)
    const Coord& max_coord    // max_coord(blk_m/n, blk_k)
) {
  CUTE_UNROLL
  for (int k = 0; k < size<3>(src); ++k) {
    safe_copy<EVEN_MN, EVEN_K, ZFILL_MN, ZFILL_K>(
        tiled_copy, src(_, _, _, k), dst(_, _, _, k), identity, max_coord);
  }
}

template <bool EVEN_K,
          bool ZFILL_K,
          class CopyAtom,
          class TV,
          class Tiler,
          class TensorS,
          class TensorD,
          class TensorC,
          class Coord,
          __CUTE_REQUIRES(TensorS::rank == 2 && TensorD::rank == 2 &&
                          TensorC::rank == 2)>
CUTE_HOST_DEVICE void safe_copy(
    const TiledCopy<CopyAtom, TV, Tiler>& tiled_copy,
    const TensorS& src,       // (CPY, CPY_K)
    TensorD& dst,             // (CPY, CPY_K)
    const TensorC& identity,  // (CPY, CPY_K) -> (blk_k)
    const Coord& max_coord    // max_coord(blk_k)
) {
  CUTE_STATIC_ASSERT_V(size<0>(src) == size<0>(dst));       // CPY == CPY
  CUTE_STATIC_ASSERT_V(size<0>(src) == size<0>(identity));  // CPY == CPY
  CUTE_STATIC_ASSERT_V(size<1>(src) == size<1>(dst));       // CPY_K == CPY_K
  CUTE_STATIC_ASSERT_V(size<1>(src) == size<1>(identity));  // CPY_K == CPY_K

  auto copy_atom = static_cast<const CopyAtom&>(tiled_copy);

  if constexpr (!EVEN_K) {
    // handle k oob
    CUTE_UNROLL
    for (int ki = 0; ki < size<1>(src); ++ki) {
      if (elem_less<0>(identity(_0{}, ki), max_coord)) {
        copy(copy_atom, src(_, ki), dst(_, ki));
      } else if constexpr (ZFILL_K) {
        zfill(copy_atom, src(_, ki), dst(_, ki));
      }
    }
  } else {
    // no oob, just copy
    copy(copy_atom, src, dst);
  }
}

}  // namespace cute
