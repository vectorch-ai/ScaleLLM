#pragma once

#include <cute/atom/mma_atom.hpp>
#include <cute/config.hpp>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>

namespace llm {
using namespace cute;

template <size_t I, class IntTupleA, class IntTupleB>
CUTE_HOST_DEVICE constexpr auto elem_less(IntTupleA const& a,
                                          IntTupleB const& b) {
  return cute::elem_less(get<I>(a), get<I>(b));
}

template <bool EVEN_K,
          bool ZFILL_M,
          bool ZFILL_K,
          class CopyAtom,
          class TV,
          class Tiler,
          class SrcTensor,
          class DstTensor,
          class PrdTensor,
          class IdenTensor,
          class ResidueMK,
          __CUTE_REQUIRES(SrcTensor::rank == 3 && DstTensor::rank == 3 &&
                          IdenTensor::rank == 3)>
CUTE_HOST_DEVICE void safe_copy_m(
    const TiledCopy<CopyAtom, TV, Tiler>& tiled_copy,
    const SrcTensor& src,        // (CPY, CPY_M, CPY_K)
    DstTensor& dst,              // (CPY, CPY_M, CPY_K)
    const PrdTensor& pred_m,     // (CPY_M) -> bool
    const IdenTensor& identity,  // (CPY, CPY_M, CPY_K) -> (m, k)
    const ResidueMK& residue_mk  // (m, k)
) {
  CUTE_STATIC_ASSERT_V(size<0>(src) == size<0>(dst));       // CPY == CPY
  CUTE_STATIC_ASSERT_V(size<0>(src) == size<0>(identity));  // CPY == CPY
  CUTE_STATIC_ASSERT_V(size<1>(src) == size<1>(dst));       // CPY_M
  CUTE_STATIC_ASSERT_V(size<1>(src) == size<1>(identity));  // CPY_M
  CUTE_STATIC_ASSERT_V(size<1>(src) == size(pred_m));       // CPY_M
  CUTE_STATIC_ASSERT_V(size<2>(src) == size<2>(dst));       // CPY_K
  CUTE_STATIC_ASSERT_V(size<2>(src) == size<2>(identity));  // CPY_K

  auto copy_atom = static_cast<const CopyAtom&>(tiled_copy);
  if constexpr (!EVEN_K) {
    CUTE_UNROLL
    for (int mi = 0; mi < size<1>(src); ++mi) {
      if (pred_m(mi)) {
        CUTE_UNROLL
        for (int ki = 0; ki < size<2>(src); ++ki) {
          if (elem_less<1>(identity(_0{}, _0{}, ki), residue_mk)) {
            copy(copy_atom, src(_, mi, ki), dst(_, mi, ki));
          } else if constexpr (ZFILL_K) {
            clear(dst(_, mi, ki));
          }
        }
      } else if constexpr (ZFILL_M) {
        clear(dst(_, mi, _));
      }
    }
  } else {
    CUTE_UNROLL
    for (int mi = 0; mi < size<1>(src); ++mi) {
      if (pred_m(mi)) {
        copy(copy_atom, src(_, mi, _), dst(_, mi, _));
      } else if constexpr (ZFILL_M) {
        clear(dst(_, mi, _));
      }
    }
  }
}

template <bool EVEN_N,
          bool EVEN_K,
          bool ZFILL_N,
          bool ZFILL_K,
          class CopyAtom,
          class TV,
          class Tiler,
          class SrcTensor,
          class DstTensor,
          class IdenTensor,
          class ResidueNK,
          __CUTE_REQUIRES(SrcTensor::rank == 3 && DstTensor::rank == 3 &&
                          IdenTensor::rank == 3)>
CUTE_HOST_DEVICE void safe_copy_n(
    const TiledCopy<CopyAtom, TV, Tiler>& tiled_copy,
    const SrcTensor& src,        // (CPY, CPY_N, CPY_K)
    DstTensor& dst,              // (CPY, CPY_N, CPY_K)
    const IdenTensor& identity,  // (CPY, CPY_N, CPY_K) -> (n, k)
    const ResidueNK& residue_nk  // (n, k)
) {
  CUTE_STATIC_ASSERT_V(size<0>(src) == size<0>(dst));       // CPY == CPY
  CUTE_STATIC_ASSERT_V(size<0>(src) == size<0>(identity));  // CPY == CPY
  CUTE_STATIC_ASSERT_V(size<1>(src) == size<1>(dst));       // CPY_N
  CUTE_STATIC_ASSERT_V(size<1>(src) == size<1>(identity));  // CPY_N
  CUTE_STATIC_ASSERT_V(size<2>(src) == size<2>(dst));       // CPY_K
  CUTE_STATIC_ASSERT_V(size<2>(src) == size<2>(identity));  // CPY_K

  auto copy_atom = static_cast<const CopyAtom&>(tiled_copy);
  if constexpr (!EVEN_N && !EVEN_K) {
    // handle both n and k oob
    CUTE_UNROLL
    for (int ni = 0; ni < size<1>(src); ++ni) {
      if (elem_less<0>(identity(_0{}, ni, _0{}), residue_nk)) {
        CUTE_UNROLL
        for (int ki = 0; ki < size<2>(src); ++ki) {
          if (elem_less<1>(identity(_0{}, _0{}, ki), residue_nk)) {
            copy(copy_atom, src(_, ni, ki), dst(_, ni, ki));
          } else if constexpr (ZFILL_K) {
            clear(dst(_, ni, ki));
          }
        }
      } else if constexpr (ZFILL_N) {
        clear(dst(_, ni, _));
      }
    }
  } else if constexpr (!EVEN_N && EVEN_K) {
    // only handle n oob
    CUTE_UNROLL
    for (int mi = 0; mi < size<1>(src); ++mi) {
      if (elem_less<0>(identity(_0{}, mi, _0{}), residue_nk)) {
        copy(copy_atom, src(_, mi, _), dst(_, mi, _));
      } else if constexpr (ZFILL_N) {
        clear(dst(_, mi, _));
      }
    }
  } else if constexpr (EVEN_N && !EVEN_K) {
    // only handle k oob
    CUTE_UNROLL
    for (int ki = 0; ki < size<2>(src); ++ki) {
      if (elem_less<1>(identity(_0{}, _0{}, ki), residue_nk)) {
        copy(copy_atom, src(_, _, ki), dst(_, _, ki));
      } else if constexpr (ZFILL_K) {
        clear(dst(_, _, ki));
      }
    }
  } else {
    // no oob, just copy
    copy(copy_atom, src, dst);
  }
}

// Accept mutable temporaries
template <bool EVEN_K,
          bool ZFILL_M,
          bool ZFILL_K,
          class CopyPolicy,
          class SrcTensor,
          class DstTensor,
          class PrdTensor,
          class IdenTensor,
          class ResidueMK>
CUTE_HOST_DEVICE void safe_copy_m(
    const CopyPolicy& tiled_copy,
    const SrcTensor& src,        // (CPY, CPY_M, CPY_K)
    DstTensor&& dst,             // (CPY, CPY_M, CPY_K)
    const PrdTensor& pred_m,     // (CPY_M) -> bool
    const IdenTensor& identity,  // (CPY, CPY_M, CPY_K) -> (m, k)
    const ResidueMK& residue_mk  // (m, k)
) {
  return safe_copy_m<EVEN_K, ZFILL_M, ZFILL_K>(
      tiled_copy, src, dst, pred_m, identity, residue_mk);
}

template <bool EVEN_N,
          bool EVEN_K,
          bool ZFILL_N,
          bool ZFILL_K,
          class CopyPolicy,
          class SrcTensor,
          class DstTensor,
          class IdenTensor,
          class ResidueNK>
CUTE_HOST_DEVICE void safe_copy_n(
    const CopyPolicy& tiled_copy,
    const SrcTensor& src,        // (CPY, CPY_N, CPY_K)
    DstTensor&& dst,             // (CPY, CPY_N, CPY_K)
    const IdenTensor& identity,  // (CPY, CPY_N, CPY_K) -> (n, k)
    const ResidueNK& residue_nk  // (n, k)
) {
  return safe_copy_n<EVEN_N, EVEN_K, ZFILL_N, ZFILL_K>(
      tiled_copy, src, dst, identity, residue_nk);
}

}  // namespace llm
