#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cute/numeric/numeric_types.hpp>
#include <cute/tensor.hpp>

namespace llm {

namespace detail {
using namespace cute;
template <typename SRC_TYPE, typename DST_TYPE>
struct type_cast {
  static_assert(dependent_false<SRC_TYPE>, "not implemented");
};

// specialization for float -> half
template <>
struct type_cast<float, cute::half_t> {
  template <typename FragmentS, typename FragmentD>
  CUTE_DEVICE static void cast(const FragmentS& src, FragmentD& dst) {
    auto src2 = recast<float2>(src);
    auto dst2 = recast<half2>(dst);

    CUTE_UNROLL
    for (int i = 0; i < size(src2); ++i) {
      dst2(i) = __float22half2_rn(src2(i));
    }
  }
};

// specialization for float -> bfloat16
template <>
struct type_cast<float, cute::bfloat16_t> {
  template <typename FragmentS, typename FragmentD>
  CUTE_DEVICE static void cast(const FragmentS& src, FragmentD& dst) {
    auto src2 = recast<float2>(src);
    auto dst2 = recast<nv_bfloat162>(dst);

    CUTE_UNROLL
    for (int i = 0; i < size(src2); ++i) {
      dst2(i) = __float22bfloat162_rn(src2(i));
    }
  }
};
// TODO: add other specializations

}  // namespace detail

// dispatch to right type_cast
// functionality: dst = (DST_TYPE)src
template <typename FragmentS, typename FragmentD>
CUTE_DEVICE void fast_cast(const FragmentS& src, FragmentD& dst) {
  CUTE_STATIC_ASSERT_V((cute::size(src) == cute::size(dst)), "size mismatch");

  using TypeSrc = typename FragmentS::value_type;
  using TypeDst = typename FragmentD::value_type;

  // dispatch to type_cast
  detail::type_cast<TypeSrc, TypeDst>::cast(src, dst);
}

}  // namespace llm