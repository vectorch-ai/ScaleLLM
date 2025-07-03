#pragma once

// #include <cute/algorithm/tuple_algorithms.hpp>
#include <cute/config.hpp>
#include <cute/int_tuple.hpp>
#include <cute/numeric/arithmetic_tuple.hpp>

namespace cute {

//
// A arithmetic tuple iterator with a coordinate transform.
//
template <class ArithTuple, class Transform>
struct GatherArithmeticTupleIterator {
  using value_type = ArithTuple;
  using element_type = ArithTuple;
  using reference = ArithTuple;

  ArithTuple coord_;
  Transform transform_;

  CUTE_HOST_DEVICE constexpr GatherArithmeticTupleIterator(
      const ArithTuple& coord,
      const Transform& transform)
      : coord_(coord), transform_(transform) {}

  CUTE_HOST_DEVICE constexpr auto coord() const {
    // apply the transform to the coordinate
    return transform_(coord_);
  }

  CUTE_HOST_DEVICE constexpr auto operator*() const { return coord(); }

  template <class Coord>
  CUTE_HOST_DEVICE constexpr auto operator+(const Coord& c) const {
    auto coord = coord_ + c;
    return GatherArithmeticTupleIterator<remove_cvref_t<decltype(coord)>,
                                         Transform>(coord, transform_);
  }
};

template <class Tuple, class Transform>
CUTE_HOST_DEVICE constexpr auto make_gather_inttuple_iter(
    const Tuple& t,
    const Transform& transform) {
  return GatherArithmeticTupleIterator(as_arithmetic_tuple(t), transform);
}

// Generate the TMA coord tensor with transform
template <class TMA, class GShape, class Transform>
CUTE_HOST_DEVICE constexpr auto make_gather_tma_tensor(
    const TMA& tma,
    const GShape& g_shape,
    const Transform& transform) {
  static_assert(is_congruent<decltype(g_shape),
                             decltype(tma.aux_params_.g_stride_)>::value);
  auto layout = make_layout(g_shape, tma.aux_params_.g_stride_);
  return make_tensor(make_gather_inttuple_iter(coprofile(layout), transform),
                     layout);
}

//
// Display utilities
//

template <class ArithTuple, class Transform>
CUTE_HOST_DEVICE void print(
    const GatherArithmeticTupleIterator<ArithTuple, Transform>& iter) {
  printf("GatherArithTuple");
  print(iter.coord());
}

#if !defined(__CUDACC_RTC__)
template <class ArithTuple, class Transform>
CUTE_HOST std::ostream& operator<<(
    std::ostream& os,
    const GatherArithmeticTupleIterator<ArithTuple, Transform>& iter) {
  return os << "GatherArithTuple" << iter.coord();
}
#endif

}  // end namespace cute
