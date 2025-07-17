// adapted from
// https://github.com/NVIDIA/cutlass/blob/main/examples/common/gather_tensor.hpp
#pragma once

#include <cute/layout.hpp>
#include <cute/layout_composed.hpp>
#include <cute/tensor.hpp>
namespace llm {

using namespace cute;

namespace detail {

// every stride must be divisible by div
template <class Stride, class Div>
CUTE_HOST_DEVICE constexpr auto safe_stride_div(Stride const& s,
                                                const Div& div) {
  if constexpr (is_tuple<Stride>::value) {
    return transform(s, [&](auto const& a) { return safe_stride_div(a, div); });
  } else {
    return safe_div(s, div);
  }
  CUTE_GCC_UNREACHABLE;
}

}  // namespace detail

/// Custom stride object that applies a function followed by a stride
template <class Func, class Stride>
struct CustomStride {
  CUTE_HOST_DEVICE constexpr CustomStride(const Func& func,
                                          const Stride& stride)
      : func_(func), stride_(stride) {}

  template <class I>
  CUTE_HOST_DEVICE constexpr friend auto operator*(I i, const CustomStride& s) {
    return inner_product(s.func_(i), s.stride_);
  }

  template <class I>
  CUTE_HOST_DEVICE constexpr friend auto operator*(const CustomStride& s, I i) {
    return inner_product(s.func_(i), s.stride_);
  }

  template <class Div>
  CUTE_HOST_DEVICE constexpr friend auto safe_div(const CustomStride& s,
                                                  const Div& div) {
    auto stride = detail::safe_stride_div(s.stride_, div);
    return CustomStride<Func, decltype(stride)>(s.func_, stride);
  }

  template <class Shape>
  CUTE_HOST_DEVICE constexpr friend auto make_layout(
      const Shape& shape,
      const CustomStride& stride) {
    return Layout<Shape, CustomStride>(shape, stride);
  }

  CUTE_HOST_DEVICE friend void print(CustomStride const& s) {
    print("CustomStride{func,");
    print(s.stride_);
    print("}");
  }

  Func func_;
  Stride stride_;
};

template <class Func, class Shape, class Stride>
CUTLASS_HOST_DEVICE auto make_custom_stride_layout(Func&& func,
                                                   const Shape& shape,
                                                   const Stride& stride) {
  // Use a dummy shape and replace the first non-unit stride with a custom
  // gather stride
  auto idx =
      find_if(stride, [](auto x) { return not is_constant<1, decltype(x)>{}; });
  constexpr int I = decltype(idx)::value;
  return make_layout(
      repeat_like(shape, _1{}),
      replace<I>(stride,
                 CustomStride{static_cast<Func&&>(func), get<I>(stride)}));
}

/// Helper function to optionally create a gather tensor
template <class Iterator, class Shape, class Stride, class Func>
CUTLASS_HOST_DEVICE auto make_gather_tensor(Iterator iter,
                                            const Shape& shape,
                                            const Stride& stride,
                                            Func&& func) {
  Layout matrix_layout = make_identity_layout(shape);
  auto offset = as_arithmetic_tuple(repeat_like(shape, _0{}));
  Layout gather_layout =
      make_custom_stride_layout(static_cast<Func&&>(func), shape, stride);
  return make_tensor(iter,
                     ComposedLayout{gather_layout, offset, matrix_layout});
}

}  // namespace llm

namespace cute {

template <int N, int I, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr auto upcast(Shape const& shape,
                                       Stride const& stride) {
  if constexpr (is_tuple<Shape>::value) {
    return transform_layout(shape, stride, [](auto const& s, auto const& d) {
      return upcast<N, I>(s, d);
    });
  } else if constexpr (is_scaled_basis<Stride>::value) {
    if constexpr (Stride::mode() == I) {
      return make_layout(ceil_div(shape, Int<N>{}), ceil_div(stride, Int<N>{}));
    } else {
      return make_layout(shape, stride);
    }
  } else {
    return upcast<N>(shape, stride);
  }

  CUTE_GCC_UNREACHABLE;
}

template <int N,
          class OuterShape,
          class OuterStride,
          class Offset,
          class Shape,
          class Stride>
CUTE_HOST_DEVICE constexpr auto upcast(
    ComposedLayout<Layout<OuterShape, OuterStride>,
                   Offset,
                   Layout<Shape, Stride>> const& layout) {
  // Find index of the stride-1 mode - that is the only one that requires
  // updating inner shape and offset
  auto idx = find_if(layout.layout_a().stride(),
                     [](auto x) { return is_constant<1, decltype(x)>{}; });
  constexpr int I = decltype(idx)::value;

  // Upcast the outer layout (works as expected)
  auto outer = upcast<N>(layout.layout_a());

  // Upcast the accumulated offset along stride-1 mode
  auto offset = as_arithmetic_tuple(
      replace<I>(layout.offset(), upcast<N>(get<I>(layout.offset()))));

  // Upcast the inner layout's shape along stride-1 mode
  auto inner =
      upcast<N, I>(layout.layout_b().shape(), layout.layout_b().stride());

  return composition(outer, offset, inner);
}

template <class ShapeA,
          class StrideA,
          class OuterShapeB,
          class OuterStrideB,
          class OffsetB,
          class ShapeB,
          class StrideB>
CUTE_HOST_DEVICE constexpr auto max_common_vector(
    Layout<ShapeA, StrideA> const& a,
    ComposedLayout<Layout<OuterShapeB, OuterStrideB>,
                   OffsetB,
                   Layout<ShapeB, StrideB>> const& b) {
  return max_common_vector(b.layout_b(), a);
}

}  // namespace cute
