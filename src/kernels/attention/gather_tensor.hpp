// adapted from
// https://github.com/NVIDIA/cutlass/blob/main/examples/common/gather_tensor.hpp
#pragma once

#include "cute/layout.hpp"
#include "cute/tensor.hpp"

namespace llm {

using namespace cute;

/// Custom stride object that applies a function followed by a stride
template <class Func, class Stride>
struct CustomStride {
  CUTE_HOST_DEVICE constexpr CustomStride(const Func& func,
                                          const Stride& stride)
      : func_(func), stride_(stride) {}

  template <class I>
  CUTE_HOST_DEVICE constexpr friend auto operator*(I i, const CustomStride& s) {
    return s.func_(i) * s.stride_;
  }

  template <class I>
  CUTE_HOST_DEVICE constexpr friend auto operator*(const CustomStride& s, I i) {
    return s.func_(i) * s.stride_;
  }

  template <class Div>
  CUTE_HOST_DEVICE constexpr friend auto safe_div(const CustomStride& s,
                                                  const Div& div) {
    return CustomStride<Func, decltype(safe_div(s.stride_, div))>(
        s.func_, safe_div(s.stride_, div));
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

template <class Func, class Stride>
CUTLASS_HOST_DEVICE auto make_custom_stride_layout(Func&& func,
                                                   const Stride& stride) {
  // Use a dummy shape and replace the first non-unit stride with a custom
  // gather stride
  auto idx =
      find_if(stride, [](auto x) { return not is_constant<1, decltype(x)>{}; });
  constexpr int I = decltype(idx)::value;
  return make_layout(
      repeat_like(stride, _1{}),
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
      make_custom_stride_layout(static_cast<Func&&>(func), stride);
  return make_tensor(iter,
                     ComposedLayout{gather_layout, offset, matrix_layout});
}

}  // namespace llm
