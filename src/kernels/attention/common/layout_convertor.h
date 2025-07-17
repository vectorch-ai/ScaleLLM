#pragma once
#include <cute/config.hpp>
#include <cute/tensor.hpp>

namespace llm {
using namespace cute;

// Convert fragment layout for different purposes
// Only works for TiledMMA (64x16x16) with SM80_16x8x16_F32F16F16F32_TN
struct LayoutConvertor {
  // Convert fragment layout to rowcol layout for iterating
  // (MMA=4, MMA_M, MMA_N, ...) => ((2, MMA_M), (2, MMA_N), ...)
  template <typename LayoutC>
  CUTE_HOST_DEVICE static constexpr auto to_mn(const LayoutC& layout) {
    constexpr int R = LayoutC::rank;
    static_assert(R >= 3, "Expected at least 3 modes in LayoutC.");

    // ((2, 2), MMA_M, MMA_N, ...)
    auto l = logical_divide(layout, Shape<_2>{});
    // ((2, MMA_M), (2, MMA_N), ...)
    if constexpr (R > 3) {
      return make_layout(make_layout(get<0, 1>(l), get<1>(l)),
                         make_layout(get<0, 0>(l), get<2>(l)),
                         take<3, R>(l));
    } else {
      return make_layout(make_layout(get<0, 1>(l), get<1>(l)),
                         make_layout(get<0, 0>(l), get<2>(l)));
    }
  }

  // Convert fragment layout from gemm-I C to gemm-II A
  // (MMA_C=4,MMA_M,MMA_N) => (MMA_A=(4, 2), MMA_M, MMA_N/2)
  template <typename LayoutC>
  CUTE_HOST_DEVICE static constexpr auto to_mma_a(const LayoutC& layout) {
    auto l = logical_divide(layout.layout(), Shape<X, X, _2>{});
    return make_layout(
        make_layout(get<0>(l), get<2, 0>(l)), get<1>(l), get<2, 1>(l));
  }
};

}  // namespace llm
