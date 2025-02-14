#pragma once

#include <cuda.h>

#include <cute/tensor.hpp>

#include "ptx.cuh"

namespace llm {
using namespace cute;

namespace detail {
// performs a parallel reduction operation across N threads within the warp
//   - val: The value to be reduced within the warp.
template <int N, typename T>
CUTE_DEVICE T group_reduce_sum(T val) {
  static_assert((N & (N - 1)) == 0, "N must be power of 2 ");

  CUTE_UNROLL
  for (int mask = N / 2; mask > 0; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask, 32);
  }
  return val;
}

template <int N, typename T>
CUTE_DEVICE T group_reduce_max(T val) {
  static_assert((N & (N - 1)) == 0, "N must be power of 2 ");
  CUTE_UNROLL
  for (int mask = N / 2; mask > 0; mask >>= 1) {
    val = max(val, __shfl_xor_sync(0xffffffff, val, mask, 32));
  }
  return val;
}
}  // namespace detail

// online softmax kernel
template <int ROWS_PER_THR>
struct OnlineSoftmax {
  // Fragment type for row_max and row_sum
  using FragmentT = decltype(make_tensor<float>(Int<ROWS_PER_THR>{}));

  FragmentT row_max_;
  FragmentT row_sum_;
  float sm_scale_;

  CUTE_DEVICE OnlineSoftmax(float sm_scale = 1.0f) : sm_scale_(sm_scale) {
    // initialize row_max and row_sum
    fill(row_max_, float(-5e4));
    clear(row_sum_);
  }

  // computes the softmax scores and rescales the output
  // rAccS: ((2, MMA_M), (2, MMA_N))
  // rAccO: ((2, MMA_M), (2, MMA_K), STAGES)
  //  - score = exp(score - row_max`)
  //  - o = o * s_scale
  //  - internal: row_sum = row_sum * s_scale + row_sum`
  template <typename EngineS,
            typename LayoutS,
            typename EngineO,
            typename LayoutO,
            typename ReduceFunc>
  CUTE_DEVICE void rescale(Tensor<EngineS, LayoutS>& rAccS_mn,
                           Tensor<EngineO, LayoutO>& rAccO_mn,
                           ReduceFunc reduce_rowmax) {
    // row_max = max(row_max, scores)
    FragmentT pre_row_max;
    cute::copy(row_max_, pre_row_max);
    CUTE_UNROLL
    for (int si = 0; si < size<0>(rAccS_mn); ++si) {
      float row_max = row_max_(si);
      // rowmax within a thread
      CUTE_UNROLL
      for (int sj = 0; sj < size<1>(rAccS_mn); ++sj) {
        row_max = max(row_max, rAccS_mn(si, sj));
      }
      // rowmax across 4 threads
      row_max_(si) = detail::group_reduce_max<4>(row_max);
    }

    if constexpr (!std::is_same_v<ReduceFunc, std::nullptr_t>) {
      reduce_rowmax(row_max_);
    }

    // scores = exp(scores - row_max)
    CUTE_UNROLL
    for (int si = 0; si < size<0>(rAccS_mn); ++si) {
      const float rowmax_scale = row_max_(si) * sm_scale_;
      CUTE_UNROLL
      for (int sj = 0; sj < size<1>(rAccS_mn); sj++) {
        rAccS_mn(si, sj) =
            ptx::exp2(rAccS_mn(si, sj) * sm_scale_ - rowmax_scale);
      }
    }

    // row_sum = row_sum * s_scale + row_sum`
    CUTE_UNROLL
    for (int si = 0; si < size<0>(rAccS_mn); ++si) {
      const float s_scale =
          ptx::exp2((pre_row_max(si) - row_max_(si)) * sm_scale_);
      row_sum_(si) *= s_scale;
      CUTE_UNROLL
      for (int sj = 0; sj < size<1>(rAccS_mn); sj++) {
        // rowsum within a thread
        row_sum_(si) += rAccS_mn(si, sj);
      }
    }

    // o = o * s_scale
    constexpr int R = LayoutO::rank;
    auto rAccO_mn_ = group_modes<1, R>(rAccO_mn);
    CUTE_UNROLL
    for (int si = 0; si < size<0>(rAccO_mn_); ++si) {
      const float s_scale =
          ptx::exp2((pre_row_max(si) - row_max_(si)) * sm_scale_);
      CUTE_UNROLL
      for (int sj = 0; sj < size<1>(rAccO_mn_); ++sj) {
        rAccO_mn_(si, sj) *= s_scale;
      }
    }
  }

  template <typename EngineS,
            typename LayoutS,
            typename EngineO,
            typename LayoutO>
  CUTE_DEVICE void rescale(Tensor<EngineS, LayoutS>& rAccS_mn,
                           Tensor<EngineO, LayoutO>& rAccO_mn) {
    rescale(rAccS_mn, rAccO_mn, nullptr);
  }

  // finalizes the softmax computation with o = o / row_sum
  // rAccO: ((2, MMA_M), (2, MMA_K), STAGES)
  template <typename EngineO, typename LayoutO, typename ReduceFunc>
  CUTE_DEVICE void finalize(Tensor<EngineO, LayoutO>& rAccO_mn,
                            ReduceFunc reduce_rowsum) {
    CUTE_UNROLL
    for (int i = 0; i < size(row_sum_); ++i) {
      // rowsum across 4 threads
      row_sum_(i) = detail::group_reduce_sum<4>(row_sum_(i));
    }

    if constexpr (!std::is_same_v<ReduceFunc, std::nullptr_t>) {
      reduce_rowsum(row_sum_);
    }

    // o = o / row_sum
    constexpr int R = LayoutO::rank;
    auto rAccO_mn_ = group_modes<1, R>(rAccO_mn);
    CUTE_UNROLL
    for (int oi = 0; oi < size<0>(rAccO_mn_); ++oi) {
      CUTE_UNROLL
      for (int oj = 0; oj < size<1>(rAccO_mn_); ++oj) {
        rAccO_mn_(oi, oj) *= ptx::rcp(row_sum_(oi));
      }
    }
  }

  template <typename EngineO, typename LayoutO>
  CUTE_DEVICE void finalize(Tensor<EngineO, LayoutO>& rAccO_mn) {
    finalize(rAccO_mn, nullptr);
  }
};

}  // namespace llm