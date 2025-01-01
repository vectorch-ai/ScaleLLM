#pragma once

#include <cuda.h>
// #include <cuda_runtime.h>

#include <cute/tensor.hpp>

namespace llm {
using namespace cute;

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

// online softmax kernel
template <int ROWS_PER_THREAD>
struct OnlineSoftmax {
  // Fragment type for row_max and row_sum
  using FragmentT = decltype(make_tensor<float>(Shape<Int<ROWS_PER_THREAD>>{}));

  FragmentT row_max_;
  FragmentT row_sum_;

  __device__ OnlineSoftmax() {
    // initialize row_max and row_sum
    fill(row_max_, float(-5e4));
    clear(row_sum_);
  }

  // computes the softmax scores and rescales the output
  //  - score = exp(score - row_max`)
  //  - O = O * s_scale
  //  - internal: row_sum = row_sum * s_scale + row_sum`
  template <typename FragmentS, typename FragmentO>
  CUTE_DEVICE void rescale(FragmentS& rAccS, FragmentO& rAccO) {
    CUTE_UNROLL
    for (int si = 0; si < size<0>(rAccS); si++) {
      // rowmax across 4 threads
      float cur_rowmax = row_max_(si);
      CUTE_UNROLL
      for (int sj = 0; sj < size<1>(rAccS); sj++) {
        cur_rowmax = max(cur_rowmax, rAccS(si, sj));
      }
      cur_rowmax = group_reduce_max<4>(cur_rowmax);

      // use local rowsum
      float cur_rowsum = 0;
      CUTE_UNROLL
      for (int sj = 0; sj < size<1>(rAccS); sj++) {
        rAccS(si, sj) = exp2f(rAccS(si, sj) - cur_rowmax);
        cur_rowsum += rAccS(si, sj);
      }

      // scores_scale = exp(max - cur_rowmax)
      const float scores_scale = exp2f(row_max_(si) - cur_rowmax);
      row_max_(si) = cur_rowmax;

      // o_2 = o_1 * s_scale
      CUTE_UNROLL
      for (int sj = 0; sj < size<1>(rAccO); sj++) {
        rAccO(si, sj) *= scores_scale;
      }

      // s_2 = s_1 * s_scale + row_sum
      row_sum_(si) = row_sum_(si) * scores_scale + cur_rowsum;
    }
  }

  // finalizes the softmax computation with O = O / row_sum
  template <typename FragmentO>
  CUTE_DEVICE void finalize(FragmentO& rAccO) {
    CUTE_UNROLL
    for (int oi = 0; oi < size<0>(rAccO); oi++) {
      // rowsum across 4 threads
      row_sum_(oi) = group_reduce_sum<4>(row_sum_(oi));

      CUTE_UNROLL
      for (int oj = 0; oj < size<1>(rAccO); oj++) {
        rAccO(oi, oj) /= row_sum_(oi);
      }
    }
  }
};

}  // namespace llm