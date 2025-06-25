#pragma once
#include <cute/config.hpp>
#include <cute/tensor.hpp>

#include "fast_math.h"

namespace llm {
using namespace cute;

template <int ROWS_PER_THR, bool ALIBI, bool LOCAL>
struct Mask {
  // Fragment type for alibi slopes
  using FragmentT = decltype(make_tensor<float>(Int<ROWS_PER_THR>{}));

  const int kv_len_;
  const FastDivmod& group_size_;
  const int sliding_window_;
  const int diagonal_offset_;

  FragmentT alibi_slopes_;

  CUTE_HOST_DEVICE Mask(int q_len,
                        int kv_len,
                        const FastDivmod& group_size,
                        int sliding_window)
      : kv_len_(kv_len),
        group_size_(group_size),
        sliding_window_(sliding_window),
        diagonal_offset_(kv_len - q_len) {}

  // cS_mn: ((2, MMA_M), (2, MMA_N))
  template <typename IdentityS>
  CUTE_HOST_DEVICE void init_alibi(
      IdentityS& cS_mn,  // ((2, MMA_M), (2, MMA_N)) => (M, N)
      int kv_head_idx,
      float sm_scale,
      const float* alibi_slops_ptr) {
    // copy alibi slopes to registers
    CUTE_UNROLL
    for (int i = 0; i < size<0>(cS_mn); ++i) {
      const int q_packed_idx = get<0>(cS_mn(i, _0{}));
      const int offset = q_packed_idx % group_size_;
      const int head_idx = (kv_head_idx * group_size_) + offset;
      alibi_slopes_(i) = alibi_slops_ptr[head_idx] / sm_scale;
    }
  }

  // rS_mn/cS_mn: ((2, MMA_M), (2, MMA_N))
  template <bool OOB_MASK, typename FragmentS, typename IdentityS>
  CUTE_HOST_DEVICE void apply(
      FragmentS& rS_mn,  // ((2, MMA_M), (2, MMA_N))
      IdentityS& cS_mn   // ((2, MMA_M), (2, MMA_N)) => (M, N)
  ) const {
    CUTE_UNROLL
    for (int i = 0; i < size<0>(rS_mn); ++i) {
      const auto alibi_slope = ALIBI ? alibi_slopes_(i) : 0.0f;
      CUTE_UNROLL
      for (int j = 0; j < size<1>(rS_mn); ++j) {
        const auto [m, n] = cS_mn(i, j);
        const int q_packed_idx = m;
        const int kv_idx = n;
        const int q_idx = (q_packed_idx / group_size_) + diagonal_offset_;

        const bool out_of_boundary = [&]() {
          if constexpr (OOB_MASK && LOCAL) {
            // causal + oob mask + local mask
            return kv_idx > q_idx || kv_idx >= kv_len_ ||
                   (q_idx - kv_idx) > sliding_window_;
          } else if constexpr (OOB_MASK && !LOCAL) {
            // causal + oob mask
            return kv_idx > q_idx || kv_idx >= kv_len_;
          } else if constexpr (!OOB_MASK && LOCAL) {
            // local mask
            return (q_idx - kv_idx) > sliding_window_;
          } else {
            // !OOB_MASK && !LOCAL
            return false;
          }
        }();

        if (out_of_boundary) {
          rS_mn(i, j) = -INFINITY;
        } else if constexpr (ALIBI) {
          // Apply alibi bias to the attention scores
          rS_mn(i, j) += alibi_slope * kv_idx;
        }
      }
    }
  }
};

}  // namespace llm
