#pragma once
#include <cute/config.hpp>
#include <cute/tensor.hpp>

namespace llm {
using namespace cute;

template <int BLK_M,
          int BLK_N,
          int ROWS_PER_MMA,
          int MMA_M,
          bool ALIBI,
          bool LOCAL>
struct Mask {
  // Fragment type for alibi slopes: (2, MMA_M)
  using FragmentT =
      decltype(make_tensor<float>(Shape<Int<ROWS_PER_MMA>, Int<MMA_M>>{}));

  int q_len_;
  int kv_len_;
  int group_size_;
  int sliding_window_;

  int lane_idx_;
  int m_base_idx_;
  int diagonal_offset_;

  FragmentT alibi_slopes_;

  CUTE_HOST_DEVICE Mask(int tidx,
                        int m_block,
                        int q_len,
                        int kv_len,
                        int kv_head_idx,
                        int group_size,
                        int sliding_window,
                        float sm_scale,
                        const float* alibi_slops_ptr)
      : q_len_(q_len),
        kv_len_(kv_len),
        group_size_(group_size),
        sliding_window_(sliding_window) {
    lane_idx_ = tidx % 32;
    // Warp layout 4x1, each warp processes 16 rows (4 threads per row)
    m_base_idx_ = m_block * BLK_M + tidx / 32 * 16 + lane_idx_ / 4;
    diagonal_offset_ = kv_len - q_len;

    if constexpr (ALIBI) {
      // copy alibi slopes to registers
      CUTE_UNROLL
      for (int mi = 0; mi < MMA_M; ++mi) {                    //  MMA_M
        const int q_packed_idx_base = m_base_idx_ + mi * 64;  // stride = 64
        CUTE_UNROLL
        for (int i = 0; i < ROWS_PER_MMA; ++i) {               // 2
          const int q_packed_idx = q_packed_idx_base + i * 8;  // stride = 8
          const int offset = q_packed_idx % group_size;
          const int head_idx = kv_head_idx * group_size + offset;
          alibi_slopes_(i, mi) = alibi_slops_ptr[head_idx] / sm_scale;
        }
      }
    }
  }

  // rAccS: ((2, MMA_M), (2, MMA_N))
  template <bool OOB_MASK = true, typename FragmentS>
  CUTE_HOST_DEVICE void apply(FragmentS& rAccS, int n_block) const {
    // Warp layout 4x1, each warp processes 16 rows (4 threads per row)
    const int n_base_idx = n_block * BLK_N + (lane_idx_ % 4) * 2;

    // TiledMMA: 64x16x16, MMA_Atom: 16x8x16
    CUTE_UNROLL
    for (int mi = 0; mi < size<0, 1>(rAccS); ++mi) {        //  MMA_M
      const int q_packed_idx_base = m_base_idx_ + mi * 64;  // stride = 64
      CUTE_UNROLL
      for (int i = 0; i < size<0, 0>(rAccS); ++i) {          // 2
        const int q_packed_idx = q_packed_idx_base + i * 8;  // m stride = 8
        const int q_idx = q_packed_idx / group_size_ + diagonal_offset_;

        const auto m_coord = make_coord(i, mi);
        const auto alibi_slope = ALIBI ? alibi_slopes_(i, mi) : 0.0f;

        CUTE_UNROLL
        for (int nj = 0; nj < size<1, 1>(rAccS); ++nj) {  // MMA_N
          const auto kv_base_idx = n_base_idx + nj * 8;   // n outer stride = 8

          CUTE_UNROLL
          for (int j = 0; j < size<1, 0>(rAccS); ++j) {  // 2
            const int kv_idx = kv_base_idx + j;          // n stride = 1

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

            const auto n_coord = make_coord(j, nj);
            if (out_of_boundary) {
              rAccS(m_coord, n_coord) = -INFINITY;
            } else if constexpr (ALIBI) {
              // Apply alibi bias to the attention scores
              rAccS(m_coord, n_coord) += alibi_slope * kv_idx;
            }
          }
        }
      }
    }
  }
};

}  // namespace llm