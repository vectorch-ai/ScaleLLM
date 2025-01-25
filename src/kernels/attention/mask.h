#pragma once
#include <cute/config.hpp>
#include <cute/tensor.hpp>

namespace llm {
using namespace cute;

template <int BLK_M, int BLK_N, bool ALIBI, bool LOCAL>
struct Mask {
  int q_len_;
  int kv_len_;
  int sliding_window_;
  float alibi_slope_;

  CUTE_HOST_DEVICE Mask(int q_len,
                        int kv_len,
                        int sliding_window,
                        float alibi_slope)
      : q_len_(q_len),
        kv_len_(kv_len),
        sliding_window_(sliding_window),
        alibi_slope_(alibi_slope) {}

  // rAccS: ((2, MMA_M), (2, MMA_N))
  template <bool OOB_MASK = true, typename FragmentS>
  CUTE_HOST_DEVICE void apply(FragmentS& rAccS,
                              int m_block,
                              int n_block,
                              int tidx) const {
    // TODO: support other warp layout
    // Warp layout 4x1, each warp processes 16 rows (4 threads per row)
    const int warp_idx_x = tidx / 32;
    const int warp_idx_y = 0;
    const int lane_id = tidx % 32;
    const int m_base = m_block * BLK_M + warp_idx_x * 16 + lane_id / 4;
    const int n_base = n_block * BLK_N + warp_idx_y * 16 + (lane_id % 4) * 2;

    // TiledMMA: 64x16x16, MMA_Atom: 16x8x16
    CUTE_UNROLL
    for (int mi = 0; mi < size<0, 1>(rAccS); ++mi) {  //  MMA_M
      const int q_idx_base = m_base + mi * 64;
      CUTE_UNROLL
      for (int i = 0; i < size<0, 0>(rAccS); ++i) {  // 2
        // m inner stride = 8
        const int q_idx = q_idx_base + i * 8 + kv_len_ - q_len_;  // diagonal

        CUTE_UNROLL
        for (int nj = 0; nj < size<1, 1>(rAccS); ++nj) {  // MMA_N
          // n outer stride = 8
          const auto kv_index_base = n_base + nj * 8;

          CUTE_UNROLL
          for (int j = 0; j < size<1, 0>(rAccS); ++j) {  // 2
            // n inner stride = 1
            const int kv_idx = kv_index_base + j;

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
              rAccS(make_coord(i, mi), make_coord(j, nj)) = -INFINITY;
            } else if constexpr (ALIBI) {
              // Apply alibi bias
              rAccS(make_coord(i, mi), make_coord(j, nj)) +=
                  alibi_slope_ * kv_idx;
            }
          }
        }
      }
    }
  }
};

}  // namespace llm