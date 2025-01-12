#pragma once
#include <cute/config.hpp>
#include <cute/tensor.hpp>

#include "online_softmax.cuh"

namespace llm {
using namespace cute;

namespace detail {

// Convert fragment layout for different purposes
// Only works for TiledMMA (64x16x16) with SM80_16x8x16_F32F16F16F32_TN
struct LayoutConvertor {
  // Convert fragment layout to rowcol layout for iterating
  // (MMA=4, MMA_M, MMA_N) => ((2, MMA_M), (2, MMA_N))
  template <typename LayoutC>
  CUTE_HOST_DEVICE static constexpr auto to_rowcol(const LayoutC& layout) {
    auto l = logical_divide(layout, Shape<_2>{});
    return make_layout(make_layout(get<0, 1>(l), get<1>(l)),
                       make_layout(get<0, 0>(l), get<2>(l)));
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

template <int kBlockM, int kBlockN, bool Alibi>
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
  template <typename FragmentS>
  CUTE_HOST_DEVICE void apply(FragmentS& rAccS,
                              int m_block,
                              int n_block,
                              int tidx) const {
    // TODO: support other warp layout
    // Warp layout 4x1, each warp processes 16 rows (4 threads per row)
    const int warp_idx_x = tidx / 32;
    const int warp_idx_y = 0;
    const int lane_id = tidx % 32;
    const int m_base =
        m_block * kBlockM + warp_idx_x * 16 + lane_id / 4 + kv_len_ - q_len_;
    const int n_base = n_block * kBlockN + warp_idx_y * 16 + (lane_id % 4) * 2;

    // TiledMMA: 64x16x16, MMA_Atom: 16x8x16
    CUTE_UNROLL
    for (int mi = 0; mi < size<0, 1>(rAccS); ++mi) {  //  MMA_M
      const int q_idx_base = m_base + mi * 64;
      CUTE_UNROLL
      for (int i = 0; i < size<0, 0>(rAccS); ++i) {  // 2
        // m inner stride = 8
        const int q_idx = q_idx_base + i * 8;  // diagonal

        CUTE_UNROLL
        for (int nj = 0; nj < size<1, 1>(rAccS); ++nj) {  // MMA_N
          // n outer stride = 8
          const auto kv_index_base = n_base + nj * 8;

          CUTE_UNROLL
          for (int j = 0; j < size<1, 0>(rAccS); ++j) {  // 2
            // n inner stride = 1
            const int kv_idx = kv_index_base + j;

            const bool out_of_boundary =
                kv_idx > q_idx || kv_idx >= kv_len_     // causal + oob mask
                || (q_idx - kv_idx) > sliding_window_;  // sliding window mask
            if (out_of_boundary) {
              rAccS(make_coord(i, mi), make_coord(j, nj)) = -INFINITY;
            } else {
              // Apply alibi bias
              if constexpr (Alibi) {
                rAccS(make_coord(i, mi), make_coord(j, nj)) +=
                    alibi_slope_ * kv_idx;
              }
            }
          }
        }
      }
    }
  }
};

}  // namespace detail

template <typename Element_ = cute::half_t,
          int kHeadDim_ = 64,
          int kBlockM_ = 64,
          int kBlockN_ = 64,
          bool Alibi_ = false>
struct AttentionTraitsSM80 {
  // helpful aliases
  static constexpr int kHeadDim = kHeadDim_;
  static constexpr int kBlockM = kBlockM_;
  static constexpr int kBlockN = kBlockN_;
  static constexpr int kBlockK = kHeadDim % 64 == 0 ? 64 : 32;

  using Element = Element_;
  using _BLK_M = Int<kBlockM>;
  using _BLK_N = Int<kBlockN>;
  using _BLK_K = Int<kBlockK>;
  using _HEAD_DIM = Int<kHeadDim>;

  // ******* Mainloop *******
  // TiledMMA (64x16x16) for gemm-I and gemm-II
  // choose MMA_Atom based on Element type
  using MMA_Atom_ =
      std::conditional_t<std::is_same_v<Element, cute::half_t>,
                         MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
                         MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>>;
  using TiledMma = TiledMMA<MMA_Atom_,
                            Layout<Shape<_4, _1, _1>>,  // warp layout 4x1x1
                            Tile<_64, _16, _16>>;       // Prom Shape 64x16x16

  // Layout convertor for TiledMMA (64x16x16)
  using LayoutConvertor = detail::LayoutConvertor;

  // Mask for causal, local, alibi
  using Mask = detail::Mask<kBlockM_, kBlockN_, Alibi_>;

  // Online softmax
  using Softmax = OnlineSoftmax<2 * kBlockM_ / 64>;

  // SMEM layout for QKV
  // Q smem: (BLK_M, K):(K, 1), k-major
  using SmemLayoutQ = decltype(composition(
      Swizzle<3, 3, 3>{},
      Layout<Shape<_BLK_M, _HEAD_DIM>, Stride<_HEAD_DIM, _1>>{}));

  // KV smem: (BLK_N, K):(K, 1), k-major
  using SmemLayoutK = decltype(composition(
      Swizzle<3, 3, 3>{},
      Layout<Shape<_BLK_N, _HEAD_DIM>, Stride<_HEAD_DIM, _1>>{}));

  using SmemLayoutV = decltype(composition(
      Swizzle<3, 3, 3>{},
      Layout<Shape<_BLK_N, _HEAD_DIM>, Stride<_HEAD_DIM, _1>>{}));

  // V^T smem: (K, BLK_N):(1, K), k-major
  using SmemLayoutVt = decltype(composition(
      Swizzle<3, 3, 3>{},
      Layout<Shape<_HEAD_DIM, _BLK_N>, Stride<_1, _HEAD_DIM>>{}));

  // Tiled copy for QKV
  // g2s tiled copy for qkv
  using GmemTiledCopyQKV = decltype(make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, Element>{},
      Layout<Shape<_16, _8>, Stride<_8, _1>>{},  // Thr layout: (_16,_8):(_8,_1)
      Layout<Shape<_1, _8>>{}                    // Val layout: 8 vals per read
      ));

  // s2r tiled copy for gemm-I
  using SmemTiledCopyQ =
      decltype(make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, Element>{},
                                 TiledMma{}));
  using SmemTiledCopyK =
      decltype(make_tiled_copy_B(Copy_Atom<SM75_U32x4_LDSM_N, Element>{},
                                 TiledMma{}));

  // s2r tiled copy for gemm-II
  using SmemTiledCopyVt =
      decltype(make_tiled_copy_B(Copy_Atom<SM75_U16x8_LDSM_T, Element>{},
                                 TiledMma{}));

  // ******* Epilogue *******

  // O smem: (BLK_M, K):(K, 1), k-major, same as Q
  using SmemLayoutO = SmemLayoutQ;

  // s2g tiled copy for O
  using GmemTiledCopyO = decltype(make_tiled_copy(
      Copy_Atom<UniversalCopy<cute::uint128_t>, Element>{},
      Layout<Shape<_16, _8>, Stride<_8, _1>>{},  // Thr layout: (_16,_8):(_8,_1)
      Layout<Shape<_1, _8>>{}                    // Val layout: 8 vals per read
      ));

  // r2s tiled copy for O
  using SmemTiledCopyO =
      decltype(make_tiled_copy_C(Copy_Atom<DefaultCopy, Element>{},
                                 TiledMma{}));

  // constexpr values for kernel launch
  static constexpr size_t kSmemSize =
      (cosize(SmemLayoutQ{}) + cosize(SmemLayoutK{}) + cosize(SmemLayoutV{})) *
      sizeof(Element);

  static constexpr size_t kThreadNum = size(TiledMma{});
};

}  // namespace llm