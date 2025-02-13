#pragma once
#include <cute/config.hpp>
#include <cute/tensor.hpp>

#include "cute_extensions.cuh"

namespace llm {
using namespace cute;

namespace detail {

// Convert fragment layout for different purposes
// Only works for TiledMMA (64x16x16) with SM80_16x8x16_F32F16F16F32_TN
struct LayoutConvertor {
  // Convert fragment layout to rowcol layout for iterating
  // (MMA=4, MMA_M, MMA_N, STAGES) => ((2, MMA_M), (2, MMA_N), STAGES)
  template <typename LayoutC>
  CUTE_HOST_DEVICE static constexpr auto to_mn(const LayoutC& layout) {
    auto l = logical_divide(layout, Shape<_2>{});
    return make_layout(make_layout(get<0, 1>(l), get<1>(l)),
                       make_layout(get<0, 0>(l), get<2>(l)));
  }

  template <typename LayoutC>
  CUTE_HOST_DEVICE static constexpr auto to_mns(const LayoutC& layout) {
    auto l = logical_divide(layout, Shape<_2>{});
    return make_layout(make_layout(get<0, 1>(l), get<1>(l)),
                       make_layout(get<0, 0>(l), get<2>(l)),
                       get<3>(l));
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

}  // namespace detail

template <typename DTYPE,
          int HEAD_DIM,
          int ROPE_HEAD_DIM,
          int BLK_M,
          int BLK_N,
          int BLK_K>
struct MLATraitsSM80 {
  // helpful aliases
  static constexpr int kHeadDim = HEAD_DIM;
  static constexpr int kRopeHeadDim = ROPE_HEAD_DIM;
  static constexpr int kBlockM = BLK_M;
  static constexpr int kBlockN = BLK_N;
  static constexpr int kBlockK = BLK_K;
  static constexpr int kRowsPerMMA = 2;

  static_assert(kHeadDim % kBlockK == 0);
  static constexpr int kStages = kHeadDim / kBlockK;

  using DType = DTYPE;
  using _BLK_M = Int<kBlockM>;
  using _BLK_N = Int<kBlockN>;
  using _BLK_K = Int<kBlockK>;
  using _STAGES = Int<kStages>;
  using _HEAD_DIM = Int<kHeadDim>;
  using _ROPE_HEAD_DIM = Int<kRopeHeadDim>;

  // ******* Mainloop *******
  // TiledMMA (64x16x16) for gemm-I and gemm-II
  // choose MMA_Atom based on Element type
  using MMA_Atom_ =
      std::conditional_t<std::is_same_v<DType, cute::half_t>,
                         MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
                         MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>>;
  using TiledMma = TiledMMA<MMA_Atom_,
                            Layout<Shape<_4, _1, _1>>,  // warp layout 4x1x1
                            Tile<_64, _16, _16>>;       // Prom Shape 64x16x16

  // Layout convertor for TiledMMA (64x16x16)
  using LayoutConvertor = detail::LayoutConvertor;

  // SMEM layout for QKV
  // Atom layout: (8, BLK_K):(BLK_K, 1) k-major
  using SmemLayoutAtom =
      decltype(composition(Swizzle<3, 3, 3>{},
                           Layout<Shape<_8, _64>, Stride<_64, _1>>{}));

  // Q smem: (BLK_M, BLK_K, STAGES)
  using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtom{},
                                             Shape<_BLK_M, _BLK_K, _STAGES>{}));

  // KV smem: (BLK_N, BLK_K, STAGES)
  using SmemLayoutKV =
      decltype(tile_to_shape(SmemLayoutAtom{},
                             Shape<_BLK_N, _BLK_K, _STAGES>{}));

  // V^T smem: (BLK_K, BLK_N, STAGES)
  using SmemLayoutVt = decltype(permute<1, 0, 2>(SmemLayoutKV{}));

  // QRope smem: (BLK_M, ROPE_HEAD_DIM)
  using SmemLayoutQRope =
      decltype(tile_to_shape(SmemLayoutAtom{},
                             Shape<_BLK_M, _ROPE_HEAD_DIM>{}));

  // KRoep smem: (BLK_N, ROPE_HEAD_DIM)
  using SmemLayoutKRope =
      decltype(tile_to_shape(SmemLayoutAtom{},
                             Shape<_BLK_N, _ROPE_HEAD_DIM>{}));

  // Thr layout for gmem copy
  using GmemCopyThrLayout =
      std::conditional_t<BLK_K == 32,
                         Layout<Shape<_32, _4>, Stride<_4, _1>>,
                         Layout<Shape<_16, _8>, Stride<_8, _1>>>;

  // Tiled copy for QKV
  // g2s tiled copy for q
  using GmemTiledCopyQ = decltype(make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, DType>{},
      GmemCopyThrLayout{},     // Thr layout: (_16,_8)/(_32, _4)
      Layout<Shape<_1, _8>>{}  // Val layout: 8 vals per read
      ));

  // g2s tiled copy for kv
  using GmemTiledCopyKV = GmemTiledCopyQ;

  // s2r tiled copy for gemm-I
  using SmemTiledCopyQ =
      decltype(make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, DType>{},
                                 TiledMma{}));
  using SmemTiledCopyK =
      decltype(make_tiled_copy_B(Copy_Atom<SM75_U32x4_LDSM_N, DType>{},
                                 TiledMma{}));

  // s2r tiled copy for gemm-II
  using SmemTiledCopyVt =
      decltype(make_tiled_copy_B(Copy_Atom<SM75_U16x8_LDSM_T, DType>{},
                                 TiledMma{}));

  // ******* Epilogue *******

  // O smem: (BLK_M, BLK_K, STAGES) k-major
  using SmemLayoutO = decltype(tile_to_shape(SmemLayoutAtom{},
                                             Shape<_BLK_M, _BLK_K, _STAGES>{}));

  // use 128-bit vectorizing copy
  using VectorizingCopy = AutoVectorizingCopyWithAssumedAlignment<128>;

  // s2g tiled copy for O
  using GmemTiledCopyO = decltype(make_tiled_copy(
      Copy_Atom<VectorizingCopy, DType>{},
      GmemCopyThrLayout{},     // Thr layout: (_16,_8)/(_32, _4)
      Layout<Shape<_1, _8>>{}  // Val layout: 8 vals per read
      ));

  // r2s tiled copy for O
  using SmemTiledCopyO =
      decltype(make_tiled_copy_C(Copy_Atom<VectorizingCopy, DType>{},
                                 TiledMma{}));

  // constexpr values for kernel launch
  static constexpr size_t kSmemSize =
      sizeof(DType) * (cosize(SmemLayoutQ{}) + cosize(SmemLayoutKV{}) +
                       cosize(SmemLayoutQRope{}) + cosize(SmemLayoutKRope{}));

  static constexpr size_t kThreadNum = size(TiledMma{});
};

}  // namespace llm