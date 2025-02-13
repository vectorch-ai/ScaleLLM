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

  static_assert(kHeadDim % 32 == 0);
  static_assert(kRopeHeadDim % 32 == 0);

  static_assert(kBlockM % 64 == 0);
  static_assert(kBlockN % 32 == 0);
  static_assert(kBlockK % 32 == 0);
  
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

  // TiledMma for P = Softmax(Q*K^T), 8 warps
  using TiledMma_QK = TiledMMA<MMA_Atom_,
                               Layout<Shape<_4, _2, _1>>,  // warp layout 4x2x1
                               Tile<_64, _32, _16>>;  // Prom Shape 64x32x16

  // TiledMma for O = P*V^T, 8 warps
  using TiledMma_PV = TiledMMA<MMA_Atom_,
                               Layout<Shape<_4, _2, _1>>,  // warp layout 4x2x1
                               Tile<_64, _32, _16>>;  // Prom Shape 64x32x16

  // Layout convertor for TiledMMA (64x16x16)
  using LayoutConvertor = detail::LayoutConvertor;

  // Smem LayoutAtom
  using SmemLayoutAtom_8x64 =
      decltype(composition(Swizzle<3, 3, 3>{},
                           Layout<Shape<_8, _64>, Stride<_64, _1>>{}));
  using SmemLayoutAtom_8x32 =
      decltype(composition(Swizzle<2, 3, 3>{},
                           Layout<Shape<_8, _32>, Stride<_32, _1>>{}));

  using SmemLayoutAtomK = std::conditional_t<kBlockK % 64 == 0,
                                             SmemLayoutAtom_8x64,
                                             SmemLayoutAtom_8x32>;
  using SmemLayoutAtomN = std::conditional_t<kBlockN % 64 == 0,
                                             SmemLayoutAtom_8x64,
                                             SmemLayoutAtom_8x32>;
  using SmemLayoutAtomR = std::conditional_t<kRopeHeadDim % 64 == 0,
                                             SmemLayoutAtom_8x64,
                                             SmemLayoutAtom_8x32>;

  // SMEM layout for QKV
  // Q smem: (BLK_M, BLK_K, STAGES)
  using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomK{},
                                             Shape<_BLK_M, _BLK_K, _STAGES>{}));

  // KV smem: (BLK_N, BLK_K, STAGES)
  using SmemLayoutKV =
      decltype(tile_to_shape(SmemLayoutAtomK{},
                             Shape<_BLK_N, _BLK_K, _STAGES>{}));

  // P smem: (BLK_M, BLK_N)
  using SmemLayoutP =
      decltype(tile_to_shape(SmemLayoutAtomN{}, Shape<_BLK_M, _BLK_N>{}));

  // V^T smem: (BLK_K, BLK_N, STAGES)
  using SmemLayoutVt = decltype(permute<1, 0, 2>(SmemLayoutKV{}));

  // QRope smem: (BLK_M, ROPE_HEAD_DIM)
  using SmemLayoutQRope =
      decltype(tile_to_shape(SmemLayoutAtomR{},
                             Shape<_BLK_M, _ROPE_HEAD_DIM>{}));

  // KRoep smem: (BLK_N, ROPE_HEAD_DIM)
  using SmemLayoutKRope =
      decltype(tile_to_shape(SmemLayoutAtomR{},
                             Shape<_BLK_N, _ROPE_HEAD_DIM>{}));

  // Thr layout for gmem copy, 32x8
  using GmemCopyThrLayout = Layout<Shape<_32, _8>, Stride<_8, _1>>;

  // Tiled copy for QKV
  // use 128-bit vectorizing copy
  using VectorizingCopy = AutoVectorizingCopyWithAssumedAlignment<128>;

  // g2s tiled copy for q
  using GmemTiledCopyQ = decltype(make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, DType>{},
      GmemCopyThrLayout{},     // Thr layout: (_32, _8)
      Layout<Shape<_1, _8>>{}  // Val layout: 8 vals per read
      ));

  // g2s tiled copy for kv
  using GmemTiledCopyKV = GmemTiledCopyQ;

  // s2r tiled copy for gemm-I S = Q*K^T
  using SmemTiledCopyQ =
      decltype(make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, DType>{},
                                 TiledMma_QK{}));
  using SmemTiledCopyK =
      decltype(make_tiled_copy_B(Copy_Atom<SM75_U32x4_LDSM_N, DType>{},
                                 TiledMma_QK{}));

  // r2s tiled copy for gemm-I S
  using SmemTiledCopyS =
      decltype(make_tiled_copy_C(Copy_Atom<VectorizingCopy, DType>{},
                                 TiledMma_QK{}));

  // s2r tiled copy for gemm-II: O = P*V^T
  using SmemTiledCopyP =
      decltype(make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, DType>{},
                                 TiledMma_PV{}));
  using SmemTiledCopyVt =
      decltype(make_tiled_copy_B(Copy_Atom<SM75_U16x8_LDSM_T, DType>{},
                                 TiledMma_PV{}));

  // r2s tiled copy for gemm-II O
  using SmemTiledCopyO =
      decltype(make_tiled_copy_C(Copy_Atom<VectorizingCopy, DType>{},
                                 TiledMma_PV{}));

  // ******* Epilogue *******

  // O smem: (BLK_M, BLK_K, STAGES)
  using SmemLayoutO = decltype(tile_to_shape(SmemLayoutAtomK{},
                                             Shape<_BLK_M, _BLK_K, _STAGES>{}));

  // s2g tiled copy for O
  using GmemTiledCopyO = decltype(make_tiled_copy(
      Copy_Atom<VectorizingCopy, DType>{},
      GmemCopyThrLayout{},     // Thr layout: (_32, _8)
      Layout<Shape<_1, _8>>{}  // Val layout: 8 vals per read
      ));

  // constexpr values for kernel launch
  static constexpr size_t kSmemSize =
      sizeof(DType) *
      (cosize(SmemLayoutQ{}) + cosize(SmemLayoutKV{}) + cosize(SmemLayoutP{}) +
       cosize(SmemLayoutQRope{}) + cosize(SmemLayoutKRope{}));

  static constexpr size_t kThreadNum = size(TiledMma_PV{});
};

}  // namespace llm