#pragma once
#include <cute/tensor.hpp>

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
  template <typename LayoutC>
  CUTE_HOST_DEVICE static constexpr auto to_mma_a(const LayoutC& layout) {
    // (MMA_C=4,MMA_M,MMA_N) => (MMA_A=(4, 2), MMA_M, MMA_N/2)
    auto l = logical_divide(layout.layout(), Shape<X, X, _2>{});
    return make_layout(
        make_layout(get<0>(l), get<2, 0>(l)), get<1>(l), get<2, 1>(l));
  }
};

}  // namespace detail

template <typename Element_ = cutlass::half_t,
          int kHeadDim_ = 64,
          int kBlockM_ = 64,
          int kBlockN_ = 64>
struct AttentionTraitsSM80 {
  // helpful aliases
  using Element = Element_;
  using BLK_M = Int<kBlockM_>;
  using BLK_N = Int<kBlockN_>;
  using BLK_K = Int<kHeadDim_ % 64 == 0 ? 64 : 32>;
  using HEAD_DIM = Int<kHeadDim_>;

  // ******* Mainloop *******
  // TiledMMA (64x16x16) for gemm-I and gemm-II
  using TiledMMA = TiledMMA<MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
                            Layout<Shape<_4, _1, _1>>,  // warp layout 4x1x1
                            Tile<_64, _16, _16>>;       // Prom Shape 64x16x16

  // Layout convertor for TiledMMA (64x16x16)
  using LayoutConvertor = detail::LayoutConvertor;

  // SMEM layout for QKV
  // Q smem: (BLK_M, K):(K, 1), k-major
  using SmemLayoutQ = Layout<Shape<BLK_M, HEAD_DIM>, Stride<HEAD_DIM, _1>>;

  // KV smem: (BLK_N, K):(K, 1), k-major
  using SmemLayoutKV = Layout<Shape<BLK_N, HEAD_DIM>, Stride<HEAD_DIM, _1>>;

  // V^T smem: (K, BLK_N):(1, K), k-major
  using SmemLayoutVt = Layout<Shape<HEAD_DIM, BLK_N>, Stride<_1, HEAD_DIM>>;

  // Tiled copy for QKV
  // s2r CopyAtom for QK: 16x16
  using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
  // s2r CopyAtom for V^T
  using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, Element>;

  // g2s tiled copy for qkv
  using GmemTiledCopyQKV = decltype(make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, Element>{},
      Layout<Shape<_16, _8>, Stride<_8, _1>>{},  // Thr layout: (_16,_8):(_8,_1)
      Layout<Shape<_1, _8>>{}                    // Val layout: 8 vals per read
      ));

  // s2r tiled copy for gemm-I
  using SmemTiledCopyQ =
      decltype(make_tiled_copy_A(SmemCopyAtom{}, TiledMMA{}));
  using SmemTiledCopyK =
      decltype(make_tiled_copy_B(SmemCopyAtom{}, TiledMMA{}));

  // s2r tiled copy for gemm-II
  using SmemTiledCopyVT =
      decltype(make_tiled_copy_B(SmemCopyAtomTransposed{}, TiledMMA{}));

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
  using SmemCopyAtomO = Copy_Atom<DefaultCopy, Element>;
  using SmemTiledCopyO =
      decltype(make_tiled_copy_C(SmemCopyAtomO{}, TiledMMA{}));

  // constexpr values for kernel launch
  static constexpr int kSmemSize =
      (cosize(SmemLayoutQ{}) + cosize(SmemLayoutKV{}) * 2) * sizeof(Element);

  static constexpr int kThreadNum = size(TiledMMA{});
};

}  // namespace llm