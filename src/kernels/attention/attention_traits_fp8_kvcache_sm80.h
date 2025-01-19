#pragma once
#include <cute/config.hpp>
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
  // (MMA_C=4,MMA_M,MMA_N) => (MMA_A=(4, 2), MMA_M, MMA_N/2)
  template <typename LayoutC>
  CUTE_HOST_DEVICE static constexpr auto to_mma_a(const LayoutC& layout) {
    auto l = logical_divide(layout.layout(), Shape<X, X, _2>{});
    return make_layout(
        make_layout(get<0>(l), get<2, 0>(l)), get<1>(l), get<2, 1>(l));
  }
};

}  // namespace detail

template <typename DTYPE_, int HEAD_DIM, int BLK_M, int BLK_N, int BLK_K>
struct AttentionTraitsFp8KVCacheSM80 {
  // helpful aliases
  static constexpr int kHeadDim = HEAD_DIM;
  static constexpr int kBlockM = BLK_M;
  static constexpr int kBlockN = BLK_N;
  static constexpr int kBlockK = BLK_K;
  static constexpr int kRowsPerMMA = 2;

  using DType = DTYPE_;
  using _BLK_M = Int<kBlockM>;
  using _BLK_N = Int<kBlockN>;
  using _BLK_K = Int<kBlockK>;
  using _HEAD_DIM = Int<kHeadDim>;

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
                           Layout<Shape<_8, _BLK_K>, Stride<_BLK_K, _1>>{}));

  // Q smem: (BLK_M, HEAD_DIM)
  using SmemLayoutQ =
      decltype(tile_to_shape(SmemLayoutAtom{}, Shape<_BLK_M, _HEAD_DIM>{}));

  // KV smem: (BLK_N, HEAD_DIM)
  using SmemLayoutK =
      decltype(tile_to_shape(SmemLayoutAtom{}, Shape<_BLK_N, _HEAD_DIM>{}));

  using SmemLayoutV =
      decltype(tile_to_shape(SmemLayoutAtom{}, Shape<_BLK_N, _HEAD_DIM>{}));

  // V^T smem: (HEAD_DIM, BLK_N) row-major
  using SmemLayoutVt = decltype(composition(
      SmemLayoutV{},
      make_layout(Shape<_HEAD_DIM, _BLK_N>{}, GenRowMajor{})));

  // Thr layout for gmem copy
  using GmemCopyThrLayout =
      std::conditional_t<BLK_K == 32,
                         Layout<Shape<_32, _4>, Stride<_4, _1>>,
                         Layout<Shape<_16, _8>, Stride<_8, _1>>>;

  // Tiled copy for QKV
  // g2s tiled copy for qkv
  using GmemTiledCopyQKV = decltype(make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, DType>{},
      GmemCopyThrLayout{},     // Thr layout: (_16,_8)/(_32, _4)
      Layout<Shape<_1, _8>>{}  // Val layout: 8 vals per read
      ));

  // s2r tiled copy for gemm-I
  using SmemTiledCopyQ =
      decltype(make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, DType>{},
                                 TiledMma{}));

  // s2r tiled copy for k (fp8/int8)
  // ((_4, _8), (_2, _2)):((_32, _1), (_16, _8))
  using Layout_TV_K = Layout<Shape<Shape<_4, _8>, Shape<_2, _2>>,
                             Stride<Stride<_32, _1>, Stride<_16, _8>>>;
  using SmemTiledCopyK = TiledCopy<Copy_Atom<SM75_U32x2_LDSM_N, DType>,
                                   Layout_TV_K,
                                   Shape<_16, _8>>;  // N x K => 16 x 16

  // s2r tiled copy for vt
  // ((_4, _8), (_2, _2)):((_16, _1), (_8, _64))
  using Layout_TV_Vt = Layout<Shape<Shape<_4, _8>, Shape<_2, _2>>,
                              Stride<Stride<_16, _1>, Stride<_8, _64>>>;
  using SmemTiledCopyVt = TiledCopy<Copy_Atom<SM75_U16x4_LDSM_T, cute::half_t>,
                                    Layout_TV_Vt,
                                    Shape<_8, _16>>;  // K x N => 16 x 16

  // ******* Epilogue *******

  // O smem: (BLK_M, K):(K, 1), k-major, same as Q
  using SmemLayoutO = SmemLayoutQ;

  // s2g tiled copy for O
  using GmemTiledCopyO = decltype(make_tiled_copy(
      Copy_Atom<DefaultCopy, DType>{},
      GmemCopyThrLayout{},     // Thr layout: (_16,_8)/(_32, _4)
      Layout<Shape<_1, _8>>{}  // Val layout: 8 vals per read
      ));

  // r2s tiled copy for O
  using SmemTiledCopyO =
      decltype(make_tiled_copy_C(Copy_Atom<DefaultCopy, DType>{}, TiledMma{}));

  // constexpr values for kernel launch
  static constexpr size_t kSmemSize =
      (cosize(SmemLayoutQ{}) + cosize(SmemLayoutK{}) + cosize(SmemLayoutV{})) *
      sizeof(DType);

  static constexpr size_t kThreadNum = size(TiledMma{});
};

}  // namespace llm