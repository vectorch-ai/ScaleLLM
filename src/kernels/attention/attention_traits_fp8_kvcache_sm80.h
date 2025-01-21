#pragma once
#include <cute/config.hpp>
#include <cute/tensor.hpp>

namespace llm {
using namespace cute;


template <typename DTYPE_,
          typename KV_DTYPE,
          int HEAD_DIM,
          int BLK_M,
          int BLK_N,
          int BLK_K>
struct AttentionTraitsFp8KVCacheSM80 {
  // helpful aliases
  static constexpr int kHeadDim = HEAD_DIM;
  static constexpr int kBlockM = BLK_M;
  static constexpr int kBlockN = BLK_N;
  static constexpr int kBlockK = BLK_K;
  static constexpr int kRowsPerMMA = 2;

  using DType = DTYPE_;
  using KV_DType = KV_DTYPE;
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
  //   using LayoutConvertor = detail::LayoutConvertor;

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
  // TODO: choose based on BLK_K and kv cache type
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
  using GmemCopyThrLayoutKV =
      std::conditional_t<BLK_K == 32,
                         Layout<Shape<_64, _2>, Stride<_2, _1>>,
                         Layout<Shape<_32, _4>, Stride<_4, _1>>>;
  using GmemTiledCopyKV = decltype(make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, KV_DTYPE>{},
      GmemCopyThrLayoutKV{},    // Thr layout: (_16,_4)/(_32, _4)
      Layout<Shape<_1, _16>>{}  // Val layout: 16 vals = 128 bits per read
      ));

  // s2r tiled copy for gemm-I
  using SmemTiledCopyQ =
      decltype(make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, DType>{},
                                 TiledMma{}));

  // s2r tiled copy for k (fp8/int8)
  // ((_4, _8), (_4, _2)):((_64, _1), (_16, _8))
  using Layout_TV_K = Layout<Shape<Shape<_4, _8>, Shape<_4, _2>>,
                             Stride<Stride<_64, _1>, Stride<_16, _8>>>;
  using SmemTiledCopyK = TiledCopy<Copy_Atom<SM75_U32x2_LDSM_N, KV_DType>,
                                   Layout_TV_K,
                                   Shape<_16, _16>>;  // N x K

  // s2r tiled copy for vt (fp8/int8)
  // ((_4, _8), (_2, _2, _2)):((_32, _2), (_1, _16, _128))
  using Layout_TV_Vt = Layout<Shape<Shape<_4, _8>, Shape<_2, _2, _2>>,
                              Stride<Stride<_32, _2>, Stride<_1, _16, _128>>>;
  using SmemTiledCopyVt = TiledCopy<Copy_Atom<SM75_U16x4_LDSM_T, KV_DType>,
                                    Layout_TV_Vt,
                                    Shape<_16, _16>>;  // K x N

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
      cosize(SmemLayoutQ{}) * sizeof(DType) +
      (cosize(SmemLayoutK{}) + cosize(SmemLayoutV{})) * sizeof(KV_DType);

  static constexpr size_t kThreadNum = size(TiledMma{});
};

}  // namespace llm