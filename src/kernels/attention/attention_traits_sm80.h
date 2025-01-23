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

template <typename DType, typename KV_DType>
CUTE_HOST_DEVICE constexpr auto tiled_mma_selector() {
  CUTE_STATIC_ASSERT(
      sizeof_bits_v<KV_DType> == 16 || sizeof_bits_v<KV_DType> == 8,
      "KV_DType must be 8 or 16 bits");

  using MMA_Atom = std::conditional_t<std::is_same_v<DType, cute::half_t>,
                                      MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
                                      MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>>;
  using TiledMma = TiledMMA<MMA_Atom,
                            Layout<Shape<_4, _1, _1>>,  // warp layout 4x1x1
                            Tile<_64, _16, _16>>;       // Prom Shape 64x16x16
  return TiledMma{};
}

template <typename COPY_Atom, int BLK_K, int THREADS>
CUTE_HOST_DEVICE constexpr auto tiled_copy_selector() {
  using DType = typename COPY_Atom::ValType;
  // use 128 bits vectorized copy
  constexpr int kValPerThr = 128 / sizeof_bits_v<DType>;
  constexpr int kThrsPerRow = BLK_K / kValPerThr;
  using ThrLayout = Layout<Shape<Int<THREADS / kThrsPerRow>, Int<kThrsPerRow>>,
                           Stride<Int<kThrsPerRow>, _1>>;
  using ValLayout = Layout<Shape<_1, Int<kValPerThr>>>;
  return make_tiled_copy(COPY_Atom{}, ThrLayout{}, ValLayout{});
}

template <typename KV_DType, typename TiledMma>
CUTE_HOST_DEVICE constexpr auto tiled_copy_B_selector() {
  if constexpr (sizeof_bits_v<KV_DType> == 16) {
    // ((_4,_8,_4),((_2,_2),(_2,_1))):((_32,_1,_0),((_16,_128),(_8,_0)))
    return make_tiled_copy_B(Copy_Atom<SM75_U32x4_LDSM_N, KV_DType>{},
                             TiledMma{});
  } else if constexpr (sizeof_bits_v<KV_DType> == 8) {
    // ((_4, _8), (_4, _2)):((_64, _1), (_16, _8))
    using Layout_TV_K = Layout<Shape<Shape<_4, _8>, Shape<_4, _2>>,
                               Stride<Stride<_64, _1>, Stride<_16, _8>>>;
    // use cute::uint8_t as InternalType
    using SmemTiledCopyK =
        TiledCopy<Copy_Atom<SM75_U32x2_LDSM_N, cute::uint8_t>,
                  Layout_TV_K,
                  Shape<_16, _16>>;  // N x K
    return SmemTiledCopyK{};
  } else {
    CUTE_STATIC_ASSERT(
        sizeof_bits_v<KV_DType> == 8 || sizeof_bits_v<KV_DType> == 16,
        "KV_DType must be 8 or 16 bits");
  }
}

template <typename DType, typename TiledMma>
CUTE_HOST_DEVICE constexpr auto tiled_copy_B_T_selector() {
  if constexpr (sizeof_bits_v<DType> == 16) {
    // ((_4,_8,_4),((_2,_2),(_2,_1))):((_32,_1,_0),((_16,_128),(_8,_0)))
    return make_tiled_copy_B(Copy_Atom<SM75_U16x8_LDSM_T, DType>{}, TiledMma{});
  } else if constexpr (sizeof_bits_v<DType> == 8) {
    // ((_4, _8), (_2, _2, _2)):((_32, _2), (_1, _16, _128))
    using Layout_TV_Vt = Layout<Shape<Shape<_4, _8>, Shape<_2, _2, _2>>,
                                Stride<Stride<_32, _2>, Stride<_1, _16, _128>>>;
    // use cute::uint8_t as InternalType
    using SmemTiledCopyVt =
        TiledCopy<Copy_Atom<SM75_U16x4_LDSM_T, cute::uint8_t>,
                  Layout_TV_Vt,
                  Shape<_16, _16>>;  // K x N
    return SmemTiledCopyVt{};
  } else {
    CUTE_STATIC_ASSERT_V(
        sizeof_bits_v<DType> == 8 || sizeof_bits_v<DType> == 16,
        "DType must be 8 or 16 bits");
  }
}

}  // namespace detail

template <typename DTYPE,
          typename KV_DTYPE,
          int HEAD_DIM,
          int BLK_M,
          int BLK_N,
          int BLK_K>
struct AttentionTraitsSM80 {
  // helpful aliases
  static constexpr int kHeadDim = HEAD_DIM;
  static constexpr int kBlockM = BLK_M;
  static constexpr int kBlockN = BLK_N;
  static constexpr int kBlockK = BLK_K;
  static constexpr int kRowsPerMMA = 2;

  using DType = DTYPE;
  using KV_DType = KV_DTYPE;
  using _BLK_M = Int<kBlockM>;
  using _BLK_N = Int<kBlockN>;
  using _BLK_K = Int<kBlockK>;
  using _HEAD_DIM = Int<kHeadDim>;

  // ******* Mainloop *******
  // TiledMMA (64x16x16) for gemm-I and gemm-II
  // choose MMA_Atom based on Element type
  using TiledMma = decltype(detail::tiled_mma_selector<DType, KV_DType>());
  static constexpr size_t kThreadNum = size(TiledMma{});

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

  // V^T smem (transpose view of V): (HEAD_DIM, BLK_N)
  using SmemLayoutVt = decltype(composition(
      SmemLayoutV{},
      make_layout(Shape<_HEAD_DIM, _BLK_N>{}, GenRowMajor{})));

  // Tiled copy for QKV
  // g2s tiled copy for q
  using GmemTiledCopyQ =
      decltype(detail::tiled_copy_selector<
               Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, DType>,
               BLK_K,
               kThreadNum>());

  // g2s tiled copy for kv
  // TODO: choose based on BLK_K and kv cache type
  using GmemTiledCopyKV =
      decltype(detail::tiled_copy_selector<
               Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, KV_DType>,
               BLK_K,
               kThreadNum>());

  // s2r tiled copy for gemm-I
  using SmemTiledCopyQ =
      decltype(make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, DType>{},
                                 TiledMma{}));
  using SmemTiledCopyK =
      decltype(detail::tiled_copy_B_selector<KV_DType, TiledMma>());

  // s2r tiled copy for gemm-II
  using SmemTiledCopyVt =
      decltype(detail::tiled_copy_B_T_selector<KV_DType, TiledMma>());

  // ******* Epilogue *******

  // O smem: (BLK_M, K):(K, 1), k-major, same as Q
  using SmemLayoutO = SmemLayoutQ;

  // s2g tiled copy for O
  using GmemTiledCopyO =
      decltype(detail::tiled_copy_selector<Copy_Atom<DefaultCopy, DType>,
                                           BLK_K,
                                           kThreadNum>());

  // r2s tiled copy for O
  using SmemTiledCopyO =
      decltype(make_tiled_copy_C(Copy_Atom<DefaultCopy, DType>{}, TiledMma{}));

  // constexpr values for kernel launch
  static constexpr size_t kSmemSize =
      cosize(SmemLayoutQ{}) * sizeof(DType) +
      (cosize(SmemLayoutK{}) + cosize(SmemLayoutV{})) * sizeof(KV_DType);
};

}  // namespace llm