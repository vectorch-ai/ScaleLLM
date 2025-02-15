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

  static_assert(kHeadDim % 64 == 0);
  static_assert(kRopeHeadDim % 64 == 0);

  static_assert(kBlockM % 64 == 0);
  static_assert(kBlockN % 16 == 0);
  static_assert(kBlockK % 64 == 0);

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

  using TiledMma_64x32x16_ =
      TiledMMA<MMA_Atom_,
               Layout<Shape<_4, _2, _1>>,  // warp layout 4x2x1
               Tile<_64, _32, _16>>;       // Shape 64x32x16
  using TiledMma_64x16x16_ =
      TiledMMA<MMA_Atom_,
               Layout<Shape<_4, _2, _1>>,  // warp layout 4x2x1
               Tile<_64, _16, _16>>;       // Shape 64x16x16

  // TiledMma for P = Softmax(Q*K^T), warp layout 4x2x1
  using TiledMma_QK = std::conditional_t<kBlockN % 32 == 0,
                                         TiledMma_64x32x16_,
                                         TiledMma_64x16x16_>;

  // TiledMma for O = P*V^T, warp layout 4x2x1
  using TiledMma_PV = TiledMma_64x32x16_;

  // Layout convertor for TiledMMA (64x16x16)
  using LayoutConvertor = detail::LayoutConvertor;

  // Smem LayoutAtom
  using SmemLayoutAtom_8x64 =
      decltype(composition(Swizzle<3, 3, 3>{},
                           Layout<Shape<_8, _64>, Stride<_64, _1>>{}));
  using SmemLayoutAtom_8x32 =
      decltype(composition(Swizzle<2, 3, 3>{},
                           Layout<Shape<_8, _32>, Stride<_32, _1>>{}));
  using SmemLayoutAtom_8x16 =
      decltype(composition(Swizzle<1, 3, 3>{},
                           Layout<Shape<_8, _16>, Stride<_16, _1>>{}));

  using SmemLayoutAtomK = std::conditional_t<kBlockK % 64 == 0,
                                             SmemLayoutAtom_8x64,
                                             SmemLayoutAtom_8x32>;
  using SmemLayoutAtomN =
      std::conditional_t<kBlockN % 64 == 0,
                         SmemLayoutAtom_8x64,
                         std::conditional_t<kBlockN % 32 == 0,
                                            SmemLayoutAtom_8x32,
                                            SmemLayoutAtom_8x16>>;
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

  // shared memory for sync between warp groups
  // rowmax/rowsum smem: (_BLK_M, _2)
  using SmemLayoutRowmax = Layout<Shape<Int<2 * kBlockM>>>;
  using SmemLayoutRowsum = Layout<Shape<Int<2 * kBlockM>>>;

  // Tiled copy for differnt block sizes
  using GmemTiledCopy_32x64_ = decltype(make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, DType>{},
      Layout<Shape<_32, _8>, Stride<_8, _1>>{},  // Thr layout: (_32, _8)
      Layout<Shape<_1, _8>>{}                    // Val layout: 8 vals per read
      ));
  using GmemTiledCopy_16x128_ = decltype(make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, DType>{},
      Layout<Shape<_16, _16>, Stride<_16, _1>>{},  // Thr layout: (_16, _16)
      Layout<Shape<_1, _8>>{}  // Val layout: 8 vals per read
      ));
  using GmemTiledCopy_16x64_ = decltype(make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint64_t>, DType>{},
      Layout<Shape<_16, _16>, Stride<_16, _1>>{},  // Thr layout: (_16, _16)
      Layout<Shape<_1, _4>>{}  // Val layout: 4 vals per read
      ));

  // g2s tiled copy for q
  using GmemTiledCopyQ = GmemTiledCopy_32x64_;
  // g2s tiled copy for q_rope
  using GmemTiledCopyQRope = GmemTiledCopy_32x64_;

  // g2s tiled copy for kv: (32x64), (16x64) or (16x128),
  using GmemTiledCopyKV =
      std::conditional_t<kBlockN % 32 == 0,
                         GmemTiledCopy_32x64_,
                         std::conditional_t<kBlockK % 128 == 0,
                                            GmemTiledCopy_16x128_,
                                            GmemTiledCopy_16x64_>>;

  // g2s tiled copy for k_rope: (32x64) or (16x64)
  using GmemTiledCopyKRope = std::conditional_t<kBlockN % 32 == 0,
                                                GmemTiledCopy_32x64_,
                                                GmemTiledCopy_16x64_>;

  // s2r tiled copy for gemm-I S = Q*K^T
  // warp layout: 4x2x1, tiledmma mxnxk: 64x32x16 or 64x16x16
  // Smem tiled copy for Q, 4 warps mxk: 64x16
  using SmemTiledCopyQ =
      decltype(make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, DType>{},
                                 TiledMma_QK{}));

  using Copy_Atom_K = std::conditional_t<kBlockN % 32 == 0,
                                         Copy_Atom<SM75_U32x4_LDSM_N, DType>,
                                         Copy_Atom<SM75_U32x2_LDSM_N, DType>>;
  // Smem tiled copy for KV, 2 warps nxk: 32x16 or 16x16
  using SmemTiledCopyK =
      decltype(make_tiled_copy_B(Copy_Atom_K{}, TiledMma_QK{}));

  // r2s tiled copy for gemm-I S
  // use 128-bit vectorizing copy
  using VectorizingCopy = AutoVectorizingCopyWithAssumedAlignment<128>;

  using SmemTiledCopyS =
      decltype(make_tiled_copy_C(Copy_Atom<VectorizingCopy, DType>{},
                                 TiledMma_QK{}));

  // s2r tiled copy for gemm-II: O = P*V^T
  // warp layout: 4x2x1, TiledMma mxnxk: 64x32x16
  // Smem tiled copy for P, 4 warps mxk: 64x16
  using SmemTiledCopyP =
      decltype(make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, DType>{},
                                 TiledMma_PV{}));

  // Smem tiled copy for V^T, 2 warps nxk: 32x16
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

  // s2g tiled copy for O (32x64)
  using GmemTiledCopyO = decltype(make_tiled_copy(
      Copy_Atom<VectorizingCopy, DType>{},
      Layout<Shape<_32, _8>, Stride<_8, _1>>{},  // Thr layout: (_32, _8)
      Layout<Shape<_1, _8>>{}                    // Val layout: 8 vals per read
      ));

  // constexpr values for kernel launch
  static constexpr size_t kSmemSize =
      sizeof(DType) * (cosize(SmemLayoutQ{}) + cosize(SmemLayoutKV{}) +
                       cosize(SmemLayoutP{}) + cosize(SmemLayoutQRope{}) +
                       cosize(SmemLayoutKRope{})) +
      sizeof(float) * (cosize(SmemLayoutRowmax{}));

  static constexpr size_t kThreadNum = size(TiledMma_PV{});
};

}  // namespace llm