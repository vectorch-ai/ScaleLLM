#pragma once
#include <cute/tensor.hpp>

namespace llm {
using namespace cute;

template <typename T_, int kHeadDim_ = 64, int kBlockM_ = 64, int kBlockN_ = 64>
struct AttentionTraitsSM80 {
  using T = T_;
  static constexpr int kHeadDim = kHeadDim_;
  static constexpr int kBlockM = kBlockM_;
  static constexpr int kBlockN = kBlockN_;
  static constexpr int kBlockK = kHeadDim % 64 == 0 ? 64 : 32;

  // tiled MMA (64x16x16)
  using TiledMMA = TiledMMA<MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
                            Layout<Shape<_4, _1, _1>>,  // warp layout 4x1x1
                            Tile<_64, _16, _16>>;       // Prom Shape 64x16x16

  static constexpr int kThreadNum = size(TiledMMA{});

  // s2r TiledCopy
  // Atom for QK: 16x16
  using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, T>;
  // Copy Atom for V^T
  using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, T>;

  // Copy Atom for O
  using SmemCopyAtomO = Copy_Atom<DefaultCopy, T>;

  // g2s tiled copy for qkv
  using GmemTiledCopyQKV = decltype(make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, T>{},
      Layout<Shape<_16, _8>, Stride<_8, _1>>{},  // Thr layout: (_16,_8):(_8,_1)
      Layout<Shape<_1, _8>>{}                    // Val layout: 8 vals per read
      ));

  // g2s tiled copy for O
  using GmemTiledCopyO = decltype(make_tiled_copy(
      Copy_Atom<UniversalCopy<cute::uint128_t>, T>{},
      Layout<Shape<_16, _8>, Stride<_8, _1>>{},  // Thr layout: (_16,_8):(_8,_1)
      Layout<Shape<_1, _8>>{}                    // Val layout: 8 vals per read
      ));

  // Q smem: (BLK_M, K):(K, 1) row-major
  using SmemLayoutQ =
      Layout<Shape<Int<kBlockM>, Int<kHeadDim>>, Stride<Int<kHeadDim>, _1>>;

  // KV smem: (BLK_N, K):(K, 1)
  using SmemLayoutKV =
      Layout<Shape<Int<kBlockN>, Int<kHeadDim>>, Stride<Int<kHeadDim>, _1>>;

  using SmemLayoutVt =
      Layout<Shape<Int<kHeadDim>, Int<kBlockN>>, Stride<_1, Int<kHeadDim>>>;

  // O smem layout: (BLK_M, K):(K, 1), same as Q
  using SmemLayoutO =
      Layout<Shape<Int<kBlockM>, Int<kHeadDim>>, Stride<Int<kHeadDim>, _1>>;

  static constexpr int shm_size_q = cute::cosize(SmemLayoutQ{});
  static constexpr int shm_size_kv = cute::cosize(SmemLayoutKV{}) * 2;
  static constexpr int kShmSize = (shm_size_kv + shm_size_q) * sizeof(half);
};

}  // namespace llm