#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cute/container/array_aligned.hpp>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>

#include "fast_cast.cuh"
#include "safe_copy.h"

namespace llm {
using namespace cute;

template <class TileShape_, class Element_, int HeadDim_, bool EVEN_K_>
struct Sm80CollectiveEpilogue {
  using TileShape = TileShape_;
  using Element = Element_;

  static constexpr int kHeadDim = HeadDim_;
  static constexpr bool EVEN_K = EVEN_K_;

  static constexpr int kBlockM = get<0>(TileShape{});
  static constexpr int kBlockK = get<2>(TileShape{});

  using BLK_M = Int<kBlockM>;
  using BLK_K = Int<kBlockK>;
  using HEAD_DIM = Int<kHeadDim>;

  using SmemLayoutAtom_ =
      decltype(composition(Swizzle<3, 3, 3>{},
                           Layout<Shape<_8, BLK_K>, Stride<BLK_K, _1>>{}));

  // Q smem: (BLK_M, HEAD_DIM)
  using SmemLayoutO =
      decltype(tile_to_shape(SmemLayoutAtom_{}, Shape<BLK_M, HEAD_DIM>{}));

  // use 128-bit vectorizing copy
  using VectorizingCopy_ = AutoVectorizingCopyWithAssumedAlignment<128>;

  // r2s copy atom for O
  using SmemCopyAtom_ = Copy_Atom<VectorizingCopy_, Element>;

  // Thr layout for gmem copy
  using GmemCopyThrLayout_ =
      std::conditional_t<kBlockK == 32,
                         Layout<Shape<_32, _4>, Stride<_4, _1>>,
                         Layout<Shape<_16, _8>, Stride<_8, _1>>>;

  // s2g tiled copy for O
  using GmemTiledCopyO = decltype(make_tiled_copy(
      Copy_Atom<VectorizingCopy_, Element>{},
      GmemCopyThrLayout_{},    // Thr layout: (_16,_8)/(_32, _4)
      Layout<Shape<_1, _8>>{}  // Val layout: 8 vals per read
      ));

  struct SharedStorage : cute::aligned_struct<128> {
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutO>> smem_o;
  };

  // Host side kernel arguments
  struct Arguments {};

  // Device side kernel params
  using Params = Arguments;

  // Convert host side arguments to device side params
  static Params to_underlying_arguments(Arguments const& args) { return args; }

  template <class FrgTensor,
            class TiledMma,
            class TensorO,
            class BlockCoordMNK,
            class ProblemShapeMNK>
  CUTE_DEVICE void operator()(const Params& /*params*/,
                              const FrgTensor& tOrAccO,  // (MMA, MMA_M, MMA_N)
                              TiledMma tiled_mma,
                              TensorO& gO,  // (BLK_M, HEAD_DIM)
                              int tidx,
                              const BlockCoordMNK& block_coord_mnk,
                              const ProblemShapeMNK& problem_shape_mnk,
                              char* smem) {
    static constexpr int kBlockM = get<0>(TileShape{});

    const auto [batch_idx, m_block_idx, kv_head_idx] = block_coord_mnk;
    const auto [q_packed_len, kv_len, head_dim] = problem_shape_mnk;

    // Smem
    auto& ss = *reinterpret_cast<SharedStorage*>(smem);
    // (BLK_M, HEAD_DIM)
    Tensor sO = make_tensor(make_smem_ptr(ss.smem_o.data()), SmemLayoutO{});

    // 1. cast output from ElementAccumulator to Element
    auto tOrO = make_tensor_like<Element>(tOrAccO);
    fast_cast(tOrAccO, tOrO);

    // 2. copy output from reg to smem
    auto smem_tiled_copy_O = make_tiled_copy_C(SmemCopyAtom_{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
    auto tSrO = smem_thr_copy_O.retile_S(tOrO);
    auto tSsO = smem_thr_copy_O.partition_D(sO);
    cute::copy(smem_tiled_copy_O, tSrO, tSsO);

    // 3. copy output from smem to gmem
    GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);

    // (BLK_M, HEAD_DIM) -> (blk_m, head_dim)
    auto cO = make_identity_tensor(Shape<BLK_M, HEAD_DIM>{});

    auto tOsO = gmem_thr_copy_O.partition_S(sO);  // (CPY,CPY_M,CPY_K)
    auto tOgO = gmem_thr_copy_O.partition_D(gO);  // (CPY,CPY_M,CPY_K)
    // (CPY,CPY_M,CPY_K) -> (blk_m, head_dim)
    auto tOcO = gmem_thr_copy_O.partition_D(cO);

    // wait for smem copy done before gmem copy
    __syncthreads();

    auto max_coord = make_coord(q_packed_len - m_block_idx * kBlockM, head_dim);
    safe_copy</*EVEN_M=*/false, EVEN_K, /*ZFILL_M=*/false, /*ZFILL_K=*/false>(
        gmem_tiled_copy_O, tOsO, tOgO, tOcO, max_coord);
  }
};
}  // namespace llm
