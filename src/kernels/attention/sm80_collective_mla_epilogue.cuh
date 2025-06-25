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

template <class TileShape_, class Element_, int HeadDim_>
struct Sm80CollectiveMlaEpilogue {
  using TileShape = TileShape_;
  using Element = Element_;

  static constexpr int kHeadDim = HeadDim_;

  static constexpr int kBlockM = get<0>(TileShape{});
  static constexpr int kBlockK = get<2>(TileShape{});
  // number of steps per stage
  static constexpr int kSteps = kHeadDim / kBlockK;

  using BLK_M = Int<kBlockM>;
  using BLK_K = Int<kBlockK>;
  using HEAD_DIM = Int<kHeadDim>;
  using STEPS = Int<kSteps>;

  // Shared memory LayoutAtom for differnt block sizes
  using SmemLayoutAtom_8x64 =
      decltype(composition(Swizzle<3, 3, 3>{},
                           Layout<Shape<_8, _64>, Stride<_64, _1>>{}));
  using SmemLayoutAtom_8x32 =
      decltype(composition(Swizzle<2, 3, 3>{},
                           Layout<Shape<_8, _32>, Stride<_32, _1>>{}));
  using SmemLayoutAtom_ = std::conditional_t<kBlockK % 64 == 0,
                                             SmemLayoutAtom_8x64,
                                             SmemLayoutAtom_8x32>;

  // Q smem: (BLK_M, HEAD_DIM)
  using SmemLayoutO =
      decltype(tile_to_shape(SmemLayoutAtom_{}, Shape<BLK_M, BLK_K, STEPS>{}));

  // use 128-bit vectorizing copy
  using VectorizingCopy_ = AutoVectorizingCopyWithAssumedAlignment<128>;

  // r2s copy atom for O
  using SmemCopyAtom_ = Copy_Atom<VectorizingCopy_, Element>;

  // s2g tiled copy for O
  using GmemTiledCopyO = decltype(make_tiled_copy(
      Copy_Atom<VectorizingCopy_, Element>{},
      Layout<Shape<_32, _8>, Stride<_8, _1>>{},  // Thr layout: (_32, _8)
      Layout<Shape<_1, _8>>{}                    // Val layout: 8 vals per read
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
            class TensorCO,
            class ResidueMNK>
  CUTE_DEVICE void operator()(
      const Params& /*params*/,
      const FrgTensor& tOrAccO,  // (MMA, MMA_M, MMA_N, k)
      TiledMma tiled_mma,
      TensorO& gO,         // (BLK_M, BLK_K, k)
      const TensorCO& cO,  // (BLK_M, BLK_K, k) => (m, k)
      int tidx,
      const ResidueMNK& residue_mnk,
      char* smem) {
    static constexpr int kBlockM = get<0>(TileShape{});
    static constexpr int kBlockK = get<2>(TileShape{});

    // Smem
    auto& ss = *reinterpret_cast<SharedStorage*>(smem);
    // (BLK_M, BLK_K, k)
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

    // wait for smem copy done before gmem copy
    __syncthreads();

    // 3. copy output from smem to gmem
    GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);

    auto tOsO = gmem_thr_copy_O.partition_S(sO);  // (CPY,CPY_M,CPY_K, k)
    auto tOgO = gmem_thr_copy_O.partition_D(gO);  // (CPY,CPY_M,CPY_K, k)

    // (CPY,CPY_M,CPY_K, k) -> (m, k)
    auto tOcO = gmem_thr_copy_O.partition_D(cO);
    auto residue_mk = select<0, 2>(residue_mnk);
    safe_copy</*EVEN_M=*/false,
              /*EVEN_K=*/true,
              /*ZFILL_M=*/false,
              /*ZFILL_K=*/false>(
        gmem_tiled_copy_O, tOsO, tOgO, tOcO, residue_mk);
  }
};
}  // namespace llm
