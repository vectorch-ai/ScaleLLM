#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cute/config.hpp>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>

#include "fast_cast.cuh"
#include "safe_copy.hpp"

namespace llm {
using namespace cute;

template <class TileShape_, class Element_, bool EVEN_N_>
struct Sm80CollectiveEpilogue {
  using TileShape = TileShape_;
  using Element = Element_;

  static constexpr bool EVEN_N = EVEN_N_;

  static constexpr int kBlockM = get<0>(TileShape{});
  static constexpr int kBlockN = get<1>(TileShape{});

  using BLK_M = Int<kBlockM>;
  using BLK_N = Int<kBlockN>;

  // Shared memory LayoutAtom (8x64)
  using SmemLayoutAtom_8x64 =
      decltype(composition(Swizzle<3, 3, 3>{},
                           Layout<Shape<_8, _64>, Stride<_64, _1>>{}));
  using SmemLayoutAtom_8x32 =
      decltype(composition(Swizzle<2, 3, 3>{},
                           Layout<Shape<_8, _32>, Stride<_32, _1>>{}));

  using SmemLayoutAtomC = std::conditional_t<kBlockN % 64 == 0,
                                             SmemLayoutAtom_8x64,
                                             SmemLayoutAtom_8x32>;
  using SmemLayoutC =
      decltype(tile_to_shape(SmemLayoutAtomC{}, Shape<BLK_M, BLK_N>{}));

  // use 128-bit vectorizing copy
  using VectorizingCopy_ = AutoVectorizingCopyWithAssumedAlignment<128>;
  // r2s tiled copy for C
  using SmemCopyAtomC_ = Copy_Atom<VectorizingCopy_, Element>;

  // Thread layout for gmem copy: (_16,_8)/(_32, _4)
  using GmemCopyThrLayout =
      std::conditional_t<kBlockN == 32,
                         Layout<Shape<_32, _4>, Stride<_4, _1>>,
                         Layout<Shape<_16, _8>, Stride<_8, _1>>>;

  // s2g tiled copy for O
  using GmemTiledCopyC = decltype(make_tiled_copy(
      Copy_Atom<VectorizingCopy_, Element>{},
      GmemCopyThrLayout{},     // Thr layout: (_16,_8)/(_32, _4)
      Layout<Shape<_1, _8>>{}  // Val layout: 8 vals per read
      ));

  struct SharedStorage : cute::aligned_struct<128> {
    // Shared memory for C: (BLK_M, BLK_N)
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutC>> c_smem;
  };

  // Host side kernel arguments
  struct Arguments {};

  // Device side kernel params
  using Params = Arguments;

  // Convert host side arguments to device side params
  static Params to_underlying_arguments(Arguments const& args) { return args; }

  template <class FrgTensor,
            class TiledMma,
            class TensorC,
            class IdentTensorC,
            class PredTensor,
            class ResidueMNK>
  CUTE_DEVICE void operator()(const Params& /*params*/,
                              const FrgTensor& tCrAccC,  // (MMA, MMA_M, MMA_N)
                              TiledMma tiled_mma,
                              TensorC& gC,       // (BLK_M, BLK_M)
                              IdentTensorC& cC,  // (BLK_M, BLK_N) => (M, N)
                              PredTensor& tGpC,  // (BLK_M) => bool
                              int tidx,
                              ResidueMNK residue_mnk,
                              char* smem) {
    static constexpr int kBlockM = get<0>(TileShape{});

    auto residue_mn = select<0, 1>(residue_mnk);

    // Smem
    auto& ss = *reinterpret_cast<SharedStorage*>(smem);
    // (BLK_M, BLK_N)
    auto sC = make_tensor(make_smem_ptr(ss.c_smem.data()), SmemLayoutC{});

    // fastcast tCrAccC to Element
    auto tCrC = make_tensor_like<Element>(tCrAccC);
    fast_cast(tCrAccC, tCrC);

    // copy tCrC from registers to smem
    auto smem_tiled_copy_c = make_tiled_copy_C(SmemCopyAtomC_{}, tiled_mma);
    auto smem_thr_copy_c = smem_tiled_copy_c.get_thread_slice(tidx);
    auto tSrC = smem_thr_copy_c.retile_S(tCrC);
    auto tSsC = smem_thr_copy_c.partition_D(sC);
    cute::copy(smem_tiled_copy_c, tSrC, tSsC);

    // wait for smem copy done before gmem copy
    __syncthreads();

    // copy sC from smem to gmem
    GmemTiledCopyC gmem_tiled_copy_c;
    auto gmem_thr_copy_c = gmem_tiled_copy_c.get_thread_slice(tidx);
    auto tGsC = gmem_thr_copy_c.partition_S(sC);
    auto tGgC = gmem_thr_copy_c.partition_D(gC);
    // (CPY, CPY_M, CPY_N) => (M, N)
    auto tGcC = gmem_thr_copy_c.partition_D(cC);
    safe_copy_m<EVEN_N, /*ZFILL_M=*/false, /*ZFILL_K=*/false>(
        gmem_tiled_copy_c, tGsC, tGgC, tGpC, tGcC, residue_mn);
  }
};
}  // namespace llm
