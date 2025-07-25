#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cute/container/array_aligned.hpp>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>

#include "common/fast_cast.cuh"
#include "common/safe_copy.h"
#include "common/selector.h"

namespace llm {
using namespace cute;

template <class TileShape_, class Element_, bool EVEN_K_>
struct Sm80CollectiveEpilogue {
  using TileShape = TileShape_;
  using Element = Element_;

  static constexpr int kThreads = 128;
  static constexpr bool EVEN_K = EVEN_K_;

  static constexpr int kBlockM = get<0>(TileShape{});
  static constexpr int kBlockK = get<2>(TileShape{});

  using BLK_M = Int<kBlockM>;
  using BLK_K = Int<kBlockK>;

  using SmemLayoutAtom_ =
      decltype(smem_layout_atom_selector<Element, kBlockK>());

  // Q smem: (BLK_M, BLK_K)
  using SmemLayoutO =
      decltype(tile_to_shape(SmemLayoutAtom_{}, Shape<BLK_M, BLK_K>{}));

  // use 128-bit vectorizing copy
  using VectorizingCopy_ = AutoVectorizingCopyWithAssumedAlignment<128>;

  // r2s copy atom for O
  using SmemCopyAtom_ = Copy_Atom<VectorizingCopy_, Element>;

  // s2g tiled copy for O
  using GmemTiledCopyO =
      decltype(gmem_tiled_copy_selector<Element, kThreads, kBlockK>(
          Copy_Atom<VectorizingCopy_, Element>{}));

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
  CUTE_DEVICE void operator()(const Params& /*params*/,
                              const FrgTensor& tOrAccO,  // (MMA, MMA_M, MMA_N)
                              TiledMma tiled_mma,
                              TensorO& gO,         // (BLK_M, BLK_K)
                              const TensorCO& cO,  // (BLK_M, BLK_K) => (M, K)
                              int tidx,
                              const ResidueMNK& residue_mnk,
                              char* smem) {
    static constexpr int kBlockM = get<0>(TileShape{});

    // Smem
    auto& ss = *reinterpret_cast<SharedStorage*>(smem);
    // (BLK_M, BLK_K)
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

    auto tOsO = gmem_thr_copy_O.partition_S(sO);  // (CPY,CPY_M,CPY_K)
    auto tOgO = gmem_thr_copy_O.partition_D(gO);  // (CPY,CPY_M,CPY_K)
    // (CPY,CPY_M,CPY_K) -> (blk_m, head_dim)
    auto tOcO = gmem_thr_copy_O.partition_D(cO);

    // wait for smem copy done before gmem copy
    __syncthreads();

    const auto residue_mk = select<0, 2>(residue_mnk);
    safe_copy</*EVEN_M=*/false, EVEN_K, /*ZFILL_M=*/false, /*ZFILL_K=*/false>(
        gmem_tiled_copy_O, tOsO, tOgO, tOcO, residue_mk);
  }
};
}  // namespace llm
