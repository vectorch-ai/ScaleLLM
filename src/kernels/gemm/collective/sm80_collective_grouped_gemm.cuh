#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include <cute/config.hpp>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>

#include "common/safe_copy.h"

namespace llm {
using namespace cute;

template <int Stages,
          class TileShape_,
          class Element_,
          bool EVEN_N,
          bool EVEN_K>
struct Sm80CollectiveGroupedGEMM {
  using TileShape = TileShape_;
  using Element = Element_;
  using ElementAccum = float;

  static constexpr int kBlockM = get<0>(TileShape{});
  static constexpr int kBlockN = get<1>(TileShape{});
  static constexpr int kBlockK = get<2>(TileShape{});

  static_assert(kBlockM % 64 == 0);
  static_assert(kBlockN % 32 == 0);
  static_assert(kBlockK % 16 == 0);

  using BLK_M = Int<kBlockM>;
  using BLK_N = Int<kBlockN>;
  using BLK_K = Int<kBlockK>;
  using PIPE = Int<Stages>;

  // MMA Atom: (16x8x16) for F32F16F16F32 or F32BF16BF16F32
  using MMA_Atom_ =
      std::conditional_t<std::is_same_v<Element, cute::half_t>,
                         MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
                         MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>>;

  // TiledMMA: (64x16x16)
  using TiledMma = TiledMMA<MMA_Atom_,
                            Layout<Shape<_4, _1, _1>>,  // warp layout: (4x1x1)
                            Tile<_64, _16, _16>>;  // tile layout: (64x16x16)

  static constexpr int kMmaThreads = size(TiledMma{});

  // Shared memory LayoutAtom (8x64)
  using SmemLayoutAtom_8x64 =
      decltype(composition(Swizzle<3, 3, 3>{},
                           Layout<Shape<_8, _64>, Stride<_64, _1>>{}));
  using SmemLayoutAtom_8x32 =
      decltype(composition(Swizzle<2, 3, 3>{},
                           Layout<Shape<_8, _32>, Stride<_32, _1>>{}));

  using SmemLayoutAtom = std::conditional_t<kBlockK % 64 == 0,
                                            SmemLayoutAtom_8x64,
                                            SmemLayoutAtom_8x32>;
  // SMEM Layout for A: (BLK_M, BLK_K, PIPE)
  using SmemLayoutA =
      decltype(tile_to_shape(SmemLayoutAtom{}, Shape<BLK_M, BLK_K, PIPE>{}));
  // SMEM Layout for B: (BLK_N, BLK_K, PIPE)
  using SmemLayoutB =
      decltype(tile_to_shape(SmemLayoutAtom{}, Shape<BLK_N, BLK_K, PIPE>{}));

  // Thread layout for gmem copy: (_16,_8)/(_32, _4)
  using GmemCopyThrLayout =
      std::conditional_t<kBlockK == 32,
                         Layout<Shape<_32, _4>, Stride<_4, _1>>,
                         Layout<Shape<_16, _8>, Stride<_8, _1>>>;
  // g2s tiled copy: copy A/B from global memory to shared memory
  using GmemTiledCopy = decltype(make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, Element>{},
      GmemCopyThrLayout{},     // Thr layout: (_16,_8)/(_32, _4)
      Layout<Shape<_1, _8>>{}  // Val layout: 8 vals per read
      ));

  // s2r tiled copy for A and B
  using SmemTiledCopyA =
      decltype(make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, Element>{},
                                 TiledMma{}));
  using SmemTiledCopyB =
      decltype(make_tiled_copy_B(Copy_Atom<SM75_U32x4_LDSM_N, Element>{},
                                 TiledMma{}));

  struct SharedStorage : cute::aligned_struct<128> {
    // Shared memory for A: (BLK_M, BLK_K, PIPE)
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutA>> a_smem;
    // Shared memory for B: (BLK_N, BLK_K, PIPE)
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutB>> b_smem;
  };

  // Host side arguments
  struct Arguments {
    const int* sorted_token_idxes_ptr = nullptr;
    int n_flatten_tokens = 0;
  };

  // Device side params
  using Params = Arguments;

  // Convert host side arguments to device side params
  static Params to_underlying_arguments(Arguments const& args) {
    // no convertion needed.
    return args;
  }

  // returns false if the block has been skipped
  template <class TensorA,
            class IdentTensorA,
            class TensorB,
            class IdentTensorB,
            class FrgTensor,
            class ResidueMNK>
  CUTE_DEVICE void operator()(
      const Params& params,
      const TensorA& gA,       // (BLK_M, BLK_K, k)
      const IdentTensorA& cA,  // (BLK_M, BLK_K, k) => (M, K)
      const TensorB& gB,       // (BLK_N, HEAD_DIM, n)
      const IdentTensorB& cB,  // (BLK_N, BLK_K, k) => (N, K)
      FrgTensor& tCrAccC,      // (MMA, MMA_M, MMA_N)
      int tidx,
      const ResidueMNK& residue_mnk,
      char* smem) {
    static_assert(is_rmem<FrgTensor>::value,
                  "Accum tensor must be rmem resident.");
    static_assert(is_gmem<TensorA>::value, "A tensor must be gmem resident.");
    static_assert(is_gmem<TensorB>::value, "B tensor must be gmem resident.");

    const auto residue_mk = select<0, 2>(residue_mnk);
    const auto residue_nk = select<1, 2>(residue_mnk);

    auto& ss = *reinterpret_cast<SharedStorage*>(smem);

    // (BLK_M, BLK_K, PIPE)
    Tensor sA = make_tensor(make_smem_ptr(ss.a_smem.data()), SmemLayoutA{});
    // (BLK_N, BLK_K, PIPE)
    Tensor sB = make_tensor(make_smem_ptr(ss.b_smem.data()), SmemLayoutB{});

    // Tiled Copy
    GmemTiledCopy gmem_tiled_copy;
    auto gmem_thr_copy = gmem_tiled_copy.get_thread_slice(tidx);

    // (BLK_M, BLK_K, k) => (CPY, CPY_M, CPY_K, k)
    auto tAgA = gmem_thr_copy.partition_S(gA);
    // (CPY, CPY_M, CPY_K, k) => (M, K)
    auto tAcA = gmem_thr_copy.partition_S(cA);
    // (BLK_M, BLK_K, PIPE) => (CPY, CPY_M, CPY_K, PIPE)
    auto tAsA = gmem_thr_copy.partition_D(sA);

    // (CPY_M) => (M, K)
    const int* sorted_token_idxes = params.sorted_token_idxes_ptr;
    const int n_flatten_tokens = params.n_flatten_tokens;
    auto tAcA_m = tAcA(_0{}, _, _0{}, _0{});
    // (CPY_M) => bool
    auto tApA = make_tensor<bool>(make_shape(size(tAcA_m)));
    CUTE_UNROLL
    for (int i = 0; i < size(tApA); ++i) {
      const auto f_idx = sorted_token_idxes[get<0>(tAcA_m(i))];
      tApA(i) = f_idx < n_flatten_tokens;
    }

    // (BLK_N, BLK_K, k) => (CPY, CPY_N, CPY_K, k)
    auto tBgB = gmem_thr_copy.partition_S(gB);
    // (CPY, CPY_N, CPY_K, k) => (N, K)
    auto tBcB = gmem_thr_copy.partition_S(cB);
    // (BLK_N, BLK_K, PIPE) => (CPY, CPY_N, CPY_K, PIPE)
    auto tBsB = gmem_thr_copy.partition_D(sB);

    auto produce_ab = [&](int k_tile, int k_pipe) {
      safe_copy_m<EVEN_K, /*ZFILL_M=*/true, /*ZFILL_K=*/true>(
          gmem_tiled_copy,
          tAgA(_, _, _, k_tile),
          tAsA(_, _, _, k_pipe),
          tApA,
          tAcA(_, _, _, k_tile),
          residue_mk);

      safe_copy_n<EVEN_N, EVEN_K, /*ZFILL_N=*/true, /*ZFILL_K=*/true>(
          gmem_tiled_copy,
          tBgB(_, _, _, k_tile),
          tBsB(_, _, _, k_pipe),
          tBcB(_, _, _, k_tile),
          residue_nk);
    };

    auto produce_ab_no_oob = [&](int k_tile, int k_pipe) {
      safe_copy_m<EVEN_K, /*ZFILL_M=*/false, /*ZFILL_K=*/true>(
          gmem_tiled_copy,
          tAgA(_, _, _, k_tile),
          tAsA(_, _, _, k_pipe),
          tApA,
          tAcA(_, _, _, k_tile),
          residue_mk);

      safe_copy_n<EVEN_N, EVEN_K, /*ZFILL_N=*/false, /*ZFILL_K=*/true>(
          gmem_tiled_copy,
          tBgB(_, _, _, k_tile),
          tBsB(_, _, _, k_pipe),
          tBcB(_, _, _, k_tile),
          residue_nk);
    };

    // GEMM: C = A@B.T
    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    // rA: (BLK_M, BLK_K) => (MMA,MMA_M,MMA_K)
    auto tCrA = thr_mma.partition_fragment_A(sA(_, _, _0{}));
    // rB: (BLK_N, BLK_K) => (MMA,MMA_N,MMA_K)
    auto tCrB = thr_mma.partition_fragment_B(sB(_, _, _0{}));

    // s2r tiled copy for A and B
    auto smem_tiled_copy_a = SmemTiledCopyA{};
    auto smem_thr_copy_a = smem_tiled_copy_a.get_thread_slice(tidx);
    // (BLK_M, BLK_K, PIPE) => (CPY, CPY_M, CPY_K, PIPE)
    auto tCsA = smem_thr_copy_a.partition_S(sA);
    // (CPY, CPY_M, CPY_K)
    auto tCrA_cpv = smem_thr_copy_a.retile_D(tCrA);

    auto smem_tiled_copy_b = SmemTiledCopyB{};
    auto smem_thr_copy_b = smem_tiled_copy_b.get_thread_slice(tidx);
    // (BLK_N, BLK_K, PIPE) => (CPY, CPY_N, CPY_K, PIPE)
    auto tCsB = smem_thr_copy_b.partition_S(sB);
    // (CPY, CPY_N, CPY_K)
    auto tCrB_cpv = smem_thr_copy_b.retile_D(tCrB);

    // ###############  Prologue  ###############
    // remaining k-tile count
    int k_tiles_remaining = size<3>(tAgA);
    // next tile index in gmem to read from
    int k_tile = 0;

    // async loads for all pipes except the last one
    auto kPipe = size<3>(tAsA);
    CUTE_UNROLL
    for (int k_pipe = 0; k_pipe < kPipe - 1; ++k_pipe) {
      if (k_pipe == 0) {
        produce_ab(k_tile, k_pipe);
      } else {
        produce_ab_no_oob(k_tile, k_pipe);
      }
      cp_async_fence();

      // advance to next k-tile
      if (--k_tiles_remaining > 0) {
        ++k_tile;
      }
    }

    // ###############  Mainloop  ###############
    // pipe index in smem to read from
    int pipe_read = 0;
    // pipe index in smem to write to
    int pipe_write = kPipe - 1;

    // pipe to read from: (CPY, CPY_N, CPY_K)
    Tensor tCsA_p = tCsA(_, _, _, pipe_read);
    Tensor tCsB_p = tCsB(_, _, _, pipe_read);

    // Size of the register pipeline
    auto kBlocks = size<2>(tCrA);

    // prefetch register pipeline
    if (kBlocks > 1) {
      // wait until our first prefetched tile is loaded in
      cp_async_wait<kPipe - 2>();
      __syncthreads();

      // prefetch the first rmem from the first k-tile
      cute::copy(smem_tiled_copy_a, tCsA_p(_, _, _0{}), tCrA_cpv(_, _, _0{}));
      cute::copy(smem_tiled_copy_b, tCsB_p(_, _, _0{}), tCrB_cpv(_, _, _0{}));
    }

    CUTE_NO_UNROLL
    while (k_tiles_remaining > -(kPipe - 1)) {
      CUTE_UNROLL
      for (int ki = 0; ki < kBlocks; ++ki) {
        // first block
        if (ki == 0) {
          // copy gmem to smem for next pipe
          produce_ab_no_oob(k_tile, pipe_write);
          cp_async_fence();

          // advance to next k-tile
          if (--k_tiles_remaining > 0) {
            ++k_tile;
          }
        }
        // last block
        if (ki == kBlocks - 1) {
          // advance to next pipe
          pipe_write = pipe_read;
          pipe_read = (pipe_read == kPipe - 1) ? 0 : pipe_read + 1;

          // advance to next pipe to read from
          tCsA_p = tCsA(_, _, _, pipe_read);
          tCsB_p = tCsB(_, _, _, pipe_read);

          // wait until our next prefetched tile is loaded in
          cp_async_wait<kPipe - 2>();
          __syncthreads();
        }

        // prefetch for next ki
        auto ki_next = (ki + _1{}) % kBlocks;
        copy(smem_tiled_copy_a, tCsA_p(_, _, ki_next), tCrA_cpv(_, _, ki_next));
        copy(smem_tiled_copy_b, tCsB_p(_, _, ki_next), tCrB_cpv(_, _, ki_next));

        // thread-level gemm for ki
        gemm(tiled_mma, tCrA(_, _, ki), tCrB(_, _, ki), tCrAccC);
      }
    }
  }
};

}  // namespace llm
