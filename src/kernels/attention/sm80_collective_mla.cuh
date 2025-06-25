#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cute/config.hpp>
#include <cute/container/array_aligned.hpp>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>

#include "fast_cast.cuh"
#include "layout_convertor.h"
#include "mask.h"
#include "safe_copy.h"

namespace llm {

using namespace cute;

template <int Stages,
          class TileShape_,
          class Element_,
          int HeadDim_,
          int RopeHeadDim_>
struct Sm80CollectiveMla {
  using TileShape = TileShape_;
  using Element = Element_;
  using ElementAccum = float;

  static constexpr int kHeadDim = HeadDim_;
  static constexpr int kRopeHeadDim = RopeHeadDim_;
  static constexpr int kBlockM = get<0>(TileShape{});
  static constexpr int kBlockN = get<1>(TileShape{});
  static constexpr int kBlockK = get<2>(TileShape{});
  static constexpr int kStages = Stages;

  static_assert(kHeadDim % 64 == 0);
  static_assert(kRopeHeadDim % 64 == 0);

  static_assert(kBlockM % 64 == 0);
  static_assert(kBlockN % 16 == 0);
  static_assert(kBlockK % 64 == 0);
  static_assert(kStages == 1 || kStages == 2);

  // number of steps per stage
  static_assert(kHeadDim % kBlockK == 0);
  static constexpr int kSteps = kHeadDim / kBlockK;

  using BLK_M = Int<kBlockM>;
  using BLK_N = Int<kBlockN>;
  using BLK_K = Int<kBlockK>;
  using HEAD_DIM = Int<kHeadDim>;
  using ROPE_HEAD_DIM = Int<kRopeHeadDim>;
  using STEPS = Int<kSteps>;
  using STAGES = Int<kStages>;

  // TiledMMA (64x16x16) for gemm-I and gemm-II
  using MMA_Atom_ =
      std::conditional_t<std::is_same_v<Element, cute::half_t>,
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

  static constexpr int kRowsPerMMA = 2;
  static constexpr int kMmaThreads =
      max(size(TiledMma_QK{}), size(TiledMma_PV{}));

  // Shared memory LayoutAtom for differnt block sizes
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

  // SMEM layout for Q/K/V/P
  // Q smem: (BLK_M, BLK_K, k)
  using SmemLayoutQ =
      decltype(tile_to_shape(SmemLayoutAtomK{}, Shape<BLK_M, BLK_K, STEPS>{}));

  // KV smem: (BLK_N, BLK_K, k, STAGES)
  using SmemLayoutKV =
      decltype(tile_to_shape(SmemLayoutAtomK{},
                             Shape<BLK_N, BLK_K, STEPS, STAGES>{}));

  // P smem: (BLK_M, BLK_N)
  using SmemLayoutP =
      decltype(tile_to_shape(SmemLayoutAtomN{}, Shape<BLK_M, BLK_N>{}));

  // V^T smem: (BLK_K, BLK_N, k, STAGES)
  using SmemLayoutVt = decltype(select<1, 0, 2, 3>(SmemLayoutKV{}));

  // QRope smem: (BLK_M, ROPE_HEAD_DIM)
  using SmemLayoutQRope =
      decltype(tile_to_shape(SmemLayoutAtomR{}, Shape<BLK_M, ROPE_HEAD_DIM>{}));

  // KRoep smem: (BLK_N, ROPE_HEAD_DIM, STAGES)
  using SmemLayoutKRope =
      decltype(tile_to_shape(SmemLayoutAtomR{},
                             Shape<BLK_N, ROPE_HEAD_DIM, STAGES>{}));

  // Shared memory for reduce between warps
  // rowmax/rowsum smem: (_BLK_M, _2)
  using SmemLayoutRowmax = Layout<Shape<Int<2 * kBlockM>>>;
  using SmemLayoutRowsum = Layout<Shape<Int<2 * kBlockM>>>;

  // Tiled copy for differnt block sizes
  using GmemTiledCopy_32x64_ = decltype(make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, Element>{},
      Layout<Shape<_32, _8>, Stride<_8, _1>>{},  // Thr layout: (_32, _8)
      Layout<Shape<_1, _8>>{}                    // Val layout: 8 vals per read
      ));
  using GmemTiledCopy_16x128_ = decltype(make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, Element>{},
      Layout<Shape<_16, _16>, Stride<_16, _1>>{},  // Thr layout: (_16, _16)
      Layout<Shape<_1, _8>>{}  // Val layout: 8 vals per read
      ));
  using GmemTiledCopy_16x64_ = decltype(make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint64_t>, Element>{},
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
      decltype(make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, Element>{},
                                 TiledMma_QK{}));

  using Copy_Atom_K_ =
      std::conditional_t<kBlockN % 32 == 0,
                         Copy_Atom<SM75_U32x4_LDSM_N, Element>,
                         Copy_Atom<SM75_U32x2_LDSM_N, Element>>;
  // Smem tiled copy for KV, 2 warps nxk: 32x16 or 16x16
  using SmemTiledCopyK =
      decltype(make_tiled_copy_B(Copy_Atom_K_{}, TiledMma_QK{}));

  // r2s tiled copy for gemm-I S
  // use 128-bit vectorizing copy
  using VectorizingCopy = AutoVectorizingCopyWithAssumedAlignment<128>;

  using SmemTiledCopyS =
      decltype(make_tiled_copy_C(Copy_Atom<VectorizingCopy, Element>{},
                                 TiledMma_QK{}));

  // s2r tiled copy for gemm-II: O = P*V^T
  // warp layout: 4x2x1, TiledMma mxnxk: 64x32x16
  // Smem tiled copy for P, 4 warps mxk: 64x16
  using SmemTiledCopyP =
      decltype(make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, Element>{},
                                 TiledMma_PV{}));

  // Smem tiled copy for V^T, 2 warps nxk: 32x16
  using SmemTiledCopyVt =
      decltype(make_tiled_copy_B(Copy_Atom<SM75_U16x8_LDSM_T, Element>{},
                                 TiledMma_PV{}));

  struct SharedStorage : cute::aligned_struct<128> {
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> q_smem;
    union {
      cute::array_aligned<Element, cute::cosize_v<SmemLayoutKV>> kv_smem;
      cute::array_aligned<Element, cute::cosize_v<SmemLayoutVt>> vt_smem;
    };
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutP>> p_smem;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutQRope>> q_rope_smem;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutKRope>> k_rope_smem;
    union {
      cute::array_aligned<float, cute::cosize_v<SmemLayoutRowmax>> row_max_smem;
      cute::array_aligned<float, cute::cosize_v<SmemLayoutRowsum>> row_sum_smem;
    };
  };

  // Host side arguments
  struct Arguments {
    FastDivmod group_size;
  };

  // Device side params
  using Params = Arguments;

  // Convert host side arguments to device side params
  static Params to_underlying_arguments(Arguments const& args) {
    // no convertion needed.
    return args;
  }

  // returns false if the block has been skipped
  template <class TensorQ,
            class TensorCQ,
            class TensorKV,
            class TensorCKV,
            class TensorQR,
            class TensorCQR,
            class TensorKR,
            class TensorCKR,
            class TensorCMN,
            class FrgTensor,
            class Softmax,
            class BlockCoord,
            class ResidueMNK,
            class RopeResidueMNK>
  CUTE_DEVICE void operator()(
      const Params& params,
      const TensorQ& gQ,         // (BLK_M, BLK_K, k)
      const TensorCQ& cQ,        // (BLK_M, BLK_K, k) => (M, K)
      const TensorKV& gKV,       // (BLK_N, BLK_K, n, k)
      const TensorCKV& cKV,      // (BLK_N, BLK_K, n, k) => (N, K)
      const TensorQR& gQ_rope,   // (BLK_M, ROPE_HEAD_DIM)
      const TensorCQR& cQ_rope,  // (BLK_M, ROPE_HEAD_DIM) =>(M, K)
      const TensorKR& gK_rope,   // (BLK_N, HEAD_DIM, n)
      const TensorCKR& cK_rope,  // (BLK_N, HEAD_DIM, n) => (N, K)
      const TensorCMN& cMN,      // (BLK_M, BLK_N, n) => (M, N)
      FrgTensor& tOrO,           // (BLK_N, ROPE_HEAD_DIM, n)
      Softmax& softmax,
      int tidx,
      const BlockCoord& blk_coord,
      const ResidueMNK& residue_mnk,
      const RopeResidueMNK& rope_residue_mnk,
      char* smem) {
    static_assert(is_rmem<FrgTensor>::value,
                  "Accum tensor must be rmem resident.");
    static_assert(is_gmem<TensorQ>::value, "Q tensor must be gmem resident.");
    static_assert(is_gmem<TensorKV>::value, "KV tensor must be gmem resident.");
    static_assert(is_gmem<TensorQR>::value,
                  "Q_Rope tensor must be gmem resident.");
    static_assert(is_gmem<TensorKR>::value,
                  "K_Rope tensor must be gmem resident.");

    static constexpr int kBlockM = get<0>(TileShape{});
    static constexpr int kBlockN = get<1>(TileShape{});
    static constexpr int kBlockK = get<2>(TileShape{});

    const int q_idx = get<0>(blk_coord);
    const int q_packed_len = get<0>(residue_mnk);
    const int kv_len = get<1>(residue_mnk);

    const auto& group_size = params.group_size;
    const int q_len = q_packed_len / group_size;

    // Construct shared memory tiles
    auto& ss = *reinterpret_cast<SharedStorage*>(smem);

    // (BLK_M, BLK_K, k), k-major
    Tensor sQ = make_tensor(make_smem_ptr(ss.q_smem.data()), SmemLayoutQ{});
    // (BLK_N, BLK_K, k, STAGES), k-major
    Tensor sKV = make_tensor(make_smem_ptr(ss.kv_smem.data()), SmemLayoutKV{});

    // (BLK_M, BLK_N), k-major
    Tensor sP = make_tensor(make_smem_ptr(ss.p_smem.data()), SmemLayoutP{});

    // (BLK_M, ROPE_HEAD_DIM), k-major
    Tensor sQ_rope =
        make_tensor(make_smem_ptr(ss.q_rope_smem.data()), SmemLayoutQRope{});
    // (BLK_N, ROPE_HEAD_DIM, STAGES), k-major
    Tensor sK_rope =
        make_tensor(make_smem_ptr(ss.k_rope_smem.data()), SmemLayoutKRope{});

    // Tensor for V^t; used in GEMM-II.
    // (BLK_K, BLK_N, k, STAGES)
    Tensor sVt = make_tensor(make_smem_ptr(ss.vt_smem.data()), SmemLayoutVt{});

    // (BLK_M, 2)
    Tensor sRowmax =
        make_tensor(make_smem_ptr(ss.row_max_smem.data()), SmemLayoutRowmax{});
    Tensor sRowsum =
        make_tensor(make_smem_ptr(ss.row_max_smem.data()), SmemLayoutRowsum{});

    // reduce rowmax/rowsum accross 2 warps via shared memory
    // thread layout: (32, (4, 2)), each thread process 2 rows
    // (store_idx, load_idx) = (0, 64) or (1, 65), ...
    const int row_store_idx = tidx / 4 * 2;
    const int row_load_idx = row_store_idx ^ kBlockM;
    auto reduce_rowmax = [&](auto& row_max) {
      CUTE_UNROLL
      for (int i = 0; i < size(row_max); ++i) {
        sRowmax(row_store_idx + i) = row_max(i);
      }
      __syncthreads();
      CUTE_UNROLL
      for (int i = 0; i < size(row_max); ++i) {
        row_max(i) = max(row_max(i), sRowmax(row_load_idx + i));
      }
    };
    auto reduce_rowsum = [&](auto& row_sum) {
      CUTE_UNROLL
      for (int i = 0; i < size(row_sum); ++i) {
        sRowsum(row_store_idx + i) = row_sum(i);
      }
      __syncthreads();
      CUTE_UNROLL
      for (int i = 0; i < size(row_sum); ++i) {
        row_sum(i) += sRowsum(row_load_idx + i);
      }
    };

    // g2s tiled copy for q
    GmemTiledCopyQ gmem_tiled_copy_Q;
    auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_slice(tidx);

    // coordinate tensor for oob handling
    // (CPY, CPY_M, CPY_K, k) => (M, K)
    Tensor tGcQ = gmem_thr_copy_Q.partition_S(cQ);
    // (CPY, CPY_M, CPY_K, k)
    Tensor tGgQ = gmem_thr_copy_Q.partition_S(gQ);
    Tensor tGsQ = gmem_thr_copy_Q.partition_D(sQ);
    const auto residue_mk = select<0, 2>(residue_mnk);

    auto produce_q = [&](int ki) {
      safe_copy</*EVEN_MN=*/false,
                /*EVEN_K=*/true,
                /*ZFILL_MN=*/true,
                /*ZFILL_K=*/true>(gmem_tiled_copy_Q,
                                  tGgQ(_, _, _, ki),
                                  tGsQ(_, _, _, ki),
                                  tGcQ(_, _, _, ki),
                                  residue_mk);
    };

    // g2s tiled copy for q_rope
    GmemTiledCopyQRope gmem_tiled_copy_Q_rope;
    auto gmem_thr_copy_Q_rope = gmem_tiled_copy_Q_rope.get_slice(tidx);

    // (CPY, CPY_M, CPY_K) => (blk_m, rope_head_dim)
    Tensor tGcQ_rope = gmem_thr_copy_Q_rope.partition_S(cQ_rope);
    Tensor tGgQ_rope = gmem_thr_copy_Q_rope.partition_S(gQ_rope);
    Tensor tGsQ_rope = gmem_thr_copy_Q_rope.partition_D(sQ_rope);
    const auto rope_residue_mk = select<0, 2>(rope_residue_mnk);

    auto produce_q_rope = [&]() {
      safe_copy</*EVEN_MN=*/false,
                /*EVEN_K=*/true,
                /*ZFILL_MN=*/true,
                /*ZFILL_K=*/true>(gmem_tiled_copy_Q_rope,
                                  tGgQ_rope,
                                  tGsQ_rope,
                                  tGcQ_rope,
                                  rope_residue_mk);
    };

    // g2s tiled copy for kv
    GmemTiledCopyKV gmem_tiled_copy_KV;
    auto gmem_thr_copy_KV = gmem_tiled_copy_KV.get_slice(tidx);

    // (CPY, CPY_M, CPY_K, n, k) => (N, K)
    Tensor tGcKV = gmem_thr_copy_KV.partition_S(cKV);
    // (CPY, CPY_M, CPY_K, n, k)
    auto tGgKV = gmem_thr_copy_KV.partition_S(gKV);
    // (CPY, CPY_M, CPY_K, k, STAGES)
    auto tGsKV = gmem_thr_copy_KV.partition_D(sKV);
    const auto residue_nk = select<1, 2>(residue_mnk);

    auto produce_kv = [&](int ni, int ki, int stage) {
      safe_copy</*EVEN_MN=*/false,
                /*EVEN_K=*/true,
                /*ZFILL_MN=*/true,
                /*ZFILL_K=*/false>(gmem_tiled_copy_KV,
                                   tGgKV(_, _, _, ni, ki),
                                   tGsKV(_, _, _, ki, stage),
                                   tGcKV(_, _, _, ni, ki),
                                   residue_nk);
    };

    auto produce_kv_no_oob = [&](int ni, int ki, int stage) {
      cute::copy(gmem_tiled_copy_KV,
                 tGgKV(_, _, _, ni, ki),
                 tGsKV(_, _, _, ki, stage));
    };

    // g2s tiled copy for k_rope
    GmemTiledCopyKRope gmem_tiled_copy_K_rope;
    auto gmem_thr_copy_K_rope = gmem_tiled_copy_K_rope.get_slice(tidx);

    // (CPY, CPY_M, CPY_K, n) => (N, K)
    Tensor tGcK_rope = gmem_thr_copy_K_rope.partition_S(cK_rope);
    // (CPY, CPY_M, CPY_K, n)
    Tensor tGgK_rope = gmem_thr_copy_K_rope.partition_S(gK_rope);
    // (CPY, CPY_M, CPY_K, STAGES)
    Tensor tGsK_rope = gmem_thr_copy_K_rope.partition_D(sK_rope);
    const auto rope_residue_nk = select<1, 2>(rope_residue_mnk);

    auto produce_k_rope = [&](int ni, int stage) {
      safe_copy</*EVEN_MN=*/false,
                /*EVEN_K=*/true,
                /*ZFILL_MN=*/true,
                /*ZFILL_K=*/false>(gmem_tiled_copy_K_rope,
                                   tGgK_rope(_, _, _, ni),
                                   tGsK_rope(_, _, _, stage),
                                   tGcK_rope(_, _, _, ni),
                                   rope_residue_nk);
    };

    auto produce_k_rope_no_oob = [&](int ni, int stage) {
      cute::copy(gmem_tiled_copy_K_rope,
                 tGgK_rope(_, _, _, ni),
                 tGsK_rope(_, _, _, stage));
    };

    // GEMM-I: S = Q@K.T
    TiledMma_QK tiled_mma_qk;
    auto thr_mma_qk = tiled_mma_qk.get_slice(tidx);
    // sQ: (BLK_M, BLK_K, k)
    auto tSrQ = thr_mma_qk.partition_fragment_A(sQ(_, _, _0{}));
    // sKV: (BLK_N, BLK_K, k, STAGES)
    auto tSrK = thr_mma_qk.partition_fragment_B(sKV(_, _, _0{}, _0{}));

    // s2r tiled copy for q/q_rope
    SmemTiledCopyQ smem_tiled_copy_Q;
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_slice(tidx);
    // (CPY, CPY_M, CPY_K, k)
    auto tCsQ = smem_thr_copy_Q.partition_S(sQ);
    // (CPY, CPY_M, CPY_K)
    auto tCrQ = smem_thr_copy_Q.retile_D(tSrQ);

    // s2r tiled copy for k/k_rope
    SmemTiledCopyK smem_tiled_copy_K;
    auto smem_thr_copy_K = smem_tiled_copy_K.get_slice(tidx);
    // (CPY, CPY_N, CPY_K, k, STAGES)
    auto tCsK = smem_thr_copy_K.partition_S(sKV);
    // (CPY, CPY_N, CPY_K)
    auto tCrK = smem_thr_copy_K.retile_D(tSrK);

    // S = Q@K.T
    // tSrS: (MMA,MMA_M,MMA_N)
    auto compute_qk = [&](auto& tSrS, int step, int stage) {
      // tCsQ: (CPY, CPY_M, CPY_K, k)
      auto tCsQ_s = tCsQ(_, _, _, step);
      // TCsK: (CPY, CPY_N, CPY_K, k, STAGES)
      auto tCsK_s = tCsK(_, _, _, step, stage);
      // prefetch kv
      cute::copy(smem_tiled_copy_Q, tCsQ_s(_, _, _0{}), tCrQ(_, _, _0{}));
      cute::copy(smem_tiled_copy_K, tCsK_s(_, _, _0{}), tCrK(_, _, _0{}));

      CUTE_UNROLL
      for (int k = 0; k < size<2>(tCsQ_s); ++k) {
        // prefetch next kv
        if (k != size<2>(tCsQ_s) - 1) {
          const auto next_k = k + 1;
          cute::copy(
              smem_tiled_copy_Q, tCsQ_s(_, _, next_k), tCrQ(_, _, next_k));
          cute::copy(
              smem_tiled_copy_K, tCsK_s(_, _, next_k), tCrK(_, _, next_k));
        }
        cute::gemm(tiled_mma_qk, tSrQ(_, _, k), tSrK(_, _, k), tSrS);
      }
    };

    // sQ_rope: (BLK_N, ROPE_HEAD_DIM)
    auto tSrQ_rope = thr_mma_qk.partition_fragment_A(sQ_rope);
    // sK_rope: (BLK_N, ROPE_HEAD_DIM, STAGES)
    auto tSrK_rope = thr_mma_qk.partition_fragment_B(sK_rope(_, _, _0{}));
    // (CPY, CPY_M, CPY_K)
    auto tCsQ_rope = smem_thr_copy_Q.partition_S(sQ_rope);
    // (CPY, CPY_M, CPY_K)
    auto tCrQ_rope = smem_thr_copy_Q.retile_D(tSrQ_rope);
    // (CPY, CPY_N, CPY_K, STAGES)
    auto tCsK_rope = smem_thr_copy_K.partition_S(sK_rope);
    // (CPY, CPY_N, CPY_K)
    auto tCrK_rope = smem_thr_copy_K.retile_D(tSrK_rope);
    auto compute_qk_rope = [&](auto& tSrS, int stage) {
      auto tCsK_rope_s = tCsK_rope(_, _, _, stage);
      cute::copy(
          smem_tiled_copy_Q, tCsQ_rope(_, _, _0{}), tCrQ_rope(_, _, _0{}));
      cute::copy(
          smem_tiled_copy_K, tCsK_rope_s(_, _, _0{}), tCrK_rope(_, _, _0{}));

      CUTE_UNROLL
      for (int k = 0; k < size<2>(tCsQ_rope); ++k) {
        if (k != size<2>(tCsQ_rope) - 1) {
          const auto next_k = k + 1;
          cute::copy(smem_tiled_copy_Q,
                     tCsQ_rope(_, _, next_k),
                     tCrQ_rope(_, _, next_k));
          cute::copy(smem_tiled_copy_K,
                     tCsK_rope_s(_, _, next_k),
                     tCrK_rope(_, _, next_k));
        }
        cute::gemm(tiled_mma_qk, tSrQ_rope(_, _, k), tSrK_rope(_, _, k), tSrS);
      }
    };

    // GEMM-II: O = softmax(S)@V
    TiledMma_PV tiled_mma_pv;
    auto thr_mma_pv = tiled_mma_pv.get_slice(tidx);
    // sP: (BLK_M, BLK_N)
    auto tOrP = thr_mma_pv.partition_fragment_A(sP);
    // sVt: (BLK_K, BLK_N, k, STAGES)
    auto tOrVt = thr_mma_pv.partition_fragment_B(sVt(_, _, _0{}, _0{}));

    // s2r tiled copy for p
    SmemTiledCopyP smem_tiled_copy_P;
    auto smem_thr_copy_P = smem_tiled_copy_P.get_slice(tidx);
    // (CPY, CPY_M, CPY_K)
    auto tCsP = smem_thr_copy_P.partition_S(sP);
    // (CPY, CPY_M, CPY_K)
    auto tCrP = smem_thr_copy_P.retile_D(tOrP);

    // s2r tiled copy for vt
    SmemTiledCopyVt smem_tiled_copy_Vt;
    auto smem_thr_copy_Vt = smem_tiled_copy_Vt.get_slice(tidx);
    // (CPY, CPY_N, CPY_K, k, STAGES)
    auto tCsVt = smem_thr_copy_Vt.partition_S(sVt);
    // (CPY, CPY_N, CPY_K)
    auto tCrVt = smem_thr_copy_Vt.retile_D(tOrVt);

    // O = P*V = softmax(S)*V
    // tOrO: (MMA,MMA_M,MMA_K,k)
    auto compute_pv = [&](auto& tOrO, int step, int stage) {
      auto tOrO_s = tOrO(_, _, _, step);
      auto tCsVt_s = tCsVt(_, _, _, step, stage);

      cute::copy(smem_tiled_copy_P, tCsP(_, _, _0{}), tCrP(_, _, _0{}));
      cute::copy(smem_tiled_copy_Vt, tCsVt_s(_, _, _0{}), tCrVt(_, _, _0{}));

      CUTE_UNROLL
      for (int k = 0; k < size<2>(tCsVt_s); ++k) {
        if (k != size<2>(tCsVt_s) - 1) {
          const auto next_k = k + 1;
          cute::copy(smem_tiled_copy_P, tCsP(_, _, next_k), tCrP(_, _, next_k));
          cute::copy(
              smem_tiled_copy_Vt, tCsVt_s(_, _, next_k), tCrVt(_, _, next_k));
        }
        cute::gemm(tiled_mma_pv, tCrP(_, _, k), tOrVt(_, _, k), tOrO_s);
      }
    };

    // r2s tiled copy for S/P
    SmemTiledCopyS smem_tiled_copy_S;
    auto smem_thr_copy_S = smem_tiled_copy_S.get_slice(tidx);
    auto store_s_to_smem = [&](const auto& tSrS) {
      // cast Accumulator to Element type
      auto tSrS_ = make_tensor_like<Element>(tSrS);
      fast_cast(tSrS, tSrS_);
      // copy scores from rmem to smem
      auto tCrS = smem_thr_copy_S.retile_S(tSrS_);
      auto tCsS = smem_thr_copy_S.partition_D(sP);
      cute::copy(smem_tiled_copy_S, tCrS, tCsS);
    };

    // (MMA,MMA_M,MMA_K,k) => ((2, MMA_M), (2, MMA_K), k)
    auto tOrO_mn =
        make_tensor(tOrO.data(), LayoutConvertor::to_mn(tOrO.layout()));

    const int n_block_min = 0;
    // process kv in range: [0, kv_idx_max)
    const int diagonal = q_idx + kv_len - q_len;
    const int kv_idx_max = std::min(kv_len, diagonal + kBlockM);
    const int n_block_max = cute::ceil_div(kv_idx_max, kBlockN);

    if (n_block_min >= n_block_max) {
      return;
    }

    // ###############  Prologue  ###############
    // g2s async data copy pipelines
    // |  stage  |                  queue                            |
    // |   1     | [k_r, kv0, kv1]                                   |
    // |   2     | [k_r, kv0, kv1, (nop, nop, nop, k_r, kv0, kv1)]   |
    //                  ^ kWait = (kSteps + 1) * (2*kStages - 1) - 1
    constexpr int kWait = (kSteps + 1) * (2 * kStages - 1) - 1;
    // produce q_rope/q
    produce_q_rope();
    CUTE_UNROLL
    for (int ki = 0; ki < kSteps; ++ki) {
      produce_q(ki);
    }
    // produce k_rope/kv
    CUTE_UNROLL
    for (int stage = 0; stage < kStages; ++stage) {
      const int ni = n_block_max - 1 - stage;
      // insert nops between stages for a perfect pipeline
      if (stage != 0) {
        cp_async_fence();
        CUTE_UNROLL
        for (int ki = 0; ki < kSteps; ++ki) {
          cp_async_fence();
        }
      }
      // handle oob kv
      if (ni >= n_block_min) {
        stage == 0 ? produce_k_rope(ni, stage)
                   : produce_k_rope_no_oob(ni, stage);
        cp_async_fence();
        CUTE_UNROLL
        for (int ki = 0; ki < kSteps; ++ki) {
          stage == 0 ? produce_kv(ni, ki, stage)
                     : produce_kv_no_oob(ni, ki, stage);
          cp_async_fence();
        }
      } else {
        cp_async_fence();
        CUTE_UNROLL
        for (int ki = 0; ki < kSteps; ++ki) {
          cp_async_fence();
        }
      }
    }

    // ###############  Mainloop  ###############
    // attention score accumulator, (MMA, MMA_M, MMA_N)
    auto tSrS = partition_fragment_C(tiled_mma_qk, Shape<BLK_M, BLK_N>{});
    // ((2, MMA_M), (2, MMA_N))
    auto tSrS_mn =
        make_tensor(tSrS.data(), LayoutConvertor::to_mn(tSrS.layout()));

    // (MMA, MMA_M, MMA_N, n) => (M, N)
    auto tScMN = thr_mma_qk.partition_C(cMN);
    // ((2, MMA_M), (2, MMA_N), n) => (M, N)
    auto tScMN_mn =
        make_tensor(tScMN.data(), LayoutConvertor::to_mn(tScMN.layout()));

    constexpr int kRowsPerThr = kRowsPerMMA * size<1>(tSrS);
    using Mask = Mask<kRowsPerThr, /*ALIBI=*/false, /*LOCAL=*/false>;
    Mask mask(q_len, kv_len, group_size, /*sliding_window=*/kv_len);

    constexpr int n_oob_mask = cute::ceil_div(kBlockM, kBlockN) + 1;
    const int n_blocks = n_block_max - n_block_min;
    int stage = 0;
    CUTE_NO_UNROLL
    for (int i = 0; i < n_blocks; ++i) {
      const int ni = n_block_max - 1 - i;
      clear(tSrS);

      // ((2, MMA_M), (2, MMA_N)) => (M, N)
      const auto tScS_mn = tScMN_mn(_, _, ni);

      cp_async_wait<kWait>();
      __syncthreads();

      // 1> S = Q_rope@K_rope.T
      compute_qk_rope(tSrS, stage);
      cp_async_fence();

      // 2> S += Q@K.T
      CUTE_UNROLL
      for (int ki = 0; ki < kSteps; ++ki) {
        cp_async_wait<kWait>();
        __syncthreads();

        compute_qk(tSrS, ki, stage);
        cp_async_fence();
      }

      // apply mask
      if (i < n_oob_mask) {
        mask.apply</*OOB_MASK=*/true>(tSrS_mn, tScS_mn);
      } else {
        mask.apply</*OOB_MASK=*/false>(tSrS_mn, tScS_mn);
      }

      softmax.rescale(tSrS_mn, tOrO_mn, reduce_rowmax);

      // save tSrS from rmem to smem
      store_s_to_smem(tSrS);
      __syncthreads();

      // 3> O = softmax(S)*V
      const auto next_ni = ni - kStages;
      if (next_ni >= n_block_min) {
        produce_k_rope_no_oob(next_ni, stage);
        cp_async_fence();

        CUTE_UNROLL
        for (int ki = 0; ki < kSteps; ++ki) {
          compute_pv(tOrO, ki, stage);
          __syncthreads();
          produce_kv_no_oob(next_ni, ki, stage);
          cp_async_fence();
        }
      } else {
        cp_async_fence();
        CUTE_UNROLL
        for (int ki = 0; ki < kSteps; ++ki) {
          compute_pv(tOrO, ki, stage);
          cp_async_fence();
        }
      }

      // move to next stage
      if constexpr (kStages == 1) {
        // do nothing
      } else if constexpr (kStages == 2) {
        stage = stage ^ 1;
      } else {
        stage = (stage + 1) % kStages;
      }
    }
    // normalize output: o /= rowsum
    softmax.finalize(tOrO_mn, reduce_rowsum);
  }
};

}  // namespace llm
