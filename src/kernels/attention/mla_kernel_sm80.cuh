#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cute/layout.hpp>
#include <cute/tensor.hpp>

#include "cute/config.hpp"
#include "cute_extensions.cuh"
#include "fast_cast.cuh"
#include "mask.h"
#include "mla_tile.h"
#include "online_softmax.cuh"

namespace llm {

template <typename Traits>
struct MLASharedStorage {
  using DType = typename Traits::DType;
  using SmemLayoutQ = typename Traits::SmemLayoutQ;
  using SmemLayoutKV = typename Traits::SmemLayoutKV;
  using SmemLayoutP = typename Traits::SmemLayoutP;
  using SmemLayoutQRope = typename Traits::SmemLayoutQRope;
  using SmemLayoutKRope = typename Traits::SmemLayoutKRope;
  using SmemLayoutVt = typename Traits::SmemLayoutVt;
  using SmemLayoutO = typename Traits::SmemLayoutO;
  using SmemLayoutRowmax = typename Traits::SmemLayoutRowmax;
  using SmemLayoutRowsum = typename Traits::SmemLayoutRowsum;

  union {
    struct {
      cute::array_aligned<DType, cute::cosize_v<SmemLayoutQ>> q_smem;
      union {
        cute::array_aligned<DType, cute::cosize_v<SmemLayoutKV>> kv_smem;
        cute::array_aligned<DType, cute::cosize_v<SmemLayoutVt>> vt_smem;
      };
      cute::array_aligned<DType, cute::cosize_v<SmemLayoutP>> p_smem;
      cute::array_aligned<DType, cute::cosize_v<SmemLayoutQRope>> q_rope_smem;
      cute::array_aligned<DType, cute::cosize_v<SmemLayoutKRope>> k_rope_smem;
      union {
        cute::array_aligned<float, cute::cosize_v<SmemLayoutRowmax>>
            row_max_smem;
        cute::array_aligned<float, cute::cosize_v<SmemLayoutRowsum>>
            row_sum_smem;
      };
    };

    cute::array_aligned<DType, cute::cosize_v<SmemLayoutO>> o_smem;
  };
};

template <typename Traits, typename Params>
__global__ __launch_bounds__(Traits::kThreadNum) void mla_kernel_sm80(
    __grid_constant__ const Params params) {
  using namespace cute;

  constexpr int kBlockM = Traits::kBlockM;
  constexpr int kBlockN = Traits::kBlockN;
  constexpr int kBlockK = Traits::kBlockK;
  constexpr int kHeadDim = Traits::kHeadDim;
  constexpr int kSteps = Traits::kSteps;
  constexpr int kStages = Traits::kStages;
  constexpr int kRopeHeadDim = Traits::kRopeHeadDim;
  constexpr int kRowsPerMMA = Traits::kRowsPerMMA;

  using _BLK_M = Int<kBlockM>;
  using _BLK_N = Int<kBlockN>;
  using _BLK_K = Int<kBlockK>;
  using _STEPS = Int<kSteps>;
  using _HEAD_DIM = Int<kHeadDim>;
  using _ROPE_HEAD_DIM = Int<kRopeHeadDim>;

  // type alias
  using DType = typename Traits::DType;

  using TiledMma_QK = typename Traits::TiledMma_QK;
  using TiledMma_PV = typename Traits::TiledMma_PV;
  using Layout = typename Traits::LayoutConvertor;

  using SmemLayoutQ = typename Traits::SmemLayoutQ;
  using SmemLayoutKV = typename Traits::SmemLayoutKV;
  using SmemLayoutP = typename Traits::SmemLayoutP;
  using SmemLayoutQRope = typename Traits::SmemLayoutQRope;
  using SmemLayoutKRope = typename Traits::SmemLayoutKRope;
  using SmemLayoutVt = typename Traits::SmemLayoutVt;
  using SmemLayoutO = typename Traits::SmemLayoutO;
  using SmemLayoutRowmax = typename Traits::SmemLayoutRowmax;
  using SmemLayoutRowsum = typename Traits::SmemLayoutRowsum;
  using SharedStorage = MLASharedStorage<Traits>;

  using GmemTiledCopyQ = typename Traits::GmemTiledCopyQ;
  using GmemTiledCopyQRope = typename Traits::GmemTiledCopyQRope;
  using GmemTiledCopyKV = typename Traits::GmemTiledCopyKV;
  using GmemTiledCopyKRope = typename Traits::GmemTiledCopyKRope;
  using GmemTiledCopyO = typename Traits::GmemTiledCopyO;

  using SmemTiledCopyQ = typename Traits::SmemTiledCopyQ;
  using SmemTiledCopyK = typename Traits::SmemTiledCopyK;
  using SmemTiledCopyS = typename Traits::SmemTiledCopyS;
  using SmemTiledCopyP = typename Traits::SmemTiledCopyP;
  using SmemTiledCopyVt = typename Traits::SmemTiledCopyVt;
  using SmemTiledCopyO = typename Traits::SmemTiledCopyO;

  const int m_block_idx = blockIdx.x;
  const int batch_idx = blockIdx.y;
  const int tidx = threadIdx.x;

  // for MLA/MQA, group_size = n_heads
  const int group_size = params.n_heads;

  MLATile<Params> tile(params);

  // ProblemShape
  // Q/O: (q_packed_len, HEAD_DIM)
  // Q_ROPE: (q_packed_len, ROPE_HEAD_DIM)
  auto [Q, Q_ROPE, O] = tile.template get_qo_tile<DType>(batch_idx);
  // KV: (kv_len, HEAD_DIM)
  // K_ROPE: (kv_len, ROPE_HEAD_DIM)
  auto [KV, K_ROPE] = tile.template get_kv_tile<DType>(batch_idx);

  const int q_packed_len = size<0>(Q);
  const int q_len = q_packed_len / group_size;
  const int kv_len = size<0>(KV);

  if (m_block_idx * kBlockM >= size<0>(Q)) {
    // m out of bound, return
    return;
  }

  // Gmem
  // (BLK_M, BLK_K, STEPS)
  Tensor gQ =
      local_tile(Q, Shape<_BLK_M, _BLK_K>{}, make_coord(m_block_idx, _));
  Tensor gO =
      local_tile(O, Shape<_BLK_M, _BLK_K>{}, make_coord(m_block_idx, _));
  // (BLK_N, BLK_K, n, STEPS)
  Tensor gKV = local_tile(KV, Shape<_BLK_N, _BLK_K>{}, make_coord(_, _));

  // (BLK_M, ROPE_HEAD_DIM)
  Tensor gQ_rope = local_tile(
      Q_ROPE, Shape<_BLK_M, _ROPE_HEAD_DIM>{}, make_coord(m_block_idx, _0{}));
  // (BLK_N, ROPE_HEAD_DIM, n)
  Tensor gK_rope =
      local_tile(K_ROPE, Shape<_BLK_N, _ROPE_HEAD_DIM>{}, make_coord(_, _0{}));

  // Smem
  extern __shared__ char smem[];
  auto& ss = *reinterpret_cast<SharedStorage*>(smem);

  // (BLK_M, BLK_K, STEPS), k-major
  Tensor sQ = make_tensor(make_smem_ptr(ss.q_smem.data()), SmemLayoutQ{});
  // (BLK_N, BLK_K, STEPS, STAGES), k-major
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
  // (BLK_K, BLK_N, STEPS, STAGES)
  Tensor sVt = make_tensor(make_smem_ptr(ss.vt_smem.data()), SmemLayoutVt{});

  // (BLK_M, BLK_K, STEPS), reuse smem
  Tensor sO = make_tensor(make_smem_ptr(ss.o_smem.data()), SmemLayoutO{});

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
  // (BLK_M, BLK_K) -> (blk_m, blk_k)
  Tensor cQ = make_identity_tensor(Shape<_BLK_M, _BLK_K>{});
  Tensor tCcQ = gmem_thr_copy_Q.partition_S(cQ(_, _));
  auto max_coord_Q = make_coord(q_packed_len - m_block_idx * kBlockM, kBlockK);

  auto produce_q = [&](int step) {
    // gQ/sQ: (BLK_M, BLK_K, STEPS)
    auto tCgQ = gmem_thr_copy_Q.partition_S(gQ(_, _, step));
    auto tCsQ = gmem_thr_copy_Q.partition_D(sQ(_, _, step));
    safe_copy</*EVEN_MN=*/false,
              /*EVEN_K=*/true,
              /*ZFILL_MN=*/true,
              /*ZFILL_K=*/true>(
        gmem_tiled_copy_Q, tCgQ, tCsQ, tCcQ, max_coord_Q);
  };

  // g2s tiled copy for q_rope
  GmemTiledCopyQRope gmem_tiled_copy_Q_rope;
  auto gmem_thr_copy_Q_rope = gmem_tiled_copy_Q_rope.get_slice(tidx);

  // (BLK_M, ROPE_HEAD_DIM) -> (blk_m, rope_head_dim)
  Tensor cQ_rope = make_identity_tensor(Shape<_BLK_M, _ROPE_HEAD_DIM>{});
  Tensor tCcQ_rope = gmem_thr_copy_Q_rope.partition_S(cQ_rope);

  auto produce_q_rope = [&]() {
    auto tCgQ_rope = gmem_thr_copy_Q_rope.partition_S(gQ_rope);
    auto tCsQ_rope = gmem_thr_copy_Q_rope.partition_D(sQ_rope);
    auto max_coord =
        make_coord(q_packed_len - m_block_idx * kBlockM, kRopeHeadDim);
    safe_copy</*EVEN_MN=*/false,
              /*EVEN_K=*/true,
              /*ZFILL_MN=*/true,
              /*ZFILL_K=*/true>(
        gmem_tiled_copy_Q_rope, tCgQ_rope, tCsQ_rope, tCcQ_rope, max_coord);
  };

  // g2s tiled copy for kv
  GmemTiledCopyKV gmem_tiled_copy_KV;
  auto gmem_thr_copy_KV = gmem_tiled_copy_KV.get_slice(tidx);

  // (BLK_N, BLK_K, STEPS) -> (blk_n, head_dim)
  Tensor cKV = make_identity_tensor(Shape<_BLK_N, _BLK_K>{});
  Tensor tCcKV = gmem_thr_copy_KV.partition_S(cKV);

  auto produce_kv = [&](int ni, int step, int stage) {
    // gKV: (BLK_N, BLK_K, n, STEPS)
    // sKV: (BLK_N, BLK_K, STEPS, STAGES)
    auto tCgKV = gmem_thr_copy_KV.partition_S(gKV(_, _, ni, step));
    auto tCsKV = gmem_thr_copy_KV.partition_D(sKV(_, _, step, stage));
    auto max_coord = make_coord(kv_len - ni * kBlockN, kBlockK);
    safe_copy</*EVEN_MN=*/false,
              /*EVEN_K=*/true,
              /*ZFILL_MN=*/true,
              /*ZFILL_K=*/false>(
        gmem_tiled_copy_KV, tCgKV, tCsKV, tCcKV, max_coord);
  };

  auto produce_kv_no_oob = [&](int ni, int step, int stage) {
    // gKV: (BLK_N, BLK_K, n, STEPS)
    // sKV: (BLK_N, BLK_K, STEPS, STAGES)
    auto tCgKV = gmem_thr_copy_KV.partition_S(gKV(_, _, ni, step));
    auto tCsKV = gmem_thr_copy_KV.partition_D(sKV(_, _, step, stage));
    cute::copy(gmem_tiled_copy_KV, tCgKV, tCsKV);
  };

  // g2s tiled copy for k_rope
  GmemTiledCopyKRope gmem_tiled_copy_K_rope;
  auto gmem_thr_copy_K_rope = gmem_tiled_copy_K_rope.get_slice(tidx);

  // (BLK_N, ROPE_HEAD_DIM) -> (blk_n, rope_head_dim)
  Tensor cK_rope = make_identity_tensor(Shape<_BLK_N, _ROPE_HEAD_DIM>{});
  Tensor tKcK_rope = gmem_thr_copy_K_rope.partition_S(cK_rope);

  auto produce_k_rope = [&](int ni, int stage) {
    // gK_rope: (BLK_N, ROPE_HEAD_DIM, n)
    // sK_rope: (BLK_N, ROPE_HEAD_DIM, STAGES)
    auto tKgK_rope = gmem_thr_copy_K_rope.partition_S(gK_rope(_, _, ni));
    auto tKsK_rope = gmem_thr_copy_K_rope.partition_D(sK_rope(_, _, stage));
    auto max_coord = make_coord(kv_len - ni * kBlockN, kRopeHeadDim);
    safe_copy</*EVEN_MN=*/false,
              /*EVEN_K=*/true,
              /*ZFILL_MN=*/true,
              /*ZFILL_K=*/false>(
        gmem_tiled_copy_K_rope, tKgK_rope, tKsK_rope, tKcK_rope, max_coord);
  };

  auto produce_k_rope_no_oob = [&](int ni, int stage) {
    // gK_rope: (BLK_N, ROPE_HEAD_DIM, n)
    // sK_rope: (BLK_N, ROPE_HEAD_DIM, STAGES)
    auto tKgK_rope = gmem_thr_copy_K_rope.partition_S(gK_rope(_, _, ni));
    auto tKsK_rope = gmem_thr_copy_K_rope.partition_D(sK_rope(_, _, stage));
    cute::copy(gmem_tiled_copy_K_rope, tKgK_rope, tKsK_rope);
  };

  // GEMM-I: S = Q@K.T
  TiledMma_QK tiled_mma_qk;
  auto thr_mma_qk = tiled_mma_qk.get_slice(tidx);
  // sQ: (BLK_M, BLK_K, STEPS)
  auto tSrQ = thr_mma_qk.partition_fragment_A(sQ(_, _, _0{}));
  // sKV: (BLK_N, BLK_K, STEPS, STAGES)
  auto tSrK = thr_mma_qk.partition_fragment_B(sKV(_, _, _0{}, _0{}));

  // s2r tiled copy for q/q_rope
  SmemTiledCopyQ smem_tiled_copy_Q;
  auto smem_thr_copy_Q = smem_tiled_copy_Q.get_slice(tidx);
  // (CPY, CPY_M, CPY_K, STEPS)
  auto tCsQ = smem_thr_copy_Q.partition_S(sQ);
  // (CPY, CPY_M, CPY_K)
  auto tCrQ = smem_thr_copy_Q.retile_D(tSrQ);

  // s2r tiled copy for k/k_rope
  SmemTiledCopyK smem_tiled_copy_K;
  auto smem_thr_copy_K = smem_tiled_copy_K.get_slice(tidx);
  // (CPY, CPY_N, CPY_K, STEPS, STAGES)
  auto tCsK = smem_thr_copy_K.partition_S(sKV);
  // (CPY, CPY_N, CPY_K)
  auto tCrK = smem_thr_copy_K.retile_D(tSrK);

  // S = Q@K.T
  // tSrS: (MMA,MMA_M,MMA_N)
  auto compute_qk = [&](auto& tSrS, int step, int stage) {
    // tCsQ: (CPY, CPY_M, CPY_K, STEPS)
    auto tCsQ_s = tCsQ(_, _, _, step);
    // TCsK: (CPY, CPY_N, CPY_K, STEPS, STAGES)
    auto tCsK_s = tCsK(_, _, _, step, stage);
    // prefetch kv
    cute::copy(smem_tiled_copy_Q, tCsQ_s(_, _, _0{}), tCrQ(_, _, _0{}));
    cute::copy(smem_tiled_copy_K, tCsK_s(_, _, _0{}), tCrK(_, _, _0{}));

    CUTE_UNROLL
    for (int k = 0; k < size<2>(tCsQ_s); ++k) {
      // prefetch next kv
      if (k != size<2>(tCsQ_s) - 1) {
        const auto next_k = k + 1;
        cute::copy(smem_tiled_copy_Q, tCsQ_s(_, _, next_k), tCrQ(_, _, next_k));
        cute::copy(smem_tiled_copy_K, tCsK_s(_, _, next_k), tCrK(_, _, next_k));
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
    cute::copy(smem_tiled_copy_Q, tCsQ_rope(_, _, _0{}), tCrQ_rope(_, _, _0{}));
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
  // sVt: (BLK_K, BLK_N, STEPS, STAGES)
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
  // (CPY, CPY_N, CPY_K, STEPS, STAGES)
  auto tCsVt = smem_thr_copy_Vt.partition_S(sVt);
  // (CPY, CPY_N, CPY_K)
  auto tCrVt = smem_thr_copy_Vt.retile_D(tOrVt);

  // O = P*V = softmax(S)*V
  // tOrO: (MMA,MMA_M,MMA_K,STEPS)
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
    auto tSrS_ = make_tensor_like<DType>(tSrS);
    fast_cast(tSrS, tSrS_);
    // copy scores from rmem to smem
    auto tCrS = smem_thr_copy_S.retile_S(tSrS_);
    auto tCsS = smem_thr_copy_S.partition_D(sP);
    cute::copy(smem_tiled_copy_S, tCrS, tCsS);
  };

  // tOrO: (MMA,MMA_M,MMA_K,STEPS)
  auto epilogue = [&](const auto& tOrO) {
    // write output to gmem
    // 1. copy output from reg to smem (reuse sQ)
    SmemTiledCopyO smem_tiled_copy_O;
    auto smem_thr_copy_O = smem_tiled_copy_O.get_slice(tidx);
    CUTE_UNROLL
    for (int step = 0; step < kSteps; ++step) {
      auto tOrO_s = tOrO(_, _, _, step);
      auto sO_s = sO(_, _, step);

      // cast Accumulator to Element type
      auto tOrO_ = make_tensor_like<DType>(tOrO_s);
      fast_cast(tOrO_s, tOrO_);

      auto tCrO = smem_thr_copy_O.retile_S(tOrO_);
      auto tCsO = smem_thr_copy_O.partition_D(sO_s);
      cute::copy(smem_tiled_copy_O, tCrO, tCsO);
    }

    __syncthreads();

    // 2. copy output from smem to gmem
    GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx);

    // (BLK_M, BLK_K) -> (blk_m, blk_k)
    auto cO = make_identity_tensor(Shape<_BLK_M, _BLK_K>{});
    auto tCcO = gmem_thr_copy_Q.partition_S(cO);
    auto max_coord_O =
        make_coord(q_packed_len - m_block_idx * kBlockM, kBlockK);

    CUTE_UNROLL
    for (int step = 0; step < kSteps; ++step) {
      auto tCsO = gmem_thr_copy_O.partition_S(sO(_, _, step));
      auto tCgO = gmem_thr_copy_O.partition_D(gO(_, _, step));

      safe_copy</*EVEN_MN=*/false,
                /*EVEN_K=*/true,
                /*ZFILL_MN=*/false,
                /*ZFILL_K=*/false>(
          gmem_tiled_copy_O, tCsO, tCgO, tCcO, max_coord_O);
    }
  };

  // output accumulator: (MMA,MMA_M,MMA_K,STEPS)
  auto tOrO =
      partition_fragment_C(tiled_mma_pv, Shape<_BLK_M, _BLK_K, _STEPS>{});
  auto tOrO_mn = make_tensor(tOrO.data(), Layout::to_mns(tOrO.layout()));
  clear(tOrO);

  const int n_block_min = 0;
  // process kv in range: [0, kv_idx_max)
  const int diagonal = (m_block_idx * kBlockM) / group_size + kv_len - q_len;
  const int kv_idx_max = std::min(kv_len, diagonal + kBlockM);
  const int n_block_max = cute::ceil_div(kv_idx_max, kBlockN);

  if (n_block_min >= n_block_max) {
    // write output to gmem
    epilogue(tOrO);
    return;
  }

  // ###############  Prologue  ###############
  // g2s async data copy pipelines
  // |  stage  |                  queue                            |
  // |   1     | [k_r, kv0, kv1]                                   |
  // |   2     | [k_r, kv0, kv1, (nop, nop, nop, k_r, kv0, kv1)]   |
  // |   3     | [k_r, kv0, kv1, (nop, nop, nop, k_r, kv0, kv1)*2] |
  //                  ^ kWait = (kSteps + 1) * (2*kStages - 1) - 1
  constexpr int kWait = (kSteps + 1) * (2 * kStages - 1) - 1;
  // produce q_rope/q
  produce_q_rope();
  CUTE_UNROLL
  for (int step = 0; step < kSteps; ++step) {
    produce_q(step);
  }
  // produce k_rope/kv
  CUTE_UNROLL
  for (int stage = 0; stage < kStages; ++stage) {
    const int ni = n_block_max - 1 - stage;
    // insert nops between stages for a perfect pipeline
    if (stage != 0) {
      cp_async_fence();
      CUTE_UNROLL
      for (int step = 0; step < kSteps; ++step) {
        cp_async_fence();
      }
    }
    // handle oob kv
    if (ni >= n_block_min) {
      stage == 0 ? produce_k_rope(ni, stage) : produce_k_rope_no_oob(ni, stage);
      cp_async_fence();
      CUTE_UNROLL
      for (int step = 0; step < kSteps; ++step) {
        stage == 0 ? produce_kv(ni, step, stage)
                   : produce_kv_no_oob(ni, step, stage);
        cp_async_fence();
      }
    } else {
      cp_async_fence();
      CUTE_UNROLL
      for (int step = 0; step < kSteps; ++step) {
        cp_async_fence();
      }
    }
  }

  // ###############  Mainloop  ###############
  // attention score accumulator, (MMA,MMA_M,MMA_N)
  auto tSrS = partition_fragment_C(tiled_mma_qk, Shape<_BLK_M, _BLK_N>{});
  auto tScS =
      thr_mma_qk.partition_C(make_identity_tensor(Shape<_BLK_M, _BLK_N>{}));
  auto tSrS_mn = make_tensor(tSrS.data(), Layout::to_mn(tSrS.layout()));
  auto tScS_mn = make_tensor(tScS.data(), Layout::to_mn(tScS.layout()));

  constexpr int kRowsPerThr = kRowsPerMMA * size<1>(tSrS);
  using Softmax = OnlineSoftmax<kRowsPerThr>;
  using Mask = Mask<kRowsPerThr, /*ALIBI=*/false, /*LOCAL=*/false>;

  Softmax softmax(params.sm_scale_log2);
  Mask mask(q_len, kv_len, group_size, /*sliding_window=*/kv_len);

  constexpr int n_oob_mask = cute::ceil_div(kBlockM, kBlockN) + 1;
  const int n_blocks = n_block_max - n_block_min;
  int stage = 0;
  CUTE_NO_UNROLL
  for (int i = 0; i < n_blocks; ++i) {
    const int ni = n_block_max - 1 - i;
    clear(tSrS);

    cp_async_wait<kWait>();
    __syncthreads();

    // 1> S = Q_rope@K_rope.T
    compute_qk_rope(tSrS, stage);
    cp_async_fence();

    // 2> S += Q@K.T
    CUTE_UNROLL
    for (int step = 0; step < kSteps; ++step) {
      cp_async_wait<kWait>();
      __syncthreads();

      compute_qk(tSrS, step, stage);
      cp_async_fence();
    }

    // apply mask
    if (i < n_oob_mask) {
      mask.apply(tSrS_mn, tScS_mn, m_block_idx * kBlockM, ni * kBlockN);
    } else {
      mask.apply</*OOB_MASK=*/false>(
          tSrS_mn, tScS_mn, m_block_idx * kBlockM, ni * kBlockN);
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
      for (int step = 0; step < kSteps; ++step) {
        compute_pv(tOrO, step, stage);
        __syncthreads();
        produce_kv_no_oob(next_ni, step, stage);
        cp_async_fence();
      }
    } else {
      cp_async_fence();
      CUTE_UNROLL
      for (int step = 0; step < kSteps; ++step) {
        compute_pv(tOrO, step, stage);
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

  // ###############  Epilogue  ###############

  // normalize output: o /= rowsum
  softmax.finalize(tOrO_mn, reduce_rowsum);

  // write output to gmem
  epilogue(tOrO);
}

template <typename Traits, typename Params>
void launch_mla_kernel_sm80(const Params& params, cudaStream_t stream) {
  const auto batch_size = params.batch_size;
  const auto max_q_packed_len = params.max_q_len * params.n_heads;

  const auto smem_size = sizeof(MLASharedStorage<Traits>);
  // print("smem_size: %d, %d\n", smem_size, Traits::kSmemSize);

  auto mla_kernel = mla_kernel_sm80<Traits, Params>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      mla_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
  // TODO: support persistent kernels
  dim3 grid(cute::ceil_div(max_q_packed_len, Traits::kBlockM), batch_size, 1);
  dim3 block = Traits::kThreadNum;
  mla_kernel<<<grid, block, smem_size, stream>>>(params);
}

}  // namespace llm