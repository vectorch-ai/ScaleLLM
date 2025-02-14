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
#include "ptx.cuh"

namespace llm {

template <typename Traits,
          typename Params,
          bool EVEN_K,
          bool ALIBI,
          bool SOFT_CAP,
          bool LOCAL>
__global__ __launch_bounds__(Traits::kThreadNum) void mla_kernel_sm80(
    __grid_constant__ const Params params) {
  using namespace cute;

  constexpr int kBlockM = Traits::kBlockM;
  constexpr int kBlockN = Traits::kBlockN;
  constexpr int kBlockK = Traits::kBlockK;
  constexpr int kHeadDim = Traits::kHeadDim;
  constexpr int kStages = Traits::kStages;
  constexpr int kRopeHeadDim = Traits::kRopeHeadDim;
  constexpr int kRowsPerMMA = Traits::kRowsPerMMA;

  using _BLK_M = Int<kBlockM>;
  using _BLK_N = Int<kBlockN>;
  using _BLK_K = Int<kBlockK>;
  using _STAGES = Int<kStages>;
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

  using GmemTiledCopyQ = typename Traits::GmemTiledCopyQ;
  using GmemTiledCopyKV = typename Traits::GmemTiledCopyKV;
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

  MLATile<Params> tile(params);

  // ProblemShape
  // Q/O: (q_packed_len, HEAD_DIM)
  // KV: (kv_len, HEAD_DIM)
  // Q/K_ROPE: (q_packed_len, ROPE_HEAD_DIM)
  auto [Q, Q_ROPE, O] = tile.template get_qo_tile<DType>(batch_idx);
  auto [KV, K_ROPE] = tile.template get_kv_tile<DType>(batch_idx);

  if (m_block_idx * kBlockM >= size<0>(Q)) {
    // m out of bound, return
    return;
  }

  // Gmem
  // (BLK_M, BLK_K, STAGES)
  Tensor gQ =
      local_tile(Q, Shape<_BLK_M, _BLK_K>{}, make_coord(m_block_idx, _));
  Tensor gO =
      local_tile(O, Shape<_BLK_M, _BLK_K>{}, make_coord(m_block_idx, _));
  // (BLK_N, BLK_K, n, STAGES)
  Tensor gKV = local_tile(KV, Shape<_BLK_N, _BLK_K>{}, make_coord(_, _));

  // (BLK_M, ROPE_HEAD_DIM)
  Tensor gQ_rope = local_tile(
      Q_ROPE, Shape<_BLK_M, _ROPE_HEAD_DIM>{}, make_coord(m_block_idx, _0{}));
  // (BLK_N, ROPE_HEAD_DIM, n)
  Tensor gK_rope =
      local_tile(K_ROPE, Shape<_BLK_N, _ROPE_HEAD_DIM>{}, make_coord(_, _0{}));

  // Smem
  extern __shared__ char smem[];
  DType* q_smem = (DType*)smem;
  DType* kv_smem = q_smem + cosize(SmemLayoutQ{});
  DType* p_smem = kv_smem + cosize(SmemLayoutKV{});
  DType* q_rope_smem = p_smem + cosize(SmemLayoutP{});
  DType* k_rope_smem = q_rope_smem + cosize(SmemLayoutQRope{});
  float* row_sync_smem = (float*)(k_rope_smem + cosize(SmemLayoutKRope{}));

  // (BLK_M, BLK_K, STAGES), k-major
  Tensor sQ = make_tensor(make_smem_ptr(q_smem), SmemLayoutQ{});
  // (BLK_N, BLK_K, STAGES), k-major
  Tensor sK = make_tensor(make_smem_ptr(kv_smem), SmemLayoutKV{});

  // (BLK_M, BLK_N), k-major
  Tensor sP = make_tensor(make_smem_ptr(p_smem), SmemLayoutP{});

  // (BLK_M, ROPE_HEAD_DIM), k-major
  Tensor sQ_rope = make_tensor(make_smem_ptr(q_rope_smem), SmemLayoutQRope{});
  // (BLK_N, ROPE_HEAD_DIM), k-major
  Tensor sK_rope = make_tensor(make_smem_ptr(k_rope_smem), SmemLayoutKRope{});

  // Tensor for V^t; used in GEMM-II.
  // (BLK_K, BLK_N, STAGES)
  Tensor sVt = make_tensor(make_smem_ptr(kv_smem), SmemLayoutVt{});

  // (BLK_M, BLK_K, STAGES), reuse smem
  Tensor sO = make_tensor(make_smem_ptr(q_smem), SmemLayoutO{});

  // (BLK_M, 2)
  Tensor sRowmax =
      make_tensor(make_smem_ptr(row_sync_smem), SmemLayoutRowmax{});
  Tensor sRowsum =
      make_tensor(make_smem_ptr(row_sync_smem), SmemLayoutRowsum{});

  // thread layout: (32, 8), each thread process 2 rows
  // (store_idx, load_idx) = (0, 64), (1, 65), ...
  const int row_store_idx = tidx / 4 * 2;
  const int row_load_idx = row_store_idx ^ kBlockM;
  // reduce rowmax accross 2 warps
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

  // reduce rowsum accross 2 warps
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

  // Tiled Copy
  // g2s tiled copy for qkv
  GmemTiledCopyQ gmem_tiled_copy_Q;
  GmemTiledCopyKV gmem_tiled_copy_KV;
  auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_slice(tidx);
  auto gmem_thr_copy_KV = gmem_tiled_copy_KV.get_slice(tidx);

  auto produce_q = [&](int stage) {
    // gQ/sQ: (BLK_M, BLK_K, STAGES)
    auto tCgQ = gmem_thr_copy_Q.partition_S(gQ(_, _, stage));
    auto tCsQ = gmem_thr_copy_Q.partition_D(sQ(_, _, stage));
    cute::copy(gmem_tiled_copy_Q, tCgQ, tCsQ);
  };

  auto produce_q_rope = [&]() {
    auto tCgQ_rope = gmem_thr_copy_Q.partition_S(gQ_rope);
    auto tCsQ_rope = gmem_thr_copy_Q.partition_D(sQ_rope);
    cute::copy(gmem_tiled_copy_Q, tCgQ_rope, tCsQ_rope);
  };

  // (CPY, CPY_N, CPY_K, STAGES)
  auto produce_kv = [&](int ni, int stage) {
    // gKV: (BLK_N, BLK_K, n, STAGES)
    auto tCgKV = gmem_thr_copy_KV.partition_S(gKV(_, _, ni, stage));
    // sK: (BLK_N, BLK_K, STAGES)
    auto tCsKV = gmem_thr_copy_KV.partition_D(sK(_, _, stage));
    cute::copy(gmem_tiled_copy_KV, tCgKV, tCsKV);
  };

  Tensor tKsK_rope = gmem_thr_copy_KV.partition_D(sK_rope);
  auto produce_k_rope = [&](int ni) {
    auto tKgK_rope = gmem_thr_copy_KV.partition_S(gK_rope(_, _, ni));
    cute::copy(gmem_tiled_copy_KV, tKgK_rope, tKsK_rope);
  };

  TiledMma_QK tiled_mma_qk;
  auto thr_mma_qk = tiled_mma_qk.get_slice(tidx);
  // GEMM-I: S = Q@K.T
  // sQ/sK: (BLK_M, BLK_K, STAGES)
  auto tSrQ = partition_fragment_A(
      thr_mma_qk, sQ(_, _, _0{}), _, _2{});  // (MMA, MMA_M, _2)
  auto tSrK = partition_fragment_B(
      thr_mma_qk, sK(_, _, _0{}), _, _2{});  // (MMA, MMA_N, _2)

  auto tSrQ_rope =
      partition_fragment_A(thr_mma_qk, sQ_rope, _, _2{});  // (MMA, MMA_M, _2)
  auto tSrK_rope =
      partition_fragment_B(thr_mma_qk, sK_rope, _, _2{});  // (MMA, MMA_N, _2)

  // s2r tiled copy for q
  SmemTiledCopyQ smem_tiled_copy_Q;
  auto smem_thr_copy_Q = smem_tiled_copy_Q.get_slice(tidx);
  // (CPY, CPY_M, CPY_K, STAGES)
  auto tCsQ = smem_thr_copy_Q.partition_S(sQ);
  // (CPY, CPY_M, _2)
  auto tCrQ = smem_thr_copy_Q.retile_D(tSrQ);

  // (CPY, CPY_M, CPY_K)
  auto tCsQ_rope = smem_thr_copy_Q.partition_S(sQ_rope);
  // (CPY, CPY_M, _2)
  auto tCrQ_rope = smem_thr_copy_Q.retile_D(tSrQ_rope);

  // s2r tiled copy for k
  SmemTiledCopyK smem_tiled_copy_K;
  auto smem_thr_copy_K = smem_tiled_copy_K.get_slice(tidx);
  // (CPY, CPY_N, CPY_K, STAGES)
  auto tCsK = smem_thr_copy_K.partition_S(sK);
  // (CPY, CPY_N, _2)
  auto tCrK = smem_thr_copy_K.retile_D(tSrK);

  // (CPY, CPY_N, CPY_K)
  auto tCsK_rope = smem_thr_copy_K.partition_S(sK_rope);
  // (CPY, CPY_N, _2)
  auto tCrK_rope = smem_thr_copy_K.retile_D(tSrK_rope);

  // S = Q@K.T
  // tSrS: (MMA,MMA_M,MMA_N)
  auto compute_qk = [&](auto& tSrS, int s) {
    // (CPY, CPY_M, CPY_K, STAGES)
    auto tCsQ_s = tCsQ(_, _, _, s);
    auto tCsK_s = tCsK(_, _, _, s);
    // prefetch kv
    cute::copy(smem_tiled_copy_Q, tCsQ_s(_, _, _0{}), tCrQ(_, _, _0{}));
    cute::copy(smem_tiled_copy_K, tCsK_s(_, _, _0{}), tCrK(_, _, _0{}));

    CUTE_UNROLL
    for (int k = 0; k < size<2>(tCsQ_s); ++k) {
      // prefetch next kv
      if (k != size<2>(tCsQ_s) - 1) {
        const auto next_k = k + 1;
        cute::copy(
            smem_tiled_copy_Q, tCsQ_s(_, _, next_k), tCrQ(_, _, (next_k & 1)));
        cute::copy(
            smem_tiled_copy_K, tCsK_s(_, _, next_k), tCrK(_, _, (next_k & 1)));
      }
      cute::gemm(tiled_mma_qk, tSrQ(_, _, (k & 1)), tSrK(_, _, (k & 1)), tSrS);
    }
  };

  auto compute_qk_rope = [&](auto& tSrS) {
    // tCsQ_rope: (CPY, CPY_M, CPY_K) => tCrQ_rope: (CPY, CPY_M, _2)
    cute::copy(smem_tiled_copy_Q, tCsQ_rope(_, _, _0{}), tCrQ_rope(_, _, _0{}));
    cute::copy(smem_tiled_copy_K, tCsK_rope(_, _, _0{}), tCrK_rope(_, _, _0{}));

    CUTE_UNROLL
    for (int k = 0; k < size<2>(tCsQ_rope); ++k) {
      if (k != size<2>(tCsQ_rope) - 1) {
        const auto next_k = k + 1;
        cute::copy(smem_tiled_copy_Q,
                   tCsQ_rope(_, _, next_k),
                   tCrQ_rope(_, _, (next_k & 1)));
        cute::copy(smem_tiled_copy_K,
                   tCsK_rope(_, _, next_k),
                   tCrK_rope(_, _, (next_k & 1)));
      }
      cute::gemm(tiled_mma_qk,
                 tSrQ_rope(_, _, (k & 1)),
                 tSrK_rope(_, _, (k & 1)),
                 tSrS);
    }
  };

  // GEMM-II: O = softmax(S)@V
  TiledMma_PV tiled_mma_pv;
  auto thr_mma_pv = tiled_mma_pv.get_slice(tidx);
  // sS: (BLK_M, BLK_N)
  // (MMA, MMA_M, _2)
  auto tOrP = partition_fragment_A(thr_mma_pv, sP, _, _2{});
  // sVt: (BLK_K, BLK_N, STAGES)
  // (MMA, MMA_N, _2)
  auto tOrVt = partition_fragment_B(thr_mma_pv, sVt(_, _, _0{}), _, _2{});

  // s2r tiled copy for p
  SmemTiledCopyP smem_tiled_copy_P;
  auto smem_thr_copy_P = smem_tiled_copy_P.get_slice(tidx);
  // (CPY, CPY_M, CPY_K)
  auto tCsP = smem_thr_copy_P.partition_S(sP);
  // (CPY, CPY_M, _2)
  auto tCrP = smem_thr_copy_P.retile_D(tOrP);

  // s2r tiled copy for vt
  SmemTiledCopyVt smem_tiled_copy_Vt;
  auto smem_thr_copy_Vt = smem_tiled_copy_Vt.get_slice(tidx);
  // (CPY, CPY_N, CPY_K, STAGES)
  auto tCsVt = smem_thr_copy_Vt.partition_S(sVt);
  // (CPY, CPY_N, _2)
  auto tCrVt = smem_thr_copy_Vt.retile_D(tOrVt);

  // O = P*V = softmax(S)*V
  // tOrS: (MMA,MMA_M,MMA_K)
  auto compute_pv = [&](auto& tOrO, int s) {
    // (MMA,MMA_M,MMA_N, STAGES)
    auto tOrO_s = tOrO(_, _, _, s);

    // (CPY, CPY_N, CPY_K, STAGES)
    auto tCsVt_s = tCsVt(_, _, _, s);
    // tCsVt_s: (CPY, CPY_N, CPY_K) => tCrVt: (CPY, CPY_N, _2)
    cute::copy(smem_tiled_copy_P, tCsP(_, _, _0{}), tCrP(_, _, _0{}));
    cute::copy(smem_tiled_copy_Vt, tCsVt_s(_, _, _0{}), tCrVt(_, _, _0{}));

    CUTE_UNROLL
    for (int k = 0; k < size<2>(tCsVt_s); ++k) {
      if (k != size<2>(tCsVt_s) - 1) {
        const auto next_k = k + 1;
        cute::copy(
            smem_tiled_copy_P, tCsP(_, _, next_k), tCrP(_, _, (next_k & 1)));
        cute::copy(smem_tiled_copy_Vt,
                   tCsVt_s(_, _, next_k),
                   tCrVt(_, _, (next_k & 1)));
      }
      cute::gemm(
          tiled_mma_pv, tCrP(_, _, (k & 1)), tOrVt(_, _, (k & 1)), tOrO_s);
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

  // tOrO: (MMA,MMA_M,MMA_K,STAGES)
  auto epilogue = [&](const auto& tOrO) {
    // write output to gmem
    // 1. copy output from reg to smem (reuse sQ)
    SmemTiledCopyO smem_tiled_copy_O;
    auto smem_thr_copy_O = smem_tiled_copy_O.get_slice(tidx);
    CUTE_UNROLL
    for (int s = 0; s < kStages; ++s) {
      auto tOrO_s = tOrO(_, _, _, s);
      auto sO_s = sO(_, _, s);

      // cast Accumulator to Element type
      auto tOrO_ = make_tensor_like<DType>(tOrO_s);
      fast_cast(tOrO_s, tOrO_);

      auto tCrO = smem_thr_copy_O.retile_S(tOrO_);
      auto tCsO = smem_thr_copy_O.partition_D(sO_s);
      cute::copy(smem_tiled_copy_O, tCrO, tCsO);
    }
    // wait for smem copy done before gmem copy
    __syncthreads();

    // 2. copy output from smem to gmem
    GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx);

    auto tCsO = gmem_thr_copy_O.partition_S(sO);
    auto tCgO = gmem_thr_copy_O.partition_D(gO);
    cute::copy(gmem_tiled_copy_O, tCsO, tCgO);
  };

  // output accumulator: (MMA, MMA_M, MMA_K, STAGES)
  auto tOrO =
      partition_fragment_C(tiled_mma_pv, Shape<_BLK_M, _BLK_K, _STAGES>{});
  auto tOrO_mn = make_tensor(tOrO.data(), Layout::to_mns(tOrO.layout()));
  clear(tOrO);

  const int n_block_min = 0;
  const int n_block_max = cute::ceil_div(size<0>(KV), kBlockN);

  // ###############  Prologue  ###############
  // produce q_rope: [] => [q_rope, q...]
  produce_q_rope();
  CUTE_UNROLL
  for (int s = 0; s < kStages; ++s) {
    produce_q(s);
  }
  // produce k_rope: [q_rope, q...] => [q_rope, q..., k_rope, kv...]
  produce_k_rope(0);
  cp_async_fence();
  CUTE_UNROLL
  for (int s = 0; s < kStages; ++s) {
    produce_kv(0, s);
    cp_async_fence();
  }

  // ###############  Mainloop  ###############
  constexpr int kMMA_M = size<1>(tOrO);
  using Softmax = OnlineSoftmax<kRowsPerMMA * kMMA_M>;
  Softmax softmax(params.sm_scale_log2);

  CUTE_NO_UNROLL
  for (int ni = n_block_min; ni < n_block_max; ++ni) {
    // attention score accumulator, (MMA,MMA_M,MMA_N)
    auto tSrS = partition_fragment_C(tiled_mma_qk, Shape<_BLK_M, _BLK_N>{});
    auto tSrS_mn = make_tensor(tSrS.data(), Layout::to_mn(tSrS.layout()));
    clear(tSrS);

    // wait key, queue: [q, q_rope, kv, k_rope] => []
    cp_async_wait<kStages>();
    __syncthreads();

    // 1> S = Q_rope@K_rope.T
    compute_qk_rope(tSrS);
    cp_async_fence();

    // 2> S += Q@K.T
    CUTE_UNROLL
    for (int s = 0; s < kStages; ++s) {
      cp_async_wait<kStages>();
      __syncthreads();

      compute_qk(tSrS, s);
      cp_async_fence();
    }

    softmax.rescale(tSrS_mn, tOrO_mn, reduce_rowmax);

    // save tSrS from rmem to smem
    store_s_to_smem(tSrS);
    __syncthreads();

    // 3> O = softmax(S)*V
    const auto next_ni = ni + 1;
    if (next_ni != n_block_max) {
      produce_k_rope(next_ni);
      cp_async_fence();
      CUTE_UNROLL
      for (int s = 0; s < kStages; ++s) {
        compute_pv(tOrO, s);
        produce_kv(next_ni, s);
        cp_async_fence();
      }
    } else {
      CUTE_UNROLL
      for (int s = 0; s < kStages; ++s) {
        compute_pv(tOrO, s);
      }
    }
  }

  // ###############  Epilogue  ###############

  // normalize output: o /= rowsum
  softmax.finalize(tOrO_mn, reduce_rowsum);

  // write output to gmem
  epilogue(tOrO);
}

template <typename Traits,
          typename Params,
          bool EVEN_K = false,
          bool ALIBI = false,
          bool SOFT_CAP = false,
          bool LOCAL = false>
void launch_mla_kernel_sm80(const Params& params, cudaStream_t stream) {
  const auto batch_size = params.batch_size;
  const auto max_q_packed_len = params.max_q_len * params.n_heads;

  const auto smem_size = Traits::kSmemSize;
  // print("smem_size: %d\n", smem_size);

  auto mla_kernel =
      mla_kernel_sm80<Traits, Params, EVEN_K, ALIBI, SOFT_CAP, LOCAL>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      mla_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
  // TODO: support persistent kernels
  dim3 grid(cute::ceil_div(max_q_packed_len, Traits::kBlockM), batch_size, 1);
  dim3 block = Traits::kThreadNum;
  mla_kernel<<<grid, block, smem_size, stream>>>(params);
}

}  // namespace llm