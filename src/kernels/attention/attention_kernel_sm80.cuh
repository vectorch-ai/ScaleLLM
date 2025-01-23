#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cute/layout.hpp>
#include <cute/tensor.hpp>

#include "attention_tile.h"
#include "cute/config.hpp"
#include "cute_extensions.cuh"
#include "fast_cast.cuh"
#include "mask.h"
#include "online_softmax.cuh"
#include "ptx.cuh"

namespace llm {

template <typename Traits,
          typename Params,
          bool EVEN_K,
          bool ALIBI,
          bool SOFT_CAP,
          bool LOCAL>
__global__ void mha_kernel_sm80(__grid_constant__ const Params params) {
  using namespace cute;

  constexpr int kBlockM = Traits::kBlockM;
  constexpr int kBlockN = Traits::kBlockN;
  constexpr int kBlockK = Traits::kBlockK;
  constexpr int kHeadDim = Traits::kHeadDim;
  constexpr int kRowsPerMMA = Traits::kRowsPerMMA;

  using _BLK_M = Int<kBlockM>;
  using _BLK_N = Int<kBlockN>;
  using _BLK_K = Int<kBlockK>;
  using _HEAD_DIM = Int<kHeadDim>;

  // type alias
  using DType = typename Traits::DType;

  using TiledMma = typename Traits::TiledMma;
  using Layout = typename Traits::LayoutConvertor;

  using SmemLayoutQ = typename Traits::SmemLayoutQ;
  using SmemLayoutK = typename Traits::SmemLayoutK;
  using SmemLayoutV = typename Traits::SmemLayoutV;
  using SmemLayoutVt = typename Traits::SmemLayoutVt;
  using SmemLayoutO = typename Traits::SmemLayoutO;
  using GmemTiledCopyQ = typename Traits::GmemTiledCopyQ;
  using GmemTiledCopyKV = typename Traits::GmemTiledCopyKV;
  using GmemTiledCopyO = typename Traits::GmemTiledCopyO;

  using SmemTiledCopyQ = typename Traits::SmemTiledCopyQ;
  using SmemTiledCopyK = typename Traits::SmemTiledCopyK;
  using SmemTiledCopyVt = typename Traits::SmemTiledCopyVt;
  using SmemTiledCopyO = typename Traits::SmemTiledCopyO;

  const int m_block = blockIdx.x;
  const auto batch_idx = blockIdx.y;
  const auto head_idx = blockIdx.z;
  const auto tidx = threadIdx.x;

  AttentionTile<Params> tile(params);

  const int group_size = params.n_heads / params.n_kv_heads;
  // ProblemShape
  // (q_len, HEAD_DIM)
  auto [Q, O] = tile.template get_qo_tile<DType>(batch_idx, head_idx);
  // (kv_len, HEAD_DIM)
  auto [K, V] =
      tile.template get_kv_tile<DType>(batch_idx, head_idx / group_size);

  const int q_len = size<0>(Q.shape());
  const int kv_len = size<0>(K.shape());

  if (m_block * kBlockM >= q_len) {
    // out of bound, return
    return;
  }

  const int head_dim = params.head_dim;
  const int sliding_window = LOCAL ? params.sliding_window : kv_len;
  const float logits_soft_cap = params.logits_soft_cap;
  const float sm_scale = params.sm_scale;
  const float sm_scale_log2 = params.sm_scale_log2;
  const float alibi_slope =
      ALIBI ? (params.alibi_slopes_ptr[head_idx] / sm_scale) : 0.0f;

  // preprocess input parameters
  auto apply_logits_soft_cap = [&](auto& tSrAccS) {
    if constexpr (SOFT_CAP) {
      CUTE_UNROLL
      for (int i = 0; i < size(tSrAccS); ++i) {
        tSrAccS(i) = ptx::tanh(tSrAccS(i) * logits_soft_cap);
      }
    }
  };

  // Gmem
  // (BLK_M, HEAD_DIM)
  Tensor gQ =
      local_tile(Q, Shape<_BLK_M, _HEAD_DIM>{}, make_coord(m_block, _0{}));
  Tensor gO =
      local_tile(O, Shape<_BLK_M, _HEAD_DIM>{}, make_coord(m_block, _0{}));
  // (BLK_N, HEAD_DIM, n)
  Tensor gK = local_tile(K, Shape<_BLK_N, _HEAD_DIM>{}, make_coord(_, _0{}));
  Tensor gV = local_tile(V, Shape<_BLK_N, _HEAD_DIM>{}, make_coord(_, _0{}));

  // Smem
  extern __shared__ char smem[];
  DType* q_smem = (DType*)smem;
  DType* k_smem = q_smem + cosize(SmemLayoutQ{});
  DType* v_smem = k_smem + cosize(SmemLayoutK{});

  // (BLK_M, BLK_K), k-major
  Tensor sQ = make_tensor(make_smem_ptr(q_smem), SmemLayoutQ{});
  // (BLK_N, BLK_K), k-major
  Tensor sK = make_tensor(make_smem_ptr(k_smem), SmemLayoutK{});
  Tensor sV = make_tensor(make_smem_ptr(v_smem), SmemLayoutV{});

  // Tensor for V^t; used in GEMM-II.
  // (BLK_K, BLK_N)
  Tensor sVt = make_tensor(make_smem_ptr(v_smem), SmemLayoutVt{});

  // Tiled Copy
  // g2s tiled copy for qkv
  GmemTiledCopyQ gmem_tiled_copy_Q;
  GmemTiledCopyKV gmem_tiled_copy_KV;
  auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tidx);
  auto gmem_thr_copy_KV = gmem_tiled_copy_KV.get_thread_slice(tidx);

  // coordinate tensor for oob handling
  // (BLK_M, HEAD_DIM) -> (blk_m, head_dim)
  Tensor cQ = make_identity_tensor(Shape<_BLK_M, _HEAD_DIM>{});
  Tensor tQcQ = gmem_thr_copy_Q.partition_S(cQ);
  // (BLK_N, HEAD_DIM) -> (blk_n, head_dim)
  Tensor cKV = make_identity_tensor(Shape<_BLK_N, _HEAD_DIM>{});
  Tensor tKcKV = gmem_thr_copy_KV.partition_S(cKV);

  auto produce_q = [&]() {
    auto tQgQ = gmem_thr_copy_Q.partition_S(gQ);
    auto tQsQ = gmem_thr_copy_Q.partition_D(sQ);
    safe_copy<EVEN_K,
              /*EVEN_MN=*/false,
              /*ZERO_FILL_MN=*/true,
              /*ZERO_FILL_K=*/true>(
        gmem_tiled_copy_Q,
        tQgQ,
        tQsQ,
        tQcQ,
        make_coord(q_len - m_block * kBlockM, head_dim));
  };

  // TODO: seperate mask iterations
  Tensor tKsK = gmem_thr_copy_KV.partition_D(sK);
  auto produce_k = [&](int ni) {
    auto tKgK = gmem_thr_copy_KV.partition_S(gK(_, _, ni));
    // skip zfill_mn for k since mask will mask out oob with -inf
    safe_copy<EVEN_K,
              /*EVEN_MN=*/false,
              /*ZERO_FILL_MN=*/false,
              /*ZERO_FILL_K=*/true>(
        gmem_tiled_copy_KV,
        tKgK,
        tKsK,
        tKcKV,
        make_coord(kv_len - ni * kBlockN, head_dim));
  };

  Tensor tVsV = gmem_thr_copy_KV.partition_D(sV);
  auto produce_v = [&](int ni) {
    auto tVgV = gmem_thr_copy_KV.partition_S(gV(_, _, ni));
    // skipping ZFILL_MN for v may cause nan issue
    safe_copy<EVEN_K,
              /*EVEN_MN=*/false,
              /*ZERO_FILL_MN=*/true,
              /*ZERO_FILL_K=*/true>(
        gmem_tiled_copy_KV,
        tVgV,
        tVsV,
        tKcKV,
        make_coord(kv_len - ni * kBlockN, head_dim));
  };

  TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_slice(tidx);
  // GEMM-I: S = Q@K.T
  auto tSrQ = thr_mma.partition_fragment_A(sQ);  // (MMA,MMA_M,MMA_K)
  auto tSrK = thr_mma.partition_fragment_B(sK);  // (MMA,MMA_N,MMA_K)

  // s2r tiled copy for qkv
  SmemTiledCopyQ smem_tiled_copy_Q;
  auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
  auto tSsQ = smem_thr_copy_Q.partition_S(sQ);
  auto tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);

  SmemTiledCopyK smem_tiled_copy_K;
  auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
  auto tSsK = smem_thr_copy_K.partition_S(sK);
  auto tSrK_copy_view = smem_thr_copy_K.retile_D(tSrK);

  // S = Q@K.T
  // tSrAccS: (MMA,MMA_M,MMA_N)
  auto compute_qk = [&](auto& tSrAccS) {
    // prefetch kv
    cute::copy(smem_tiled_copy_Q, tSsQ(_, _, _0{}), tSrQ_copy_view(_, _, _0{}));
    cute::copy(smem_tiled_copy_K, tSsK(_, _, _0{}), tSrK_copy_view(_, _, _0{}));

    CUTE_UNROLL
    for (int ki = 0; ki < size<2>(tSrQ); ++ki) {
      // prefetch next kv
      if (ki != size<2>(tSrQ) - 1) {
        const auto next_ki = ki + 1;
        cute::copy(smem_tiled_copy_Q,
                   tSsQ(_, _, next_ki),
                   tSrQ_copy_view(_, _, next_ki));
        cute::copy(smem_tiled_copy_K,
                   tSsK(_, _, next_ki),
                   tSrK_copy_view(_, _, next_ki));
      }
      cute::gemm(tiled_mma, tSrQ(_, _, ki), tSrK(_, _, ki), tSrAccS);
    }
  };

  // GEMM-II: O = softmax(S)@V
  auto tOrVt = thr_mma.partition_fragment_B(sVt);  // (MMA,MMA_K,MMA_N)

  SmemTiledCopyVt smem_tiled_copy_Vt;
  auto smem_thr_copy_Vt = smem_tiled_copy_Vt.get_thread_slice(tidx);
  auto tOsVt = smem_thr_copy_Vt.partition_S(sVt);
  auto tOrVt_copy_view = smem_thr_copy_Vt.retile_D(tOrVt);

  // O = softmax(S)*V
  // tSrAccS: (MMA,MMA_M,MMA_N)
  // tOrAccO: (MMA,MMA_M,MMA_K)
  auto compute_sv = [&](const auto& tSrAccS, auto& tOrAccO) {
    // cast scores from Accumulator to Element
    auto tSrS = make_tensor_like<DType>(tSrAccS);
    fast_cast(tSrAccS, tSrS);

    // convert layout from gemm-I C to gemm-II A
    auto tOrS = make_tensor(tSrS.data(), Layout::to_mma_a(tSrS.layout()));

    // prefetch V^t
    cute::copy(
        smem_tiled_copy_Vt, tOsVt(_, _, _0{}), tOrVt_copy_view(_, _, _0{}));
    CUTE_UNROLL
    for (int ki = 0; ki < size<2>(tOrS); ++ki) {
      // prefetch next V^t
      if (ki != size<2>(tOrS) - 1) {
        const auto next_ki = ki + 1;
        cute::copy(smem_tiled_copy_Vt,
                   tOsVt(_, _, next_ki),
                   tOrVt_copy_view(_, _, next_ki));
      }
      cute::gemm(tiled_mma, tOrS(_, _, ki), tOrVt(_, _, ki), tOrAccO);
    }
  };

  // tOrAccO: (MMA,MMA_M,MMA_K)
  auto epilogue = [&](const auto& tOrAccO) {
    // write output to gmem
    // 1> cast output from ElementAccumulator to Element
    auto tOrO = make_tensor_like<DType>(tOrAccO);
    fast_cast(tOrAccO, tOrO);

    // 2. copy output from reg to smem (reuse sQ)
    auto sO = make_tensor(sQ.data(), SmemLayoutO{});

    SmemTiledCopyO smem_tiled_copy_O;
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
    auto taccOrO = smem_thr_copy_O.retile_S(tOrO);
    auto taccOsO = smem_thr_copy_O.partition_D(sO);
    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

    // 3. copy output from smem to gmem
    GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);

    // (BLK_M, HEAD_DIM) -> (blk_m, head_dim)
    auto cO = make_identity_tensor(Shape<_BLK_M, _HEAD_DIM>{});

    auto tOsO = gmem_thr_copy_O.partition_S(sO);  // (CPY,CPY_M,CPY_K)
    auto tOgO = gmem_thr_copy_O.partition_D(gO);  // (CPY,CPY_M,CPY_K)
    // (CPY,CPY_M,CPY_K) -> (blk_m, head_dim)
    auto tOcO = gmem_thr_copy_O.partition_D(cO);

    // wait for smem copy done before gmem copy
    __syncthreads();
    safe_copy<EVEN_K,
              /*EVEN_MN=*/false,
              /*ZERO_FILL_MN=*/false,
              /*ZERO_FILL_K=*/false>(
        gmem_tiled_copy_O,
        tOsO,
        tOgO,
        tOcO,
        make_coord(q_len - m_block * kBlockM, head_dim));
  };

  const int diagonal = m_block * kBlockM + kv_len - q_len;
  // process kv in range: [kv_idx_min, kv_idx_max)
  const int kv_idx_min = std::max(0, diagonal - sliding_window);
  const int kv_idx_max = std::min(kv_len, diagonal + kBlockM);
  const int n_block_min = LOCAL ? kv_idx_min / kBlockN : 0;
  const int n_block_max = cute::ceil_div(kv_idx_max, kBlockN);
  // TODO: handle n_block_min >= n_block_max

  // ###############  Prologue  ###############

  // produce q: [] => [q]
  produce_q();
  cp_async_fence();
  // produce k: [q] => [q, k]
  produce_k(n_block_min);
  cp_async_fence();

  // ###############  Mainloop  ###############

  // output accumulator, (MMA,MMA_M,MMA_K)
  auto tOrAccO = partition_fragment_C(tiled_mma, Shape<_BLK_M, _HEAD_DIM>{});
  auto tOrAccO_rc_view =
      make_tensor(tOrAccO.data(), Layout::to_rowcol(tOrAccO.layout()));

  // attention score accumulator, (MMA,MMA_M,MMA_N)
  auto tSrAccS = partition_fragment_C(tiled_mma, Shape<_BLK_M, _BLK_N>{});
  auto tSrAccS_rc_view =
      make_tensor(tSrAccS.data(), Layout::to_rowcol(tSrAccS.layout()));

  OnlineSoftmax<kRowsPerMMA * size<1>(tOrAccO)> softmax(sm_scale_log2);
  Mask<kBlockM, kBlockM, ALIBI, LOCAL> mask(
      q_len, kv_len, sliding_window, alibi_slope);

  clear(tOrAccO);
  CUTE_NO_UNROLL
  for (int ni = n_block_min; ni < n_block_max; ++ni) {
    clear(tSrAccS);

    // wait k, queue: [q, k] => []
    cp_async_wait<0>();
    __syncthreads();

    // produce v, [] => [v]
    produce_v(ni);
    cp_async_fence();

    // 1> S = Q@K.T
    compute_qk(tSrAccS);

    // apply soft cap if needed
    if constexpr (SOFT_CAP) {
      apply_logits_soft_cap(tSrAccS);
    }

    // apply mask for block (m_block, ni)
    mask.apply(tSrAccS_rc_view, m_block, ni, tidx);

    // apply softmax and rescale
    softmax.rescale(tSrAccS_rc_view, tOrAccO_rc_view);

    // wait v, [v] => []
    cp_async_wait<0>();
    __syncthreads();

    // produce next k: [] => [k]
    if (ni != n_block_max - 1) {
      produce_k(ni + 1);
    }
    cp_async_fence();

    // 2> O = softmax(S)*V
    compute_sv(tSrAccS, tOrAccO);
  }

  // ###############  Epilogue  ###############

  // normalize output: o /= rowsum
  softmax.finalize(tOrAccO_rc_view);

  // write output to gmem
  epilogue(tOrAccO);
}

}  // namespace llm