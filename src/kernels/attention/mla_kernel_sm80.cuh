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
__global__ void mla_kernel_sm80(__grid_constant__ const Params params) {
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

  using TiledMma = typename Traits::TiledMma;
  using Layout = typename Traits::LayoutConvertor;

  using SmemLayoutQ = typename Traits::SmemLayoutQ;
  using SmemLayoutKV = typename Traits::SmemLayoutKV;
  using SmemLayoutQRope = typename Traits::SmemLayoutQRope;
  using SmemLayoutKRope = typename Traits::SmemLayoutKRope;
  using SmemLayoutVt = typename Traits::SmemLayoutVt;
  using SmemLayoutO = typename Traits::SmemLayoutO;

  using GmemTiledCopyQ = typename Traits::GmemTiledCopyQ;
  using GmemTiledCopyKV = typename Traits::GmemTiledCopyKV;
  using GmemTiledCopyO = typename Traits::GmemTiledCopyO;

  using SmemTiledCopyQ = typename Traits::SmemTiledCopyQ;
  using SmemTiledCopyK = typename Traits::SmemTiledCopyK;
  using SmemTiledCopyVt = typename Traits::SmemTiledCopyVt;
  using SmemTiledCopyO = typename Traits::SmemTiledCopyO;

  MLATile<Params> tile(params);

  // ProblemShape
  // Q/O: (q_packed_len, HEAD_DIM)
  // KV: (kv_len, HEAD_DIM)
  // Q/K_ROPE: (q_packed_len, ROPE_HEAD_DIM)
  auto [Q, Q_ROPE, O] = tile.template get_qo_tile<DType>(blockIdx.y);
  auto [KV, K_ROPE] = tile.template get_kv_tile<DType>(blockIdx.y);

  if (blockIdx.x * kBlockM >= size<0>(Q)) {
    // m out of bound, return
    return;
  }

  // Gmem
  // (BLK_M, BLK_K, STAGES)
  Tensor gQ = local_tile(Q, Shape<_BLK_M, _BLK_K>{}, make_coord(blockIdx.x, _));
  Tensor gO = local_tile(O, Shape<_BLK_M, _BLK_K>{}, make_coord(blockIdx.x, _));
  // (BLK_N, BLK_K, n, STAGES)
  Tensor gKV = local_tile(KV, Shape<_BLK_N, _BLK_K>{}, make_coord(_, _));

  // (BLK_M, ROPE_HEAD_DIM)
  Tensor gQ_rope = local_tile(
      Q_ROPE, Shape<_BLK_M, _ROPE_HEAD_DIM>{}, make_coord(blockIdx.x, _0{}));
  // (BLK_N, ROPE_HEAD_DIM, n)
  Tensor gK_rope =
      local_tile(K_ROPE, Shape<_BLK_N, _ROPE_HEAD_DIM>{}, make_coord(_, _0{}));

  // Smem
  extern __shared__ char smem[];
  DType* q_smem = (DType*)smem;
  DType* kv_smem = q_smem + cosize(SmemLayoutQ{});
  DType* q_rope_smem = kv_smem + cosize(SmemLayoutKV{});
  DType* k_rope_smem = q_rope_smem + cosize(SmemLayoutQRope{});

  // (BLK_M, BLK_K, STAGES), k-major
  Tensor sQ = make_tensor(make_smem_ptr(q_smem), SmemLayoutQ{});
  // (BLK_N, BLK_K, STAGES), k-major
  Tensor sK = make_tensor(make_smem_ptr(kv_smem), SmemLayoutKV{});

  // (BLK_M, ROPE_HEAD_DIM), k-major
  Tensor sQ_rope = make_tensor(make_smem_ptr(q_rope_smem), SmemLayoutQRope{});
  // (BLK_N, ROPE_HEAD_DIM), k-major
  Tensor sK_rope = make_tensor(make_smem_ptr(k_rope_smem), SmemLayoutKRope{});

  // Tensor for V^t; used in GEMM-II.
  // (BLK_K, BLK_N, STAGES)
  Tensor sVt = make_tensor(make_smem_ptr(kv_smem), SmemLayoutVt{});

  // Tiled Copy
  // g2s tiled copy for qkv
  GmemTiledCopyQ gmem_tiled_copy_Q;
  GmemTiledCopyKV gmem_tiled_copy_KV;
  auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_slice(threadIdx.x);
  auto gmem_thr_copy_KV = gmem_tiled_copy_KV.get_slice(threadIdx.x);

  auto produce_q = [&](int stage) {
    // gQ/sQ: (BLK_M, BLK_K, STAGES)
    auto tGCgQ = gmem_thr_copy_Q.partition_S(gQ(_, _, stage));
    auto tGCsQ = gmem_thr_copy_Q.partition_D(sQ(_, _, stage));
    cute::copy(gmem_tiled_copy_Q, tGCgQ, tGCsQ);
    cp_async_fence();
  };

  auto produce_q_rope = [&]() {
    auto tQgQ_rope = gmem_thr_copy_Q.partition_S(gQ_rope);
    auto tQsQ_rope = gmem_thr_copy_Q.partition_D(sQ_rope);
    cute::copy(gmem_tiled_copy_Q, tQgQ_rope, tQsQ_rope);
    cp_async_fence();
  };

  // (CPY, CPY_N, CPY_K, STAGES)
  auto produce_kv = [&](int ni, int stage) {
    // gKV: (BLK_N, BLK_K, n, STAGES)
    auto tGCgKV = gmem_thr_copy_KV.partition_S(gKV(_, _, ni, stage));
    // sK: (BLK_N, BLK_K, STAGES)
    Tensor tGCsKV = gmem_thr_copy_KV.partition_D(sK(_, _, stage));
    cute::copy(gmem_tiled_copy_KV, tGCgKV, tGCsKV);
    cp_async_fence();
  };

  Tensor tKsK_rope = gmem_thr_copy_KV.partition_D(sK_rope);
  auto produce_k_rope = [&](int ni) {
    auto tKgK_rope = gmem_thr_copy_KV.partition_S(gK_rope(_, _, ni));
    cute::copy(gmem_tiled_copy_KV, tKgK_rope, tKsK_rope);
    cp_async_fence();
  };

  TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_slice(threadIdx.x);
  // GEMM-I: S = Q@K.T
  // gQ/sQ: (BLK_M, BLK_K, STAGES)
  auto tSrQ = thr_mma.partition_fragment_A(sQ(_, _, _0{}));
  auto tSrK = thr_mma.partition_fragment_B(sK(_, _, _0{}));

  auto tSrQ_rope = thr_mma.partition_fragment_A(sQ_rope);
  auto tSrK_rope = thr_mma.partition_fragment_B(sK_rope);

  // s2r tiled copy for qkv
  SmemTiledCopyQ smem_tiled_copy_Q;
  auto smem_thr_copy_Q = smem_tiled_copy_Q.get_slice(threadIdx.x);
  // (CPY, CPY_M, CPY_K, STAGES)
  auto tCsQ = smem_thr_copy_Q.partition_S(sQ);
  // (CPY, CPY_M, CPY_K)
  auto tCrQ = smem_thr_copy_Q.retile_D(tSrQ);

  auto tCsQ_rope = smem_thr_copy_Q.partition_S(sQ_rope);
  auto tCrQ_rope = smem_thr_copy_Q.retile_D(tSrQ_rope);

  SmemTiledCopyK smem_tiled_copy_K;
  auto smem_thr_copy_K = smem_tiled_copy_K.get_slice(threadIdx.x);
  // (CPY, CPY_N, CPY_K, STAGES)
  auto tCsK = smem_thr_copy_K.partition_S(sK);
  // (CPY, CPY_M, CPY_K)
  auto tCrK = smem_thr_copy_K.retile_D(tSrK);

  auto tCsK_rope = smem_thr_copy_K.partition_S(sK_rope);
  auto tCrK_rope = smem_thr_copy_K.retile_D(tSrK_rope);

  // S = Q@K.T
  // tSrS: (MMA,MMA_M,MMA_N)
  auto compute_qk = [&](auto& tSrS, int stage) {
    // (CPY, CPY_M, CPY_K, STAGES)
    auto tCsQ_ = tCsQ(_, _, _, stage);
    auto tCsK_ = tCsK(_, _, _, stage);
    // prefetch kv
    cute::copy(smem_tiled_copy_Q, tCsQ_(_, _, _0{}), tCrQ(_, _, _0{}));
    cute::copy(smem_tiled_copy_K, tCsK_(_, _, _0{}), tCrK(_, _, _0{}));

    CUTE_UNROLL
    for (int ki = 0; ki < size<2>(tSrQ); ++ki) {
      // prefetch next kv
      if (ki != size<2>(tSrQ) - 1) {
        const auto next_ki = ki + 1;
        cute::copy(
            smem_tiled_copy_Q, tCsQ_(_, _, next_ki), tCrQ(_, _, next_ki));
        cute::copy(
            smem_tiled_copy_K, tCsK_(_, _, next_ki), tCrK(_, _, next_ki));
      }
      cute::gemm(tiled_mma, tSrQ(_, _, ki), tSrK(_, _, ki), tSrS);
    }
  };

  auto compute_qk_rope = [&](auto& tSrS) {
    // prefetch qk_rope
    cute::copy(smem_tiled_copy_Q, tCsQ_rope(_, _, _0{}), tCrQ_rope(_, _, _0{}));
    cute::copy(smem_tiled_copy_K, tCsK_rope(_, _, _0{}), tCrK_rope(_, _, _0{}));

    CUTE_UNROLL
    for (int ki = 0; ki < size<2>(tSrQ_rope); ++ki) {
      // prefetch next qk_rope
      if (ki != size<2>(tSrQ_rope) - 1) {
        const auto next_ki = ki + 1;
        cute::copy(smem_tiled_copy_Q,
                   tCsQ_rope(_, _, next_ki),
                   tCrQ_rope(_, _, next_ki));
        cute::copy(smem_tiled_copy_K,
                   tCsK_rope(_, _, next_ki),
                   tCrK_rope(_, _, next_ki));
      }
      cute::gemm(tiled_mma, tSrQ_rope(_, _, ki), tSrK_rope(_, _, ki), tSrS);
    }
  };

  // GEMM-II: O = softmax(S)@V
  // (MMA, MMA_M, MMA_N)
  auto tOrVt = thr_mma.partition_fragment_B(sVt(_, _, _0{}));

  SmemTiledCopyVt smem_tiled_copy_Vt;
  auto smem_thr_copy_Vt = smem_tiled_copy_Vt.get_slice(threadIdx.x);
  // (CPY, CPY_N, CPY_K, STAGES)
  auto tCsVt = smem_thr_copy_Vt.partition_S(sVt);
  // (CPY, CPY_M, CPY_K)
  auto tCrVt = smem_thr_copy_Vt.retile_D(tOrVt);

  // O = softmax(S)*V
  // tOrS: (MMA,MMA_M,MMA_K)
  // tOrO: (MMA,MMA_M,MMA_N, STAGES)
  auto compute_sv = [&](const auto& tOrS, auto& tOrO, int stage) {
    // (CPY, CPY_N, CPY_K, STAGES)
    auto tCsVt_ = tCsVt(_, _, _, stage);
    auto tOrO_ = tOrO(_, _, _, stage);
    // prefetch V^t
    cute::copy(smem_tiled_copy_Vt, tCsVt_(_, _, _0{}), tCrVt(_, _, _0{}));
    CUTE_UNROLL
    for (int ki = 0; ki < size<2>(tOrS); ++ki) {
      // prefetch next V^t
      if (ki != size<2>(tOrS) - 1) {
        const auto next_ki = ki + 1;
        cute::copy(
            smem_tiled_copy_Vt, tCsVt_(_, _, next_ki), tCrVt(_, _, next_ki));
      }
      cute::gemm(tiled_mma, tOrS(_, _, ki), tOrVt(_, _, ki), tOrO_);
    }
  };

  // tOrO: (MMA,MMA_M,MMA_K,STAGES)
  auto epilogue = [&](const auto& tOrO) {
    // write output to gmem
    // 1. cast output from ElementAccumulator to Element
    auto tOrO_ = make_tensor_like<DType>(tOrO);
    fast_cast(tOrO, tOrO_);

    auto sO = make_tensor(sQ.data(), SmemLayoutO{});
    // 2. copy output from reg to smem (reuse sQ)
    {
      SmemTiledCopyO smem_tiled_copy_O;
      auto smem_thr_copy_O = smem_tiled_copy_O.get_slice(threadIdx.x);
      auto tCrO = smem_thr_copy_O.retile_S(tOrO_);
      auto tCsO = smem_thr_copy_O.partition_D(sO);
      cute::copy(smem_tiled_copy_O, tCrO, tCsO);
    }
    // wait for smem copy done before gmem copy
    __syncthreads();

    // 3. copy output from smem to gmem
    {
      GmemTiledCopyO gmem_tiled_copy_O;
      auto gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(threadIdx.x);

      auto tCsO = gmem_thr_copy_O.partition_S(sO);
      auto tCgO = gmem_thr_copy_O.partition_D(gO);
      cute::copy(gmem_tiled_copy_O, tCsO, tCgO);
    }
  };

  // output accumulator: (MMA, MMA_M, MMA_K, STAGES)
  auto tOrO = partition_fragment_C(tiled_mma, Shape<_BLK_M, _BLK_K, _STAGES>{});
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
  CUTE_UNROLL
  for (int s = 0; s < kStages; ++s) {
    produce_kv(0, s);
  }

  // ###############  Mainloop  ###############
  constexpr int kMMA_M = size<1>(tOrO);
  using Softmax = OnlineSoftmax<kRowsPerMMA * kMMA_M>;
  Softmax softmax(params.sm_scale_log2);

  CUTE_NO_UNROLL
  for (int ni = n_block_min; ni < n_block_max; ++ni) {
    // attention score accumulator, (MMA,MMA_M,MMA_N)
    auto tSrS = partition_fragment_C(tiled_mma, Shape<_BLK_M, _BLK_N>{});
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

    softmax.rescale(tSrS_mn, tOrO_mn);

    // 3> O = softmax(S)*V
    // cast scores from Accumulator to Element
    auto tSrS_ = make_tensor_like<DType>(tSrS);
    fast_cast(tSrS, tSrS_);
    // convert layout from gemm-I C to gemm-II A
    auto tOrS = make_tensor(tSrS_.data(), Layout::to_mma_a(tSrS_.layout()));
    const auto next_ni = (ni + 1 < n_block_max) ? ni + 1 : ni;
    produce_k_rope(next_ni);
    CUTE_UNROLL
    for (int s = 0; s < kStages; ++s) {
      compute_sv(tOrS, tOrO, s);
      produce_kv(next_ni, s);
    }
  }

  // ###############  Epilogue  ###############

  // normalize output: o /= rowsum
  softmax.finalize(tOrO_mn);

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