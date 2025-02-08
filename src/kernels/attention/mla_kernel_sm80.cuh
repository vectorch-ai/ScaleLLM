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
  constexpr int kHeadDim = Traits::kHeadDim;
  constexpr int kRopeHeadDim = Traits::kRopeHeadDim;
  constexpr int kRowsPerMMA = Traits::kRowsPerMMA;

  using _BLK_M = Int<kBlockM>;
  using _BLK_N = Int<kBlockN>;
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

  const int m_block = blockIdx.x;
  const int batch_idx = blockIdx.y;
  const int tidx = threadIdx.x;

  const float sm_scale_log2 = params.sm_scale_log2;

  MLATile<Params> tile(params);

  // ProblemShape
  // Q/O: (q_packed_len, HEAD_DIM)
  // KV: (kv_len, HEAD_DIM)
  // Q/K_ROPE: (q_packed_len, ROPE_HEAD_DIM)
  auto [Q, Q_ROPE, O] = tile.template get_qo_tile<DType>(batch_idx);
  auto [KV, K_ROPE] = tile.template get_kv_tile<DType>(batch_idx);

  const int q_packed_len = size<0>(Q);
  // const int q_len = q_packed_len / group_size;
  const int kv_len = size<0>(KV);

  if (m_block * kBlockM >= q_packed_len) {
    // m out of bound, return
    return;
  }

  // Gmem
  // (BLK_M, HEAD_DIM)
  Tensor gQ =
      local_tile(Q, Shape<_BLK_M, _HEAD_DIM>{}, make_coord(m_block, _0{}));
  Tensor gO =
      local_tile(O, Shape<_BLK_M, _HEAD_DIM>{}, make_coord(m_block, _0{}));
  // (BLK_N, HEAD_DIM, n)
  Tensor gKV = local_tile(KV, Shape<_BLK_N, _HEAD_DIM>{}, make_coord(_, _0{}));

  // (BLK_M, ROPE_HEAD_DIM)
  Tensor gQ_rope = local_tile(
      Q_ROPE, Shape<_BLK_M, _ROPE_HEAD_DIM>{}, make_coord(m_block, _0{}));
  // (BLK_N, ROPE_HEAD_DIM, n)
  Tensor gK_rope =
      local_tile(K_ROPE, Shape<_BLK_N, _ROPE_HEAD_DIM>{}, make_coord(_, _0{}));

  // Smem
  extern __shared__ char smem[];
  DType* q_smem = (DType*)smem;
  DType* kv_smem = q_smem + cosize(SmemLayoutQ{});
  DType* q_rope_smem = kv_smem + cosize(SmemLayoutKV{});
  DType* k_rope_smem = q_rope_smem + cosize(SmemLayoutQRope{});

  // (BLK_M, HEAD_DIM), k-major
  Tensor sQ = make_tensor(make_smem_ptr(q_smem), SmemLayoutQ{});
  // (BLK_N, HEAD_DIM), k-major
  Tensor sK = make_tensor(make_smem_ptr(kv_smem), SmemLayoutKV{});

  // (BLK_M, ROPE_HEAD_DIM), k-major
  Tensor sQ_rope = make_tensor(make_smem_ptr(q_rope_smem), SmemLayoutQRope{});
  // (BLK_N, ROPE_HEAD_DIM), k-major
  Tensor sK_rope = make_tensor(make_smem_ptr(k_rope_smem), SmemLayoutKRope{});

  // Tensor for V^t; used in GEMM-II.
  // (HEAD_DIM, BLK_N), m-major
  Tensor sVt = make_tensor(make_smem_ptr(kv_smem), SmemLayoutVt{});

  // Tiled Copy
  // g2s tiled copy for qkv
  GmemTiledCopyQ gmem_tiled_copy_Q;
  GmemTiledCopyKV gmem_tiled_copy_KV;
  auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tidx);
  auto gmem_thr_copy_KV = gmem_tiled_copy_KV.get_thread_slice(tidx);

  auto produce_q = [&]() {
    auto tQgQ = gmem_thr_copy_Q.partition_S(gQ);
    auto tQsQ = gmem_thr_copy_Q.partition_D(sQ);
    cute::copy(gmem_tiled_copy_Q, tQgQ, tQsQ);
  };

  auto produce_q_rope = [&]() {
    auto tQgQ_rope = gmem_thr_copy_Q.partition_S(gQ_rope);
    auto tQsQ_rope = gmem_thr_copy_Q.partition_D(sQ_rope);
    cute::copy(gmem_tiled_copy_Q, tQgQ_rope, tQsQ_rope);
  };

  Tensor tKsKV = gmem_thr_copy_KV.partition_D(sK);
  auto produce_kv = [&](int ni) {
    auto tKgKV = gmem_thr_copy_KV.partition_S(gKV(_, _, ni));
    cute::copy(gmem_tiled_copy_KV, tKgKV, tKsKV);
  };

  Tensor tKsK_rope = gmem_thr_copy_KV.partition_D(sK_rope);
  auto produce_k_rope = [&](int ni) {
    auto tKgK_rope = gmem_thr_copy_KV.partition_S(gK_rope(_, _, ni));
    cute::copy(gmem_tiled_copy_KV, tKgK_rope, tKsK_rope);
  };

  TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_slice(tidx);
  // GEMM-I: S = Q@K.T
  auto tSrQ = thr_mma.partition_fragment_A(sQ);  // (MMA,MMA_M,MMA_K)
  auto tSrK = thr_mma.partition_fragment_B(sK);  // (MMA,MMA_N,MMA_K)

  auto tSrQ_rope = thr_mma.partition_fragment_A(sQ_rope);  // (MMA,MMA_M,MMA_K)
  auto tSrK_rope = thr_mma.partition_fragment_B(sK_rope);  // (MMA,MMA_N,MMA_K)

  // s2r tiled copy for qkv
  SmemTiledCopyQ smem_tiled_copy_Q;
  auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
  auto tCsQ = smem_thr_copy_Q.partition_S(sQ);
  auto tCrQ = smem_thr_copy_Q.retile_D(tSrQ);

  auto tCsQ_rope = smem_thr_copy_Q.partition_S(sQ_rope);
  auto tCrQ_rope = smem_thr_copy_Q.retile_D(tSrQ_rope);

  SmemTiledCopyK smem_tiled_copy_K;
  auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
  auto tCsK = smem_thr_copy_K.partition_S(sK);
  auto tCrK = smem_thr_copy_K.retile_D(tSrK);

  auto tCsK_rope = smem_thr_copy_K.partition_S(sK_rope);
  auto tCrK_rope = smem_thr_copy_K.retile_D(tSrK_rope);

  // S = Q@K.T
  // tSrS: (MMA,MMA_M,MMA_N)
  auto compute_qk = [&](auto& tSrS) {
    // prefetch kv
    cute::copy(smem_tiled_copy_Q, tCsQ(_, _, _0{}), tCrQ(_, _, _0{}));
    cute::copy(smem_tiled_copy_K, tCsK(_, _, _0{}), tCrK(_, _, _0{}));

    CUTE_UNROLL
    for (int ki = 0; ki < size<2>(tSrQ); ++ki) {
      // prefetch next kv
      if (ki != size<2>(tSrQ) - 1) {
        const auto next_ki = ki + 1;
        cute::copy(smem_tiled_copy_Q, tCsQ(_, _, next_ki), tCrQ(_, _, next_ki));
        cute::copy(smem_tiled_copy_K, tCsK(_, _, next_ki), tCrK(_, _, next_ki));
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
  auto tOrVt = thr_mma.partition_fragment_B(sVt);  // (MMA,MMA_K,MMA_N)

  SmemTiledCopyVt smem_tiled_copy_Vt;
  auto smem_thr_copy_Vt = smem_tiled_copy_Vt.get_thread_slice(tidx);
  auto tCsVt = smem_thr_copy_Vt.partition_S(sVt);
  auto tCrVt = smem_thr_copy_Vt.retile_D(tOrVt);

  // O = softmax(S)*V
  // tSrS: (MMA,MMA_M,MMA_N)
  // tOrAccO: (MMA,MMA_M,MMA_K)
  auto compute_sv = [&](const auto& tSrS, auto& tOrO) {
    // cast scores from Accumulator to Element
    auto tSrS_ = make_tensor_like<DType>(tSrS);
    fast_cast(tSrS, tSrS_);

    // convert layout from gemm-I C to gemm-II A
    auto tOrS = make_tensor(tSrS_.data(), Layout::to_mma_a(tSrS_.layout()));

    // prefetch V^t
    cute::copy(smem_tiled_copy_Vt, tCsVt(_, _, _0{}), tCrVt(_, _, _0{}));
    CUTE_UNROLL
    for (int ki = 0; ki < size<2>(tOrS); ++ki) {
      // prefetch next V^t
      if (ki != size<2>(tOrS) - 1) {
        const auto next_ki = ki + 1;
        cute::copy(
            smem_tiled_copy_Vt, tCsVt(_, _, next_ki), tCrVt(_, _, next_ki));
      }
      cute::gemm(tiled_mma, tOrS(_, _, ki), tOrVt(_, _, ki), tOrO);
    }
  };

  // tOrO: (MMA,MMA_M,MMA_K)
  auto epilogue = [&](const auto& tOrO) {
    // write output to gmem
    // 1> cast output from ElementAccumulator to Element
    auto tOrO_ = make_tensor_like<DType>(tOrO);
    fast_cast(tOrO, tOrO_);

    auto sO = make_tensor(sQ.data(), SmemLayoutO{});
    // 2. copy output from reg to smem (reuse sQ)
    {
      SmemTiledCopyO smem_tiled_copy_O;
      auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
      auto tCrO = smem_thr_copy_O.retile_S(tOrO_);
      auto tCsO = smem_thr_copy_O.partition_D(sO);
      cute::copy(smem_tiled_copy_O, tCrO, tCsO);
    }

    // 3. copy output from smem to gmem
    {
      GmemTiledCopyO gmem_tiled_copy_O;
      auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);

      auto tCsO = gmem_thr_copy_O.partition_S(sO);  // (CPY,CPY_M,CPY_K)
      auto tCgO = gmem_thr_copy_O.partition_D(gO);  // (CPY,CPY_M,CPY_K)

      // wait for smem copy done before gmem copy
      __syncthreads();
      cute::copy(gmem_tiled_copy_O, tCsO, tCgO);
    }
  };

  // output accumulator, (MMA,MMA_M,MMA_K)
  auto tOrO = partition_fragment_C(tiled_mma, Shape<_BLK_M, _HEAD_DIM>{});
  auto tOrO_mn = make_tensor(tOrO.data(), Layout::to_rowcol(tOrO.layout()));
  clear(tOrO);

  const int n_block_min = 0;
  const int n_block_max = cute::ceil_div(kv_len, kBlockN);

  // ###############  Prologue  ###############
  // produce query: [] => [q]
  produce_q();
  // produce q_rope: [q] => [q, q_rope]
  produce_q_rope();
  cp_async_fence();
  // produce key: [q, q_rope] => [q, q_rope, kv]
  produce_kv(0);
  // produce k_rope: [q, q_rope, kv] => [q, q_rope, kv, k_rope]
  produce_k_rope(0);
  cp_async_fence();

  // ###############  Mainloop  ###############
  constexpr int kMMA_M = size<1>(tOrO);
  using Softmax = OnlineSoftmax<kRowsPerMMA * kMMA_M>;
  Softmax softmax(sm_scale_log2);

  CUTE_NO_UNROLL
  for (int ni = n_block_min; ni < n_block_max; ++ni) {
    // attention score accumulator, (MMA,MMA_M,MMA_N)
    auto tSrS = partition_fragment_C(tiled_mma, Shape<_BLK_M, _BLK_N>{});
    auto tSrS_mn = make_tensor(tSrS.data(), Layout::to_rowcol(tSrS.layout()));
    clear(tSrS);

    // wait key, queue: [q, q_rope, kv, k_rope] => []
    cp_async_wait<0>();
    __syncthreads();

    // 1> S = Q@K.T
    compute_qk(tSrS);

    // 2> S += Q_rope@K_rope.T
    compute_qk_rope(tSrS);

    softmax.rescale(tSrS_mn, tOrO_mn);

    // 3> O = softmax(S)*V
    compute_sv(tSrS, tOrO);

    // produce next key: [] => [kv, k_rope]
    if (ni != n_block_max - 1) {
      produce_kv(ni + 1);
      produce_k_rope(ni + 1);
    }
    cp_async_fence();
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