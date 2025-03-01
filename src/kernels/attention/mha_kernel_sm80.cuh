#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cute/layout.hpp>
#include <cute/tensor.hpp>

#include "cute/config.hpp"
#include "cute/container/array_aligned.hpp"
#include "cute_extensions.cuh"
#include "fast_cast.cuh"
#include "layout_convertor.h"
#include "mask.h"
#include "mha_tile.h"
#include "online_softmax.cuh"

namespace llm {

template <typename Traits>
struct MHASharedStorage {
  using DType = typename Traits::DType;
  using SmemLayoutQ = typename Traits::SmemLayoutQ;
  using SmemLayoutK = typename Traits::SmemLayoutK;
  using SmemLayoutV = typename Traits::SmemLayoutV;
  using SmemLayoutVt = typename Traits::SmemLayoutVt;
  using SmemLayoutO = typename Traits::SmemLayoutO;

  union {
    union {
      cute::array_aligned<DType, cute::cosize_v<SmemLayoutQ>> q_smem;
      struct {
        cute::array_aligned<DType, cute::cosize_v<SmemLayoutK>> k_smem;
        union {
          cute::array_aligned<DType, cute::cosize_v<SmemLayoutV>> v_smem;
          cute::array_aligned<DType, cute::cosize_v<SmemLayoutVt>> vt_smem;
        };
      };
    };

    cute::array_aligned<DType, cute::cosize_v<SmemLayoutO>> o_smem;
  };
};

template <typename Traits,
          typename Params,
          bool EVEN_K,
          bool ALIBI,
          bool SOFT_CAP,
          bool LOCAL>
__global__ __launch_bounds__(Traits::kThreadNum) void mha_kernel_sm80(
    __grid_constant__ const Params params) {
  using namespace cute;

  constexpr int kBlockM = Traits::kBlockM;
  constexpr int kBlockN = Traits::kBlockN;
  constexpr int kHeadDim = Traits::kHeadDim;
  constexpr int kRowsPerMMA = Traits::kRowsPerMMA;

  using _BLK_M = Int<kBlockM>;
  using _BLK_N = Int<kBlockN>;
  using _HEAD_DIM = Int<kHeadDim>;

  // type alias
  using DType = typename Traits::DType;

  using TiledMma = typename Traits::TiledMma;

  using SmemLayoutQ = typename Traits::SmemLayoutQ;
  using SmemLayoutK = typename Traits::SmemLayoutK;
  using SmemLayoutV = typename Traits::SmemLayoutV;
  using SmemLayoutVt = typename Traits::SmemLayoutVt;
  using SmemLayoutO = typename Traits::SmemLayoutO;
  using SharedStorage = MHASharedStorage<Traits>;

  using GmemTiledCopyQ = typename Traits::GmemTiledCopyQ;
  using GmemTiledCopyKV = typename Traits::GmemTiledCopyKV;
  using GmemTiledCopyO = typename Traits::GmemTiledCopyO;

  using SmemTiledCopyQ = typename Traits::SmemTiledCopyQ;
  using SmemTiledCopyK = typename Traits::SmemTiledCopyK;
  using SmemTiledCopyVt = typename Traits::SmemTiledCopyVt;
  using SmemTiledCopyO = typename Traits::SmemTiledCopyO;

  const int m_block_idx = blockIdx.x;
  const int batch_idx = blockIdx.y;
  const int kv_head_idx = blockIdx.z;
  const int tidx = threadIdx.x;

  // preprocess input parameters
  const int head_dim = params.head_dim;
  const float logits_soft_cap = params.logits_soft_cap;
  const float sm_scale = params.sm_scale;
  const float sm_scale_log2 = params.sm_scale_log2;

  const auto& group_size = params.group_size;

  // ProblemShape
  // (q_packed_len, HEAD_DIM)
  MHATile<Params> tile(params, batch_idx, kv_head_idx);
  auto [Q, O] = tile.template get_qo_tile<DType>();
  // (kv_len, HEAD_DIM)
  auto [K, V] = tile.template get_kv_tile<DType>();

  const int q_packed_len = size<0>(Q);
  const int q_len = q_packed_len / group_size;
  const int kv_len = size<0>(K);

  if (m_block_idx * kBlockM >= q_packed_len) {
    // m out of bound, return
    return;
  }

  const int sliding_window = LOCAL ? params.sliding_window : kv_len;

  // Gmem
  // (BLK_M, HEAD_DIM)
  Tensor gQ =
      local_tile(Q, Shape<_BLK_M, _HEAD_DIM>{}, make_coord(m_block_idx, _0{}));
  Tensor gO =
      local_tile(O, Shape<_BLK_M, _HEAD_DIM>{}, make_coord(m_block_idx, _0{}));
  // (BLK_N, HEAD_DIM, n)
  Tensor gK = local_tile(K, Shape<_BLK_N, _HEAD_DIM>{}, make_coord(_, _0{}));
  Tensor gV = local_tile(V, Shape<_BLK_N, _HEAD_DIM>{}, make_coord(_, _0{}));

  // Smem
  extern __shared__ char smem[];
  auto& ss = *reinterpret_cast<SharedStorage*>(smem);

  // (BLK_M, HEAD_DIM), k-major
  Tensor sQ = make_tensor(make_smem_ptr(ss.q_smem.data()), SmemLayoutQ{});
  // (BLK_N, HEAD_DIM), k-major
  Tensor sK = make_tensor(make_smem_ptr(ss.k_smem.data()), SmemLayoutK{});
  Tensor sV = make_tensor(make_smem_ptr(ss.v_smem.data()), SmemLayoutV{});

  // Tensor for V^t; used in GEMM-II.
  // (HEAD_DIM, BLK_N), m-major
  Tensor sVt = make_tensor(make_smem_ptr(ss.vt_smem.data()), SmemLayoutVt{});

  // (BLK_M, HEAD_DIM)
  Tensor sO = make_tensor(make_smem_ptr(ss.o_smem.data()), SmemLayoutO{});

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

  auto produce_query = [&]() {
    auto tQgQ = gmem_thr_copy_Q.partition_S(gQ);
    auto tQsQ = gmem_thr_copy_Q.partition_D(sQ);
    auto max_coord = make_coord(q_packed_len - m_block_idx * kBlockM, head_dim);
    safe_copy</*EVEN_MN=*/false, EVEN_K, /*ZFILL_MN=*/true, /*ZFILL_K=*/true>(
        gmem_tiled_copy_Q, tQgQ, tQsQ, tQcQ, max_coord);
  };

  // (BLK_N, HEAD_DIM) -> (blk_n, head_dim)
  Tensor cKV = make_identity_tensor(Shape<_BLK_N, _HEAD_DIM>{});
  Tensor tKVcKV = gmem_thr_copy_KV.partition_S(cKV);

  Tensor tKsK = gmem_thr_copy_KV.partition_D(sK);
  auto produce_key = [&](int ni) {
    auto tKgK = gmem_thr_copy_KV.partition_S(gK(_, _, ni));
    auto max_coord = make_coord(kv_len - ni * kBlockN, head_dim);
    // skip ZFILL_MN for key since Mask will mask out oob with -inf
    safe_copy</*EVEN_MN=*/false, EVEN_K, /*ZFILL_MN=*/false, /*ZFILL_K=*/true>(
        gmem_tiled_copy_KV, tKgK, tKsK, tKVcKV, max_coord);
  };

  // produce key without oob handling
  auto produce_key_no_oob = [&](int ni) {
    auto tKgK = gmem_thr_copy_KV.partition_S(gK(_, _, ni));
    auto max_coord = make_coord(kv_len - ni * kBlockN, head_dim);
    safe_copy</*EVEN_MN=*/true, EVEN_K, /*ZFILL_MN=*/false, /*ZFILL_K=*/false>(
        gmem_tiled_copy_KV, tKgK, tKsK, tKVcKV, max_coord);
  };

  Tensor tVsV = gmem_thr_copy_KV.partition_D(sV);
  auto produce_value = [&](int ni) {
    auto tVgV = gmem_thr_copy_KV.partition_S(gV(_, _, ni));
    auto max_coord = make_coord(kv_len - ni * kBlockN, head_dim);
    // skipping ZFILL_MN for v may cause nan issue
    safe_copy</*EVEN_MN=*/false, EVEN_K, /*ZFILL_MN=*/true, /*ZFILL_K=*/true>(
        gmem_tiled_copy_KV, tVgV, tVsV, tKVcKV, max_coord);
  };

  // produce value without oob handling
  auto produce_value_no_oob = [&](int ni) {
    auto tVgV = gmem_thr_copy_KV.partition_S(gV(_, _, ni));
    auto max_coord = make_coord(kv_len - ni * kBlockN, head_dim);
    safe_copy</*EVEN_MN=*/true, EVEN_K, /*ZFILL_MN=*/false, /*ZFILL_K=*/false>(
        gmem_tiled_copy_KV, tVgV, tVsV, tKVcKV, max_coord);
  };

  TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_slice(tidx);
  // GEMM-I: S = Q@K.T
  auto tSrQ = thr_mma.partition_fragment_A(sQ);  // (MMA,MMA_M,MMA_K)
  auto tSrK = thr_mma.partition_fragment_B(sK);  // (MMA,MMA_N,MMA_K)

  // s2r tiled copy for qkv
  // copy query to rmem
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
    // prefetch key
    cute::copy(smem_tiled_copy_K, tSsK(_, _, _0{}), tSrK_copy_view(_, _, _0{}));

    CUTE_UNROLL
    for (int ki = 0; ki < size<2>(tSrQ); ++ki) {
      // prefetch next key
      if (ki != size<2>(tSrQ) - 1) {
        const auto next_ki = ki + 1;
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
    auto tOrS =
        make_tensor(tSrS.data(), LayoutConvertor::to_mma_a(tSrS.layout()));

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

    // 2. copy output from reg to smem
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

    auto max_coord = make_coord(q_packed_len - m_block_idx * kBlockM, head_dim);
    safe_copy</*EVEN_MN=*/false, EVEN_K, /*ZFILL_MN=*/false, /*ZFILL_K=*/false>(
        gmem_tiled_copy_O, tOsO, tOgO, tOcO, max_coord);
  };

  // output accumulator, (MMA,MMA_M,MMA_K)
  auto tOrO = partition_fragment_C(tiled_mma, Shape<_BLK_M, _HEAD_DIM>{});
  auto tOrO_mn =
      make_tensor(tOrO.data(), LayoutConvertor::to_mn(tOrO.layout()));
  clear(tOrO);

  const int diagonal = (m_block_idx * kBlockM) / group_size + kv_len - q_len;
  // process kv in range: [kv_idx_min, kv_idx_max)
  const int kv_idx_min = std::max(0, diagonal - sliding_window);
  const int kv_idx_max = std::min(kv_len, diagonal + kBlockM);
  const int n_block_min = LOCAL ? kv_idx_min / kBlockN : 0;
  const int n_block_max = cute::ceil_div(kv_idx_max, kBlockN);

  if (n_block_min >= n_block_max) {
    // write output to gmem
    epilogue(tOrO);
    return;
  }

  auto apply_logits_soft_cap = [&](auto& tSrAccS) {
    if constexpr (SOFT_CAP) {
      CUTE_UNROLL
      for (int i = 0; i < size(tSrAccS); ++i) {
        tSrAccS(i) = tanh(tSrAccS(i) * logits_soft_cap);
      }
    }
  };

  // ###############  Prologue  ###############
  // produce query: [] => [q]
  produce_query();
  cp_async_fence();

  // wait g2s copy done for query
  cp_async_wait<0>();
  __syncthreads();

  // copy query from smem to rmem
  cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
  // wait s2r copy done for query
  __syncthreads();

  // produce key: [q] => [q, k]
  produce_key(n_block_max - 1);
  cp_async_fence();

  // ###############  Mainloop  ###############
  constexpr int n_oob_mask = cute::ceil_div(kBlockM, kBlockN) + 1;
  const int n_blocks = n_block_max - n_block_min;

  // attention score accumulator, (MMA,MMA_M,MMA_N)
  auto tSrS = partition_fragment_C(tiled_mma, Shape<_BLK_M, _BLK_N>{});
  auto tSrS_mn =
      make_tensor(tSrS.data(), LayoutConvertor::to_mn(tSrS.layout()));

  // identity tensor for score accumulator
  auto tScS =
      thr_mma.partition_C(make_identity_tensor(Shape<_BLK_M, _BLK_N>{}));
  auto tScS_mn =
      make_tensor(tScS.data(), LayoutConvertor::to_mn(tScS.layout()));

  constexpr int kRowsPerThr = kRowsPerMMA * size<1>(tSrS);
  using Softmax = OnlineSoftmax<kRowsPerThr>;
  using Mask = Mask<kRowsPerThr, ALIBI, LOCAL>;

  Softmax softmax(sm_scale_log2);
  Mask mask(q_len, kv_len, group_size, sliding_window);
  if constexpr (ALIBI) {
    mask.init_alibi(tScS_mn,
                    m_block_idx * kBlockM,
                    kv_head_idx,
                    sm_scale,
                    params.alibi_slopes_ptr);
  }

  CUTE_NO_UNROLL
  for (int i = 0; i < n_blocks; ++i) {
    const int n_block_idx = n_block_max - 1 - i;
    clear(tSrS);

    // wait key, queue: [q, k] => []
    cp_async_wait<0>();
    __syncthreads();

    // produce value, [] => [v]
    if (i == 0) {
      produce_value(n_block_idx);
    } else {
      produce_value_no_oob(n_block_idx);
    }
    cp_async_fence();

    // 1> S = Q@K.T
    compute_qk(tSrS);

    // wait value, [v] => []
    cp_async_wait<0>();
    __syncthreads();

    if constexpr (SOFT_CAP) {
      apply_logits_soft_cap(tSrS);
    }

    if (i < n_oob_mask) {
      mask.apply(
          tSrS_mn, tScS_mn, m_block_idx * kBlockM, n_block_idx * kBlockN);
    } else {
      mask.apply</*OOB_MASK=*/false>(
          tSrS_mn, tScS_mn, m_block_idx * kBlockM, n_block_idx * kBlockN);
    }
    softmax.rescale(tSrS_mn, tOrO_mn);

    // produce next key: [] => [k]
    if (n_block_idx > n_block_min) {
      produce_key_no_oob(n_block_idx - 1);
    }
    cp_async_fence();

    // 2> O = softmax(S)*V
    compute_sv(tSrS, tOrO);
  }

  // ###############  Epilogue  ###############

  // normalize output: o /= rowsum
  softmax.finalize(tOrO_mn);

  // write output to gmem
  epilogue(tOrO);
}

template <typename Traits,
          typename Params,
          bool EVEN_K,
          bool ALIBI,
          bool SOFT_CAP,
          bool LOCAL>
void launch_mha_kernel_sm80(const Params& params, cudaStream_t stream) {
  const auto batch_size = params.batch_size;
  const auto n_kv_heads = params.n_kv_heads;
  const auto max_q_packed_len = params.max_q_len * params.group_size;

  const auto smem_size = sizeof(MHASharedStorage<Traits>);
  auto mha_kernel =
      mha_kernel_sm80<Traits, Params, EVEN_K, ALIBI, SOFT_CAP, LOCAL>;
  cudaFuncSetAttribute(
      mha_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  // TODO: support persistent kernels
  dim3 grid(cute::ceil_div(max_q_packed_len, Traits::kBlockM),
            batch_size,
            n_kv_heads);
  dim3 block = Traits::kThreadNum;
  mha_kernel<<<grid, block, smem_size, stream>>>(params);
}

}  // namespace llm