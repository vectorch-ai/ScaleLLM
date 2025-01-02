#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cute/tensor.hpp>

#include "online_softmax.cuh"

namespace llm {

template <typename Traits>
__global__ void mha_kernel_sm80(void* o,
                                const void* q,
                                const void* k,
                                const void* v,
                                int h_stride,
                                int q_len,
                                int kv_len,
                                float sm_scale) {
  using namespace cute;

  // type alias
  using Element = typename Traits::Element;
  using BLK_M = typename Traits::BLK_M;
  using BLK_N = typename Traits::BLK_N;
  using BLK_K = typename Traits::BLK_K;
  using HEAD_DIM = typename Traits::HEAD_DIM;

  using TiledMMA = typename Traits::TiledMMA;
  using Convertor = typename Traits::FragmentConvertor;

  using SmemLayoutQ = typename Traits::SmemLayoutQ;
  using SmemLayoutK = typename Traits::SmemLayoutKV;
  using SmemLayoutV = typename Traits::SmemLayoutKV;
  using SmemLayoutVt = typename Traits::SmemLayoutVt;
  using SmemLayoutO = typename Traits::SmemLayoutO;
  using GmemTiledCopyQKV = typename Traits::GmemTiledCopyQKV;
  using GmemTiledCopyO = typename Traits::GmemTiledCopyO;

  using SmemTiledCopyQ = typename Traits::SmemTiledCopyQ;
  using SmemTiledCopyK = typename Traits::SmemTiledCopyK;
  using SmemTiledCopyVT = typename Traits::SmemTiledCopyVT;
  using SmemTiledCopyO = typename Traits::SmemTiledCopyO;

  const int m_block = blockIdx.x;
  const int base_id = blockIdx.y;
  const int tidx = threadIdx.x;

  // ProblemShape
  // TODO: support non-contiguous layout
  const int offset = base_id * h_stride;
  // (q_len, head_dim)
  auto Q = make_tensor(make_gmem_ptr((Element*)q + offset),
                       make_shape(q_len, HEAD_DIM{}),
                       make_stride(HEAD_DIM{}, _1{}));
  auto O = make_tensor(make_gmem_ptr((Element*)o + offset),
                       make_shape(q_len, HEAD_DIM{}),
                       make_stride(HEAD_DIM{}, _1{}));
  // (kv_len, head_dim)
  auto K = make_tensor(make_gmem_ptr((Element*)k + offset),
                       make_shape(kv_len, HEAD_DIM{}),
                       make_stride(HEAD_DIM{}, _1{}));
  auto V = make_tensor(make_gmem_ptr((Element*)v + offset),
                       make_shape(kv_len, HEAD_DIM{}),
                       make_stride(HEAD_DIM{}, _1{}));

  // CTA/Block Shape
  // (BLK_M, head_dim)
  Tensor gQ =
      local_tile(Q, make_tile(BLK_M{}, HEAD_DIM{}), make_coord(m_block, _));
  Tensor gO =
      local_tile(O, make_tile(BLK_M{}, HEAD_DIM{}), make_coord(m_block, _));

  // (BLK_N, head_dim)
  Tensor gK = local_tile(K, make_tile(BLK_N{}, HEAD_DIM{}), make_coord(0, _));
  Tensor gV = local_tile(V, make_tile(BLK_N{}, HEAD_DIM{}), make_coord(0, _));

  // Smem
  extern __shared__ char smem[];
  Element* q_smem = (Element*)smem;
  Element* k_smem = q_smem + cosize(SmemLayoutQ{});
  Element* v_smem = k_smem + cosize(SmemLayoutK{});

  // (BLK_M, BLK_K), k-major
  Tensor sQ = make_tensor(make_smem_ptr(q_smem), SmemLayoutQ{});
  // (BLK_N, BLK_K), k-major
  Tensor sK = make_tensor(make_smem_ptr(k_smem), SmemLayoutK{});
  Tensor sV = make_tensor(make_smem_ptr(v_smem), SmemLayoutV{});

  // Tensor for V^t; used in GEMM-II.
  // (BLK_K, BLK_N), k-major
  Tensor sVt = make_tensor(make_smem_ptr(v_smem), SmemLayoutVt{});

  // Tiled Copy
  // g2s tiled copy for qkv
  GmemTiledCopyQKV gmem_tiled_copy_QKV;
  auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);
  auto tQgQ = gmem_thr_copy_QKV.partition_S(gQ(_, _, 0));
  auto tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
  auto tKgK = gmem_thr_copy_QKV.partition_S(gK(_, _, 0));
  auto tKsK = gmem_thr_copy_QKV.partition_D(sK);
  auto tVgV = gmem_thr_copy_QKV.partition_S(gV(_, _, 0));
  auto tVsV = gmem_thr_copy_QKV.partition_D(sV);

  auto produce_q = [&]() {
    cute::copy(gmem_tiled_copy_QKV, tQgQ, tQsQ);
    cp_async_fence();
  };

  auto produce_k = [&](int ni) {
    gK = local_tile(K, make_tile(BLK_N{}, HEAD_DIM{}), make_coord(ni, _));
    tKgK = gmem_thr_copy_QKV.partition_S(gK(_, _, 0));
    cute::copy(gmem_tiled_copy_QKV, tKgK, tKsK);
    cp_async_fence();
  };

  auto produce_v = [&](int ni) {
    gV = local_tile(V, make_tile(BLK_N{}, HEAD_DIM{}), make_coord(ni, _));
    tVgV = gmem_thr_copy_QKV.partition_S(gV(_, _, 0));
    cute::copy(gmem_tiled_copy_QKV, tVgV, tVsV);
    cp_async_fence();
  };

  TiledMMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(tidx);
  // GEMM-I: S = Q@K.T
  auto tSrQ = thr_mma.partition_fragment_A(sQ);  // (MMA,MMA_M,MMA_K)
  auto tSrK = thr_mma.partition_fragment_B(sK);  // (MMA,MMA_N,MMA_K)
  auto tSrAccS = partition_fragment_C(
      tiled_mma, Shape<BLK_M, BLK_N>{});  // (MMA,MMA_M,MMA_N)

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
  auto compute_qk = [&]() {
    CUTE_UNROLL
    for (int ki = 0; ki < size<2>(tSrQ); ++ki) {
      cute::copy(smem_tiled_copy_Q, tSsQ(_, _, ki), tSrQ_copy_view(_, _, ki));
      cute::copy(smem_tiled_copy_K, tSsK(_, _, ki), tSrK_copy_view(_, _, ki));
      cute::gemm(tiled_mma, tSrQ(_, _, ki), tSrK(_, _, ki), tSrAccS);
    }
  };

  // GEMM-II: O = softmax(S)@V
  auto tOrVt = thr_mma.partition_fragment_B(sVt);  // (MMA,MMA_K,MMA_N)
  auto tOrAccO = partition_fragment_C(
      tiled_mma, Shape<BLK_M, HEAD_DIM>{});  // (MMA,MMA_M,MMA_K)

  SmemTiledCopyVT smem_tiled_copy_Vt;
  auto smem_thr_copy_Vt = smem_tiled_copy_Vt.get_thread_slice(tidx);
  auto tOsVt = smem_thr_copy_Vt.partition_S(sVt);
  auto tOrVt_copy_view = smem_thr_copy_Vt.retile_D(tOrVt);

  // O = softmax(S)*V
  auto compute_sv = [&]() {
    // cast scores from Accumulator to Element
    auto tSrS = make_tensor_like<Element>(tSrAccS);
    CUTE_UNROLL
    for (int i = 0; i < size(tSrAccS); ++i) {
      tSrS(i) = static_cast<Element>(tSrAccS(i));
    }

    // convert layout from gemm-I C to gemm-II A
    auto tOrS = Convertor::to_mma_a(tSrS);

    CUTE_UNROLL
    for (int ki = 0; ki < size<2>(tOrS); ++ki) {
      cute::copy(
          smem_tiled_copy_Vt, tOsVt(_, _, ki), tOrVt_copy_view(_, _, ki));
      cute::gemm(tiled_mma, tOrS(_, _, ki), tOrVt(_, _, ki), tOrAccO);
    }
  };

  // write output to gmem
  auto epilogue = [&]() {
    // 1> covernt output from ElementAccumulator to Element
    auto tOrO = make_tensor_like<Element>(tOrAccO);
    CUTE_UNROLL
    for (int si = 0; si < size(tOrAccO); ++si) {
      tOrO(si) = static_cast<Element>(tOrAccO(si));
    }

    // 2. copy output from reg to smem (reuse sQ)
    auto sO = make_tensor(sQ.data(), SmemLayoutO{});

    SmemTiledCopyO smem_tiled_copy_O;
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
    // ((Atom,AtomNum),MMA_M,MMA_N)
    auto taccOrO = smem_thr_copy_O.retile_S(tOrO);
    // ((Atom,AtomNum),PIPE_M,PIPE_N)
    auto taccOsO = smem_thr_copy_O.partition_D(sO);
    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

    // 3. copy output from smem to gmem
    GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    // ((Atom,AtomNum),ATOM_M,ATOM_N)
    auto tOsO = gmem_thr_copy_O.partition_S(sO);
    auto tOgO = gmem_thr_copy_O.partition_D(gO(_, _, 0));

    // wait for smem copy before copy to gmem
    __syncthreads();
    cute::copy(gmem_tiled_copy_O, tOsO, tOgO);
  };

  // ###############  Prologue  ###############

  // produce q: [] => [q]
  produce_q();
  // produce k: [q] => [q, k]
  produce_k(0);

  // ###############  Mainloop  ###############

  // reshape for iterating over rows and columns
  auto tOrAccO_rc_view = Convertor::to_rowcol(tOrAccO);
  auto tSrAccS_rc_view = Convertor::to_rowcol(tSrAccS);

  // clear output
  clear(tOrAccO);

  // RowsPerThread = #rows_per_MMA * #MMA_M
  constexpr int RowsPerThread = 2 * size<1>(tOrAccO);
  OnlineSoftmax<RowsPerThread> softmax(sm_scale);

  const int n_block_min = 0;
  const int n_block_max = cute::ceil_div(kv_len, BLK_N{});

  CUTE_NO_UNROLL
  for (int ni = n_block_min; ni < n_block_max; ++ni) {
    // clear attention score for each block
    clear(tSrAccS);

    // wait k, queue: [q, k] => []
    cp_async_wait<0>();
    __syncthreads();

    // produce v, [] => [v]
    produce_v(ni);

    // 1> S = Q@K.T
    compute_qk();

    // apply softmax and rescale
    softmax.rescale(tSrAccS_rc_view, tOrAccO_rc_view);

    // wait v, [v] => []
    cp_async_wait<0>();
    __syncthreads();

    // produce next k: [] => [k]
    if (ni != n_block_max - 1) {
      produce_k(ni + 1);
    }

    // 2> O = softmax(S)*V
    compute_sv();
  }

  // ###############  Epilogue  ###############

  // normalize output: o /= rowsum
  softmax.finalize(tOrAccO_rc_view);

  // write output to gmem
  epilogue();
}

}  // namespace llm