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
  using T = typename Traits::T;
  using SmemLayoutQ = typename Traits::SmemLayoutQ;
  using SmemLayoutK = typename Traits::SmemLayoutKV;
  using SmemLayoutV = typename Traits::SmemLayoutKV;
  using SmemLayoutVt = typename Traits::SmemLayoutVt;

  using SmemLayoutO = typename Traits::SmemLayoutO;
  using SmemCopyAtom = typename Traits::SmemCopyAtom;
  using SmemCopyAtomO = typename Traits::SmemCopyAtomO;
  using GmemTiledCopyQKV = typename Traits::GmemTiledCopyQKV;
  using GmemTiledCopyO = typename Traits::GmemTiledCopyO;
  using SmemCopyAtomTransposed = typename Traits::SmemCopyAtomTransposed;
  using TiledMMA = typename Traits::TiledMMA;

  const int m_block = blockIdx.x;
  const int base_id = blockIdx.y;
  const int tidx = threadIdx.x;

  constexpr int kBlockM = Traits::kBlockM;
  constexpr int kBlockN = Traits::kBlockN;
  constexpr int kHeadDim = Traits::kHeadDim;

  // ProblemShape
  // TODO: support non-contiguous layout
  const int offset = base_id * h_stride;
  // (q_len, head_dim)
  auto Q = make_tensor(make_gmem_ptr(static_cast<T*>(q) + offset),
                       make_shape(q_len, Int<kHeadDim>{}),
                       make_stride(Int<kHeadDim>{}, _1{}));
  auto O = make_tensor(make_gmem_ptr(static_cast<T*>(o) + offset),
                       make_shape(q_len, Int<kHeadDim>{}),
                       make_stride(Int<kHeadDim>{}, _1{}));
  // (kv_len, head_dim)
  auto K = make_tensor(make_gmem_ptr(static_cast<T*>(k) + offset),
                       make_shape(kv_len, Int<kHeadDim>{}),
                       make_stride(Int<kHeadDim>{}, _1{}));
  auto V = make_tensor(make_gmem_ptr(static_cast<T*>(v) + offset),
                       make_shape(kv_len, Int<kHeadDim>{}),
                       make_stride(Int<kHeadDim>{}, _1{}));

  // CTA/Block Shape
  // (BLK_M, head_dim)
  Tensor gQ = local_tile(
      Q, make_tile(Int<kBlockM>{}, Int<kHeadDim>{}), make_coord(m_block, _));
  Tensor gO = local_tile(
      O, make_tile(Int<kBlockM>{}, Int<kHeadDim>{}), make_coord(m_block, _));

  // (BLK_N, head_dim)
  Tensor gK = local_tile(
      K, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(0, _));
  Tensor gV = local_tile(
      V, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(0, _));

  // Smem
  extern __shared__ char smem[];
  T* q_smem = static_cast<T*>(smem);
  T* k_smem = q_smem + cosize(SmemLayoutQ{});
  T* v_smem = k_smem + cosize(SmemLayoutK{});

  // (BLK_M, BLK_K), k-major
  Tensor sQ = make_tensor(make_smem_ptr(q_smem), SmemLayoutQ{});
  // (BLK_N, BLK_K), k-major
  Tensor sK = make_tensor(make_smem_ptr(k_smem), SmemLayoutK{});
  Tensor sV = make_tensor(make_smem_ptr(v_smem), SmemLayoutV{});

  // Tensor for V^t; used in GEMM-II.
  // (BLK_K, BLK_N), k-major
  Tensor sVt = make_tensor(make_smem_ptr(v_smem), SmemLayoutVt{});

  // rmem for mma
  TiledMMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(tidx);
  // gemm-I
  auto tSrQ = thr_mma.partition_fragment_A(sQ);  // (MMA,MMA_M,MMA_K)
  auto tSrK = thr_mma.partition_fragment_B(sK);  // (MMA,MMA_N,MMA_K)

  // gemm-II
  auto tOrVt = thr_mma.partition_fragment_B(sVt);  // (MMA,MMA_K,MMA_N)

  // g2s tiled copy for qkv
  GmemTiledCopyQKV gmem_tiled_copy_QKV;
  auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);
  auto tQgQ = gmem_thr_copy_QKV.partition_S(gQ(_, _, 0));
  auto tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
  auto tKgK = gmem_thr_copy_QKV.partition_S(gK(_, _, 0));
  auto tKsK = gmem_thr_copy_QKV.partition_D(sK);
  auto tVgV = gmem_thr_copy_QKV.partition_S(gV(_, _, 0));
  auto tVsV = gmem_thr_copy_QKV.partition_D(sV);

  // s2r tiled copy for qkv
  auto smem_tiled_copy_Q = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
  auto tSsQ = smem_thr_copy_Q.partition_S(sQ);
  auto tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);

  auto smem_tiled_copy_K = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
  auto tSsK = smem_thr_copy_K.partition_S(sK);
  auto tSrK_copy_view = smem_thr_copy_K.retile_D(tSrK);

  auto smem_tiled_copy_V =
      make_tiled_copy_B(SmemCopyAtomTransposed{}, tiled_mma);
  auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
  auto tOsVt = smem_thr_copy_V.partition_S(sVt);
  auto tOrVt_copy_view = smem_thr_copy_V.retile_D(tOrVt);

  // ###############  Prologue  ###############

  // produce q
  cute::copy(gmem_tiled_copy_QKV, tQgQ, tQsQ);
  cp_async_fence();

  // produce k
  cute::copy(gmem_tiled_copy_QKV, tKgK, tKsK);
  cp_async_fence();

  // multiply sm scale
  // wait q: [q, k] => [k]
  cp_async_wait<0>();
  __syncthreads();

  // apply sm_scale
  // TODO: use thread parallelism
  for (int i = 0; i < size(tQsQ); ++i) {
    tQsQ(i) = T(tQsQ(i) * sm_scale);
  }

  // Final output fragment
  auto tOrO =
      partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});
  clear(tOrO);

  // reshape for iteration
  auto ol = logical_divide(tOrO.layout(), Shape<_2>{});
  auto rAccOut_new_layout = make_layout(make_layout(get<0, 1>(ol), get<1>(ol)),
                                        make_layout(get<0, 0>(ol), get<2>(ol)));
  auto tOrO_rc = make_tensor(tOrO.data(), rAccOut_new_layout);

  // RowsPerThread = #rows_per_MMA * #MMA_M
  constexpr int RowsPerThread = 2 * size<1>(tOrO);
  OnlineSoftmax<RowsPerThread> softmax;

  // ###############  Mainloop  ###############

  const int n_block_min = 0;
  const int n_block_max = cute::ceil_div(kv_len, kBlockN);
  CUTE_NO_UNROLL
  for (int ni = n_block_min; ni < n_block_max; ++ni) {
    // attention score
    // (MMA=4,MMA_M,MMA_N) (fp32)
    auto tSrS =
        partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
    // clear attention score
    clear(tSrS);

    // wait k, queue: [q, k] => []
    cp_async_wait<0>();
    __syncthreads();

    // produce v, [] => [v]
    {
      gV = local_tile(
          V, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(ni, _));
      tVgV = gmem_thr_copy_QKV.partition_S(gV(_, _, 0));
      cute::copy(gmem_tiled_copy_QKV, tVgV, tVsV);
    }
    cp_async_fence();

    // 1> S = Q@K.T
    CUTE_UNROLL
    for (int ki = 0; ki < size<2>(tSrQ); ++ki) {
      cute::copy(smem_tiled_copy_Q, tSsQ(_, _, ki), tSrQ_copy_view(_, _, ki));
      cute::copy(smem_tiled_copy_K, tSsK(_, _, ki), tSrK_copy_view(_, _, ki));
      cute::gemm(tiled_mma, tSrQ(_, _, ki), tSrK(_, _, ki), tSrS);
    }

    // reshape for iteration
    auto sl = logical_divide(tSrS.layout(), Shape<_2>{});
    auto rAccScore_new_layout =
        make_layout(make_layout(get<0, 1>(sl), get<1>(sl)),
                    make_layout(get<0, 0>(sl), get<2>(sl)));
    auto tSrS_rc = make_tensor(tSrS.data(), rAccScore_new_layout);

    softmax.rescale(tSrS_rc, tOrO_rc);

    // wait v, [v] => []
    cp_async_wait<0>();
    __syncthreads();

    // produce next k: [] => [k]
    if (ni != n_block_max - 1) {
      gK = local_tile(
          K, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(ni + 1, _));
      tKgK = gmem_thr_copy_QKV.partition_S(gK(_, _, 0));
      cute::copy(gmem_tiled_copy_QKV, tKgK, tKsK);
    }
    cp_async_fence();

    // 2> O = softmax(S)*V

    // cast scores from fp32 to fp16
    auto tSrS_T = make_tensor_like<T>(tSrS);
    CUTE_UNROLL
    for (int i = 0; i < size(tSrS); ++i) {
      tSrS_T(i) = static_cast<T>(tSrS(i));
    }
    // convert layout from gemm-I C to gemm-II A
    auto l = logical_divide(tSrS_T.layout(), Shape<X, X, _2>{});
    auto scores_new_layout = make_layout(
        make_layout(get<0>(l), get<2, 0>(l)), get<1>(l), get<2, 1>(l));
    auto tOrS = make_tensor(tSrS_T.data(), scores_new_layout);

    CUTE_UNROLL
    for (int ki = 0; ki < size<2>(tOrS); ++ki) {
      cute::copy(smem_tiled_copy_V, tOsVt(_, _, ki), tOrVt_copy_view(_, _, ki));
      cute::gemm(tiled_mma, tOrS(_, _, ki), tOrVt(_, _, ki), tOrO);
    }
  }

  // ###############  Epilogue  ###############

  // normalize output: o /= rowsum
  softmax.finalize(tOrO_rc);

  // write output to gmem
  // 1> covernt output from fp32 to fp16
  auto tOrO_T = make_tensor_like<T>(tOrO);
  CUTE_UNROLL
  for (int si = 0; si < size(tOrO); ++si) {
    tOrO_T(si) = static_cast<T>(tOrO(si));
  }

  // 2. copy output from reg to smem
  auto sO = make_tensor(sQ.data(), SmemLayoutO{});
  auto smem_tiled_copy_O = make_tiled_copy_C(SmemCopyAtomO{}, tiled_mma);
  auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
  // ((Atom,AtomNum),MMA_M,MMA_N)
  auto taccOrO = smem_thr_copy_O.retile_S(tOrO_T);
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
}

}  // namespace llm