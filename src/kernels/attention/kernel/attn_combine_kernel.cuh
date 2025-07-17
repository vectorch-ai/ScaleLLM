#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cute/layout.hpp>
#include <cute/tensor.hpp>

#include "common/fast_cast.cuh"
#include "common/safe_copy.h"

namespace llm {

// combine attention ouputs from multiple splits:
//  output = sum(softmax(lseAccum) * outAccum)
//  lse = log(sum(exp(lseAccum))
// inputs:
//    outAccum:  [n_splits, batch, seq_len, n_heads, head_dim]
//    lseAccum:  [n_splits, batch, seq_len, n_heads]
// outputs:
//    out:      [batch, seq_len, n_heads, head_dim]
//    lse:      [batch, seq_len, n_heads]
template <typename Element,
          typename ElementAccum,
          int kHeadDim,
          int kSplits,
          int kThreads,
          bool EVEN_K,
          typename Params>
__global__ void attn_combine_kernel(__grid_constant__ const Params params) {
  using namespace cute;
  using VectorizingCopy = AutoVectorizingCopyWithAssumedAlignment<128>;

  const int tidx = threadIdx.x;
  // Grid: [batch, seq_len, n_heads]
  const int b_idx = blockIdx.x;
  const int s_idx = blockIdx.y;
  const int h_idx = blockIdx.z;

  // TODO: support variable number of splits per batch
  const int n_splits = params.n_splits;
  const int batch_size = params.batch_size;
  const int q_len = params.q_len;
  const int n_heads = params.n_heads;
  const int head_dim = params.head_dim;

  // only one split, handled in attn kernel already
  if (n_splits == 1) {
    return;
  }

  // scales[splits]
  __shared__ ElementAccum sScales[kSplits];

  // [n_splits, batch, seq_len, n_heads, kHeadDim]
  Tensor oAccum = make_tensor(
      make_gmem_ptr(reinterpret_cast<const ElementAccum*>(params.o_accum_ptr)),
      make_shape(n_splits, batch_size, q_len, n_heads, Int<kHeadDim>{}),
      append<5>(params.o_accum_stride, _1{}));
  // [n_splits, kHeadDim]
  Tensor gOaccum = oAccum(_, b_idx, s_idx, h_idx, _);

  // [n_splits, batch, seq_len, n_heads]
  Tensor lseAccum = make_tensor(
      make_gmem_ptr(
          reinterpret_cast<const ElementAccum*>(params.lse_accum_ptr)),
      make_shape(n_splits, batch_size, q_len, n_heads),
      append<4>(params.lse_accum_stride, _1{}));
  // [n_splits]
  Tensor gLseAccum = lseAccum(_, b_idx, s_idx, h_idx);

  // [batch, seq_len, n_heads, kHeadDim]
  Tensor O =
      make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)),
                  make_shape(batch_size, q_len, n_heads, Int<kHeadDim>{}),
                  append<4>(params.o_stride, _1{}));
  // [kHeadDim]
  Tensor gO = O(b_idx, s_idx, h_idx, _);

  // [batch, seq_len, n_heads]
  Tensor gLse = make_tensor(
      make_gmem_ptr(reinterpret_cast<ElementAccum*>(params.lse_ptr)),
      make_shape(batch_size, q_len, n_heads),
      append<3>(params.lse_stride, _1{}));

  // use one warp to calculate safe_softmax(lse)
  int warp_idx = cutlass::canonical_warp_idx_sync();
  if (warp_idx == 0) {
    // (SPLITS) -> (splits)
    auto cLseAccum = make_identity_tensor(Shape<Int<kSplits>>{});

    auto thr_layout = Layout<Shape<_32>>{};
    // [n_lse_per_thr]
    auto tTgLseAccum = local_partition(gLseAccum, thr_layout, tidx);
    auto tTcLseAccum = local_partition(cLseAccum, thr_layout, tidx);

    constexpr int kLsePerThr = size(tTcLseAccum);

    // local lse
    float lse[kLsePerThr];

    CUTE_UNROLL
    for (int i = 0; i < kLsePerThr; ++i) {
      const int split_idx = get<0>(tTcLseAccum(i));
      lse[i] = split_idx < n_splits ? tTgLseAccum(i) : -INFINITY;
    }

    // local lse max
    float lse_max = -INFINITY;
    CUTE_UNROLL
    for (int i = 0; i < kLsePerThr; ++i) {
      lse_max = max(lse_max, lse[i]);
    }

    // lse max within warp
    CUTE_UNROLL
    for (int offset = 16; offset > 0; offset >>= 1) {
      lse_max = max(lse_max, __shfl_xor_sync(uint32_t(-1), lse_max, offset));
    }

    // In case all local LSEs are -inf
    lse_max = lse_max == -INFINITY ? 0.0f : lse_max;

    // local sum of exp(lse - lse_max)
    float lse_sumexp = 0;
    CUTE_UNROLL
    for (int i = 0; i < kLsePerThr; ++i) {
      lse_sumexp += expf(lse[i] - lse_max);
    }

    // lse sumexp within warp
    CUTE_UNROLL
    for (int offset = 16; offset > 0; offset >>= 1) {
      lse_sumexp += __shfl_xor_sync(uint32_t(-1), lse_sumexp, offset);
    }

    const float lse_logsumexp = (lse_sumexp == 0.f || lse_sumexp != lse_sumexp)
                                    ? INFINITY
                                    : logf(lse_sumexp) + lse_max;

    // calculate lse scale for each split
    CUTE_UNROLL
    for (int i = 0; i < kLsePerThr; ++i) {
      const int split_idx = get<0>(tTcLseAccum(i));
      // scales[i] = exp(lse[i] - lse_max) / lse_sumexp
      //           = exp(lse[i] - lse_max - log(lse_sumexp))
      //           = exp(lse[i] - lse_logsumexp)
      sScales[split_idx] = expf(lse[i] - lse_logsumexp);
    }

    if (tidx == 0) {
      gLse(b_idx, s_idx, h_idx) = lse_logsumexp;
    }
  }
  __syncthreads();

  static_assert(kHeadDim % kThreads == 0);

  // each thread copy kHeadDim / kThreads elements
  auto gmem_tiled_copy_Oaccum =
      make_tiled_copy(Copy_Atom<VectorizingCopy, ElementAccum>{},
                      Layout<Shape<Int<kThreads>>>{},
                      Layout<Shape<Int<kHeadDim / kThreads>>>{});
  auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);

  // (kHeadDim) -> (head_dim)
  Tensor cOaccum = make_identity_tensor(Shape<Int<kHeadDim>>{});
  Tensor tOcOaccum = gmem_thr_copy_Oaccum.partition_S(cOaccum);
  auto max_coord = make_coord(head_dim);

  // [kHeaddim] => (CPY, CPY_K)
  Tensor tOrOaccum =
      make_fragment_like(gmem_thr_copy_Oaccum.partition_D(gOaccum(_0{}, _)));

  // output accumulators
  Tensor tOrO = make_tensor_like(tOrOaccum);
  clear(tOrO);

  // apply scales to each split and accumulate
  for (int split_idx = 0; split_idx < n_splits; ++split_idx) {
    Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_S(gOaccum(split_idx, _));

    // copy oAccum from gmem to rmem directly
    safe_copy<EVEN_K, /*ZFILL_K=*/true>(
        gmem_tiled_copy_Oaccum, tOgOaccum, tOrOaccum, tOcOaccum, max_coord);

    const auto lse_scale = sScales[split_idx];
    CUTE_UNROLL
    for (int i = 0; i < size(tOrO); ++i) {
      tOrO(i) += lse_scale * tOrOaccum(i);
    }
  }

  // cast output from ElementAccum to Element
  Tensor tOrO_ = make_tensor_like<Element>(tOrO);
  fast_cast(tOrO, tOrO_);

  // each thread copy kHeadDim / kThreads elements
  auto gmem_tiled_copy_O =
      make_tiled_copy(Copy_Atom<VectorizingCopy, Element>{},
                      Layout<Shape<Int<kThreads>>>{},
                      Layout<Shape<Int<kHeadDim / kThreads>>>{});
  auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);

  Tensor cO = make_identity_tensor(Shape<Int<kHeadDim>>{});

  // [kHeadDim] => (CPY, CPY_K)
  Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
  Tensor tOcO = gmem_thr_copy_O.partition_D(cO);

  safe_copy<EVEN_K, /*ZFILL_K=*/false>(
      gmem_tiled_copy_O, tOrO_, tOgO, tOcO, max_coord);
}

template <typename Element,
          typename ElementAccum,
          int kHeadDim,
          int kSplits,
          bool EVEN_K,
          typename Params>
void launch_attn_combine_kernel(const Params& params, cudaStream_t stream) {
  const auto batch_size = params.batch_size;
  const auto q_len = params.q_len;
  const auto n_heads = params.n_heads;

  // determine kThreads based on kHeadDim
  constexpr int kThreads = kHeadDim <= 32 ? 32 : kHeadDim <= 64 ? 64 : 128;
  static_assert(kHeadDim % kThreads == 0);

  auto combine_kernel = attn_combine_kernel<Element,
                                            ElementAccum,
                                            kHeadDim,
                                            kSplits,
                                            kThreads,
                                            EVEN_K,
                                            Params>;

  dim3 grid(batch_size, q_len, n_heads);
  combine_kernel<<<grid, kThreads, 0, stream>>>(params);
}

}  // namespace llm
