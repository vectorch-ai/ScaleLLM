#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cute/layout.hpp>
#include <cute/tensor.hpp>

#include "cute/config.hpp"
#include "cute/tensor_impl.hpp"

// #include "cute/config.hpp"
// #include "fast_cast.cuh"

namespace llm {

// combine ouputs from multiple splits:
//  output = sum(softmax(local_lsm) * outAccum)
//  lsm = log(sum(exp(local_lsm))
// inputs:
//    outAccum:   [n_splits, batch, seq_len, n_heads, head_dim]
//    local_lsm:  [n_splits, batch, seq_len, n_heads]
// output:
//    out:      [batch, seq_len, n_heads, head_dim]
//    lsm:      [batch, seq_len, n_heads]
template <typename Element,
          typename ElementAccum,
          int kHeadDim,
          int kSplits,
          typename Params>
__global__ void attn_combine_kernel(__grid_constant__ const Params params) {
  using namespace cute;
  constexpr int kThreads = 128;

  const int tidx = threadIdx.x;
  // Grid: [batch, seq_len, n_heads]
  const int b_idx = blockIdx.x;
  const int s_idx = blockIdx.y;
  const int h_idx = blockIdx.z;

  // TODO: support variable number of splits per batch
  const int n_splits = params.n_splits;

  // only one split, handled in attn kernel already
  if (n_splits == 1) {
    return;
  }

  // scales[splits]
  __shared__ ElementAccum sScales[kSplits];

  // [n_splits, batch, seq_len, n_heads]
  Tensor lseAccum = make_tensor(
      make_gmem_ptr(reinterpret_cast<ElementAccum*>(params.lse_accum_ptr)),
      params.lse_accum_shape,
      append<4>(params.lse_accum_stride, _1{}));
  // [n_splits]
  Tensor gLseAccum = lseAccum(_, b_idx, s_idx, h_idx);

  // [num_splits, batch, num_heads, max_seqlen_q, head_dim]
  Tensor oAccum = make_tensor(
      make_gmem_ptr(reinterpret_cast<ElementAccum*>(params.oaccum_ptr)),
      params.oaccum_shape,
      append<5>(params.oaccum_stride, _1{}));
  // [kMaxSplits, head_dim] => [head_dim, kMaxSplits]
  Tensor gOaccum = select<1, 0>(oAccum(_, b_idx, h_idx, s_idx, _));

  // [batch, seqlen_q, num_heads, head_dim]
  Tensor O =
      make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)),
                  params.o_shape,
                  append<4>(params.o_stride, _1{}));
  // [head_dim]
  Tensor gO = O(b_idx, s_idx, h_idx, _);

  // use one warp to calculate safe_softmax(lse)
  int warp_idx = cutlass::canonical_warp_idx_sync();
  if (warp_idx == 0) {
    // (SPLITS) -> (splits)
    auto cLseAccum = make_identity_tensor(Shape<Int<kSplits>>{});

    // [n_lses]
    auto tLgLseAccum = local_partition(gLseAccum, _32{}, tidx);
    auto tLcLseAccum = local_partition(cLseAccum, _32{}, tidx);

    constexpr int kLsePerThr = size(tLcLseAccum);

    // local lse
    float lse[kLsePerThr];

    CUTE_UNROLL
    for (int i = 0; i < kLsePerThr; ++i) {
      const int split_idx = tLcLseAccum(i);
      lse[i] = split_idx < n_splits ? tLgLseAccum(i) : -INFINITY;
    }

    // local lse max
    float lse_max = -INFINITY;
    CUTE_UNROLL
    for (int i = 0; i < kLsePerThr; ++i) {
      lse_max = max(lse_max, lse[i]);
    }

    // lse max within warp
    CUTE_UNROLL
    for (int offset = 16; offset >= 1; offset /= 2) {
      lse_max = max(lse_max, __shfl_xor_sync(uint32_t(-1), lse_max, offset));
    }

    lse_max = lse_max == -INFINITY
                  ? 0.0f
                  : lse_max;  // In case all local LSEs are -inf

    // local sum
    float lse_sum = 0;
    CUTE_UNROLL
    for (int i = 0; i < kLsePerThr; ++i) {
      lse_sum = lse_sum + expf(lse[i] - lse_max);
    }

    // lse sum within warp
    CUTE_UNROLL
    for (int offset = 16; offset >= 1; offset /= 2) {
      lse_sum = lse_sum + __shfl_xor_sync(uint32_t(-1), lse_sum, offset);
    }

    float lse_logsum = (lse_sum == 0.f || lse_sum != lse_sum)
                           ? INFINITY
                           : logf(lse_sum) + lse_max;

    // calculate lsescale for each split
    CUTE_UNROLL
    for (int i = 0; i < kLsePerThr; ++i) {
      const int split_idx = tLcLseAccum(i);
      // scales[i] = exp(lse[i] - lse_max) / lse_sum
      //           = exp(lse[i] - lse_max - log(lse_sum))
      //           = exp(lse[i] - lse_logsum)
      sScales[split_idx] = expf(lse[i] - lse_logsum);
    }
  }

  // wait for all threads to finish updating sLseScale
  __syncthreads();

  // TODO: handle uneven kHeadDimV
  static_assert(kHeadDim % kThreads == 0);

  // kThreads, each thread copy kHeadDim / kThreads elements
  using GmemTiledCopyOaccum = decltype(make_tiled_copy(
      Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccum>{},
      Layout<Shape<Int<kThreads>>>{},
      Layout<Shape<Int<kHeadDim / kThreads>>>{}));

  GmemTiledCopyOaccum gmem_tiled_copy_Oaccum;
  auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);

  // (CPY, CPY_K, n_splits)
  Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_S(gOaccum);
  // (CPY, CPY_K)
  Tensor tOrOaccum = make_tensor<ElementAccum>(shape(tOgOaccum(_, _, _0{})));

  // output accumulators
  Tensor tOrO = make_tensor<ElementAccum>(shape(tOgOaccum));
  clear(tOrO);

  for (int split_idx = 0; split_idx < n_splits; ++split_idx) {
    // copy oAccum from gmem to rmem directly
    cute::copy(tOgOaccum(_, _, split_idx), tOrOaccum);

    ElementAccum lse_scale = sScales[split_idx];
    CUTE_UNROLL
    for (int i = 0; i < size(tOrO); ++i) {
      tOrO(i) += lse_scale * tOrOaccum(i);
    }
  }

  // cast output from ElementAccumulator to Element
  auto tOrO_ = make_tensor_like<Element>(tOrO);
  fast_cast(tOrO, tOrO_);

  auto tOgO = gmem_thr_copy_Oaccum.partition_D(gO);
  cute::copy(tOrO_, tOgO);
}

}  // namespace llm