#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cute/layout.hpp>
#include <cute/tensor.hpp>

#include "cute_extensions.cuh"
#include "fast_cast.cuh"

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
  // [n_splits, kHeadDim] => [kHeadDim, n_splits]
  Tensor gOaccum = select<1, 0>(oAccum(_, b_idx, s_idx, h_idx, _));

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

    auto thr_layout = make_layout(make_shape(_32{}));
    // [n_lses]
    auto tLgLseAccum = local_partition(gLseAccum, thr_layout, tidx);
    auto tLcLseAccum = local_partition(cLseAccum, thr_layout, tidx);

    constexpr int kLsePerThr = size(tLcLseAccum);

    // local lse
    float lse[kLsePerThr];

    CUTE_UNROLL
    for (int i = 0; i < kLsePerThr; ++i) {
      const int split_idx = get<0>(tLcLseAccum(i));
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
    for (int offset = 16; offset > 0; offset >>= 1) {
      lse_max = max(lse_max, __shfl_xor_sync(uint32_t(-1), lse_max, offset));
    }

    lse_max = lse_max == -INFINITY
                  ? 0.0f
                  : lse_max;  // In case all local LSEs are -inf

    // local sum
    float lse_sum = 0;
    CUTE_UNROLL
    for (int i = 0; i < kLsePerThr; ++i) {
      lse_sum += expf(lse[i] - lse_max);
    }

    // lse sum within warp
    CUTE_UNROLL
    for (int offset = 16; offset > 0; offset >>= 1) {
      lse_sum += __shfl_xor_sync(uint32_t(-1), lse_sum, offset);
    }

    const float lse_logsum = (lse_sum == 0.f || lse_sum != lse_sum)
                                 ? INFINITY
                                 : logf(lse_sum) + lse_max;

    // calculate lsescale for each split
    CUTE_UNROLL
    for (int i = 0; i < kLsePerThr; ++i) {
      const int split_idx = get<0>(tLcLseAccum(i));
      // scales[i] = exp(lse[i] - lse_max) / lse_sum
      //           = exp(lse[i] - lse_max - log(lse_sum))
      //           = exp(lse[i] - lse_logsum)
      sScales[split_idx] = expf(lse[i] - lse_logsum);
    }

    if (tidx == 0) {
      gLse(b_idx, s_idx, h_idx) = lse_logsum;
    }
  }
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

  // [kHeaddim, n_splits] => (CPY, CPY_K, n_splits)
  Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_S(gOaccum);
  // (CPY, CPY_K)
  Tensor tOrOaccum = make_tensor<ElementAccum>(shape(tOgOaccum(_, _, _0{})));

  // output accumulators
  Tensor tOrO = make_tensor_like(tOrOaccum);
  clear(tOrO);

  for (int split_idx = 0; split_idx < n_splits; ++split_idx) {
    // copy oAccum from gmem to rmem directly
    cute::copy(gmem_tiled_copy_Oaccum, tOgOaccum(_, _, split_idx), tOrOaccum);

    const auto lse_scale = sScales[split_idx];
    CUTE_UNROLL
    for (int i = 0; i < size(tOrO); ++i) {
      tOrO(i) += lse_scale * tOrOaccum(i);
    }
  }

  // cast output from ElementAccumulator to Element
  Tensor tOrO_ = make_tensor_like<Element>(tOrO);
  fast_cast(tOrO, tOrO_);

  Tensor tOgO = gmem_thr_copy_Oaccum.partition_D(gO);
  cute::copy(gmem_tiled_copy_Oaccum, tOrO_, tOgO);
}

template <typename Element,
          typename ElementAccum,
          int kHeadDim,
          int kSplits,
          typename Params>
void launch_attn_combine_kernel(const Params& params, cudaStream_t stream) {
  const auto batch_size = params.batch_size;
  const auto q_len = params.q_len;
  const auto n_heads = params.n_heads;

  auto combine_kernel =
      attn_combine_kernel<Element, ElementAccum, kHeadDim, kSplits, Params>;

  dim3 grid(batch_size, q_len, n_heads);
  combine_kernel<<<grid, 128, 0, stream>>>(params);
}

}  // namespace llm