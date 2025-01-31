#pragma once

#include <cute/int_tuple.hpp>
#include <cute/layout.hpp>

#include "mha_kernel_sm80.cuh"
#include "mha_traits_sm80.h"
#include "static_dispatch.h"

namespace llm {
namespace detail {
template <typename Traits,
          typename Params,
          bool EVEN_K,
          bool ALIBI,
          bool SOFT_CAP,
          bool LOCAL>
void launch_mha_kernel(const Params& params, cudaStream_t stream) {
  const auto batch_size = params.batch_size;
  const auto n_kv_heads = params.n_kv_heads;
  const auto max_q_packed_len = params.max_q_len * params.group_size;

  const auto smem_size = Traits::kSmemSize;
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

template <typename Traits, typename Params>
void run_mha_kernel(const Params& params, cudaStream_t stream) {
  // dispatch to proper kernel instantiation based on params
  DISPATCH_BOOL(params.head_dim == Traits::kHeadDim, EVEN_K, [&] {
    DISPATCH_BOOL(params.alibi_slopes_ptr != nullptr, ALIBI, [&] {
      DISPATCH_BOOL(params.logits_soft_cap > 0, SOFT_CAP, [&] {
        DISPATCH_BOOL(params.sliding_window >= 0, LOCAL, [&] {
          launch_mha_kernel<Traits,
                                  Params,
                                  EVEN_K,
                                  ALIBI,
                                  SOFT_CAP,
                                  LOCAL>(params, stream);
        });
      });
    });
  });
}

}  // namespace detail

// user-facing function to run the attention kernel
template <typename Dtype, int HEAD_DIM, typename Params>
void run_mha_kernel_sm80(Params& params, cudaStream_t stream = nullptr) {
  // normalize params that for performance optimization
  params.normalize();

  // TODO: tune block shape MNK based on the head dim and smem size
  if constexpr (HEAD_DIM == 64) {
    using Traits = AttentionTraitsSM80<Dtype,
                                       HEAD_DIM,
                                       /*BLK_M=*/64,
                                       /*BLK_N=*/64,
                                       /*BLK_K=*/64>;
    detail::run_mha_kernel<Traits>(params, stream);
  } else if constexpr (HEAD_DIM == 96) {
    using Traits = AttentionTraitsSM80<Dtype,
                                       HEAD_DIM,
                                       /*BLK_M=*/64,
                                       /*BLK_N=*/64,
                                       /*BLK_K=*/32>;
    detail::run_mha_kernel<Traits>(params, stream);
  } else if constexpr (HEAD_DIM == 128) {
    using Traits = AttentionTraitsSM80<Dtype,
                                       HEAD_DIM,
                                       /*BLK_M=*/64,
                                       /*BLK_N=*/64,
                                       /*BLK_K=*/64>;
    detail::run_mha_kernel<Traits>(params, stream);
  } else if constexpr (HEAD_DIM == 256) {
    using Traits = AttentionTraitsSM80<Dtype,
                                       HEAD_DIM,
                                       /*BLK_M=*/64,
                                       /*BLK_N=*/64,
                                       /*BLK_K=*/64>;
    detail::run_mha_kernel<Traits>(params, stream);
  } else {
    // use the default block size
    using Traits = AttentionTraitsSM80<Dtype,
                                       HEAD_DIM,
                                       /*BLK_M=*/64,
                                       /*BLK_N=*/64,
                                       /*BLK_K=*/64>;
    detail::run_mha_kernel<Traits>(params, stream);
  }
}

}  // namespace llm