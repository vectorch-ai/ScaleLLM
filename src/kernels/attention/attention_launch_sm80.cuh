#pragma once

#include "attention_kernel_sm80.cuh"
#include "attention_traits_sm80.h"
#include "static_dispatch.h"

namespace llm {
namespace detail {
template <typename Traits, typename Params>
void launch_attention_kernel(const Params& params, cudaStream_t stream) {
  const auto batch_size = params.batch_size;
  const auto n_heads = params.n_heads;
  const auto max_q_len = params.max_q_len;

  const auto smem_size = Traits::kSmemSize;
  auto attention_kernel = mha_kernel_sm80<Traits, Params>;
  cudaFuncSetAttribute(
      attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  // TODO: support persistent kernels
  dim3 grid(
      (max_q_len + Traits::kBlockM - 1) / Traits::kBlockM, batch_size, n_heads);
  dim3 block = Traits::kThreadNum;
  attention_kernel<<<grid, block, smem_size, stream>>>(params);
}

template <typename Element,
          int HEAD_DIM,
          int BLK_M,
          int BLK_N,
          int BLK_K,
          typename Params>
void run_attention_kernel(const Params& params, cudaStream_t stream) {
  // dispatch to proper kernel instantiation based on params
  DISPATCH_BOOL(params.alibi_slopes_ptr != nullptr, ALIBI, [&] {
    using Traits =
        AttentionTraitsSM80<Element, HEAD_DIM, BLK_M, BLK_N, BLK_K, ALIBI>;
    launch_attention_kernel<Traits>(params, stream);
  });
}

}  // namespace detail

// user-facing function to run the attention kernel
template <typename Element, int HEAD_DIM, typename Params>
void run_attention_kernel_sm80(const Params& params,
                               cudaStream_t stream = nullptr) {
  // TODO: tune block shape MNK based on the head dim and smem size
  if constexpr (HEAD_DIM == 64) {
    detail::run_attention_kernel<Element,
                                 HEAD_DIM,
                                 /*BLK_M=*/64,
                                 /*BLK_N=*/64,
                                 /*BLK_K=*/64>(params, stream);
  } else if constexpr (HEAD_DIM == 96) {
    detail::run_attention_kernel<Element,
                                 HEAD_DIM,
                                 /*BLK_M=*/64,
                                 /*BLK_N=*/64,
                                 /*BLK_K=*/32>(params, stream);
  } else if constexpr (HEAD_DIM == 128) {
    detail::run_attention_kernel<Element,
                                 HEAD_DIM,
                                 /*BLK_M=*/64,
                                 /*BLK_N=*/64,
                                 /*BLK_K=*/64>(params, stream);
  } else if constexpr (HEAD_DIM == 256) {
    detail::run_attention_kernel<Element,
                                 HEAD_DIM,
                                 /*BLK_M=*/64,
                                 /*BLK_N=*/64,
                                 /*BLK_K=*/64>(params, stream);
  } else {
    // use the default block size
    detail::run_attention_kernel<Element,
                                 HEAD_DIM,
                                 /*BLK_M=*/64,
                                 /*BLK_N=*/64,
                                 /*BLK_K=*/32>(params, stream);
  }
}

}  // namespace llm