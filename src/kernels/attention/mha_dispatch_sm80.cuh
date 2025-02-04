#pragma once

#include <cute/int_tuple.hpp>
#include <cute/layout.hpp>

#include "mha_traits_sm80.h"
#include "static_dispatch.h"

namespace llm {
// forward declaration
template <typename Traits,
          typename Params,
          bool EVEN_K,
          bool ALIBI,
          bool SOFT_CAP,
          bool LOCAL>
void launch_mha_kernel_sm80(const Params& params, cudaStream_t stream);

namespace detail {

template <typename Traits, typename Params>
void dispatch_mha_kernel_sm80(const Params& params, cudaStream_t stream) {
  // dispatch to proper kernel instantiation based on params
  DISPATCH_BOOL(params.head_dim == Traits::kHeadDim, EVEN_K, [&] {
    DISPATCH_BOOL(params.alibi_slopes_ptr != nullptr, ALIBI, [&] {
      DISPATCH_BOOL(params.logits_soft_cap > 0, SOFT_CAP, [&] {
        DISPATCH_BOOL(params.sliding_window >= 0, LOCAL, [&] {
          launch_mha_kernel_sm80<Traits,
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
    using Traits = MHATraitsSM80<Dtype,
                                 HEAD_DIM,
                                 /*BLK_M=*/64,
                                 /*BLK_N=*/64,
                                 /*BLK_K=*/64>;
    detail::dispatch_mha_kernel_sm80<Traits>(params, stream);
  } else if constexpr (HEAD_DIM == 96) {
    using Traits = MHATraitsSM80<Dtype,
                                 HEAD_DIM,
                                 /*BLK_M=*/64,
                                 /*BLK_N=*/64,
                                 /*BLK_K=*/32>;
    detail::dispatch_mha_kernel_sm80<Traits>(params, stream);
  } else if constexpr (HEAD_DIM == 128) {
    using Traits = MHATraitsSM80<Dtype,
                                 HEAD_DIM,
                                 /*BLK_M=*/64,
                                 /*BLK_N=*/64,
                                 /*BLK_K=*/64>;
    detail::dispatch_mha_kernel_sm80<Traits>(params, stream);
  } else if constexpr (HEAD_DIM == 256) {
    using Traits = MHATraitsSM80<Dtype,
                                 HEAD_DIM,
                                 /*BLK_M=*/64,
                                 /*BLK_N=*/64,
                                 /*BLK_K=*/64>;
    detail::dispatch_mha_kernel_sm80<Traits>(params, stream);
  } else {
    // use the default block size
    using Traits = MHATraitsSM80<Dtype,
                                 HEAD_DIM,
                                 /*BLK_M=*/64,
                                 /*BLK_N=*/64,
                                 /*BLK_K=*/64>;
    detail::dispatch_mha_kernel_sm80<Traits>(params, stream);
  }
}

}  // namespace llm