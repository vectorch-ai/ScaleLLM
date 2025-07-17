#pragma once

#include <cute/int_tuple.hpp>
#include <cute/layout.hpp>

#include "common/static_dispatch.h"

namespace llm {
// forward declaration
template <typename Dtype,
          int HEAD_DIM,
          bool EVEN_K,
          bool ALIBI,
          bool SOFT_CAP,
          bool LOCAL,
          typename Params>
void sm80_launch_mha_kernel(const Params& params, cudaStream_t stream);

// user-facing function to run the attention kernel
template <typename Dtype, int HEAD_DIM, typename Params>
void sm80_run_mha(Params& params, cudaStream_t stream = nullptr) {
  // normalize params that for performance optimization
  params.normalize();

  // dispatch to proper kernel instantiation based on params
  DISPATCH_BOOL(params.head_dim == HEAD_DIM, EVEN_K, [&] {
    DISPATCH_BOOL(params.alibi_slopes_ptr != nullptr, ALIBI, [&] {
      DISPATCH_BOOL(params.logits_soft_cap > 0, SOFT_CAP, [&] {
        DISPATCH_BOOL(params.sliding_window >= 0, LOCAL, [&] {
          sm80_launch_mha_kernel<Dtype,
                                 HEAD_DIM,
                                 EVEN_K,
                                 ALIBI,
                                 SOFT_CAP,
                                 LOCAL,
                                 Params>(params, stream);
        });
      });
    });
  });
}

}  // namespace llm
