#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include <cute/layout.hpp>
#include <cute/tensor.hpp>

#include "collective/sm120_collective_epilogue.cuh"
#include "collective/sm120_collective_fmha_mainloop_ws.cuh"
#include "common/fmha_block.h"
#include "common/tile_scheduler.cuh"
#include "device/fmha.cuh"
#include "fmha_params.h"
#include "kernel/sm120_kernel_fmha_ws.cuh"

namespace llm {
// ? Should include ArchTag?
//  * select right kernel based on ArchTag?
// ? how to support fast compliling?
//  * only compile the kernel for the target compute capability
template <typename Element, int kHeadDim>
class FmhaRunner {
 public:
  static bool run(const FmhaParams& params, cudaStream_t stream = nullptr) {
    assert(params.head_dim <= kHeadDim);
    // dispatch to proper kernel instantiation based on params
    DISPATCH_BOOL(params.head_dim == kHeadDim, EVEN_K, [&] {
      DISPATCH_BOOL(params.alibi_slopes_ptr != nullptr, ALIBI, [&] {
        DISPATCH_BOOL(params.logits_soft_cap > 0, SOFT_CAP, [&] {
          DISPATCH_BOOL(params.sliding_window >= 0, LOCAL, [&] {
            return run_kernel<EVEN_K, ALIBI, SOFT_CAP, LOCAL>(params, stream);
          });
        });
      });
    });
    return false;  // should never reach here
  }

  template <bool EVEN_K, bool ALIBI, bool SOFT_CAP, bool LOCAL>
  static bool run_kernel(const FmhaParams& params,
                         cudaStream_t stream = nullptr) {
    // TODO: tune tile shape M/N based on the head dim and smem size
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications-technical-specifications-per-compute-capability
    // SM           | 7.0 | 7.2 | 7.5 | 8.0 | 8.6 | 8.7 | 8.9 | 9.0 | 10.x
    // | 12.0| Max SMEM (KB)|     96    |  64 | 164 | 100 | 164 | 100 |     228
    // | 100 | valid dynamic shared memory sizes for different compute
    // capabilities:
    // * 7.0 | 7.2 : 0, 8, 16, 32, 64, 96
    // * 7.5       : 0, 32, 64
    // * 8.0 | 8.7 : 0, 8, 16, 32, 64, 100, 132, 164
    // * 8.6 | 8.9 : 0, 8, 16, 32, 64, 100
    // * 9.0 | 10.x: 0, 8, 16, 32, 64, 100, 132, 164, 196, 228
    // * 12.0      : 0, 8, 16, 32, 64, 100
    static constexpr int BLK_M = 64;
    static constexpr int BLK_N = 64;

    // TMA is used for K/V loading
    constexpr bool KV_USE_TMA = false;

    // TODO: pass in max_q_len and max_kv_len for variable length
    // MNKL: (Q K D ((KH G), B))
    using ProblemShape =
        cute::tuple<int, int, int, cute::tuple<cute::tuple<int, int>, int>>;

    using TileShape = Shape<Int<BLK_M>, Int<BLK_N>, Int<kHeadDim>>;

    using Block = FmhaBlock<TileShape, Element, LOCAL>;

    using CollectiveMainloop = Sm120CollectiveFMhaWs<TileShape,
                                                     Element,
                                                     EVEN_K,
                                                     ALIBI,
                                                     SOFT_CAP,
                                                     LOCAL,
                                                     KV_USE_TMA>;
    using CollectiveEpilogue =
        Sm120CollectiveEpilogue<TileShape, Element, EVEN_K>;

    // TODO: support persistent kernels
    using TileScheduler = SingleTileScheduler;

    using AttnKernel = Sm120KernelFmhaWs<ProblemShape,
                                         Block,
                                         CollectiveMainloop,
                                         CollectiveEpilogue,
                                         TileScheduler>;

    assert(params.n_heads % params.n_kv_heads == 0 &&
           "n_heads must be divisible by n_kv_heads");
    const int group_size = params.n_heads / params.n_kv_heads;

    ProblemShape problem_shape =
        make_tuple(params.q_len,
                   params.kv_len,
                   params.head_dim,
                   make_tuple(make_tuple(params.n_kv_heads, group_size),
                              params.batch_size));

    auto q_stride = make_stride(
        params.q_batch_stride, params.q_seq_stride, params.q_head_stride, _1{});
    auto k_stride = make_stride(
        params.k_batch_stride, params.k_seq_stride, params.k_head_stride, _1{});
    auto v_stride = make_stride(
        params.v_batch_stride, params.v_seq_stride, params.v_head_stride, _1{});
    auto o_stride = make_stride(
        params.o_batch_stride, params.o_seq_stride, params.o_head_stride, _1{});

    typename AttnKernel::Arguments attn_args{
        .problem_shape = problem_shape,
        .block =
            {
                .q_ptr = params.q_ptr,
                .k_ptr = params.k_ptr,
                .v_ptr = params.v_ptr,
                .o_ptr = params.o_ptr,
                .q_stride = q_stride,
                .k_stride = k_stride,
                .v_stride = v_stride,
                .o_stride = o_stride,
                .sliding_window = params.sliding_window,
            },
        .mainloop =
            {
                .sliding_window = params.sliding_window,
                .logits_soft_cap = params.logits_soft_cap,
                .sm_scale = params.sm_scale,
                .alibi_slopes_ptr = params.alibi_slopes_ptr,
            },
        .epilogue = {},
    };

    Fmha<AttnKernel> fmha;
    if (!fmha.initialize(attn_args, /*workspace=*/nullptr)) {
      return false;
    }
    return fmha.run(stream);
  }
};
}  // namespace llm
