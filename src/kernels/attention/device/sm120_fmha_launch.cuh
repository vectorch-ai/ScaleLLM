#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cute/layout.hpp>
#include <cute/tensor.hpp>

#include "collective/sm120_collective_epilogue.cuh"
#include "collective/sm120_collective_fmha_mainloop_ws.cuh"
#include "common/fmha_block.h"
#include "common/tile_scheduler.cuh"
#include "kernel/sm120_kernel_fmha_ws.cuh"

namespace llm {

namespace detail {
/// Generic kernel template.
template <typename Operator>
__global__ __launch_bounds__(Operator::kThreadsPerBlock) void device_kernel(
    __grid_constant__ const typename Operator::Params params) {
  extern __shared__ char smem[];
  Operator op;
  op(params, smem);
}
}  // namespace detail

template <typename Dtype,
          int kHeadDim,
          bool EVEN_K,
          bool ALIBI,
          bool SOFT_CAP,
          bool LOCAL,
          typename Params>
void sm120_launch_mha_kernel(const Params& params, cudaStream_t stream) {
  // TODO: tune tile shape M/N based on the head dim and smem size
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications-technical-specifications-per-compute-capability
  // SM           | 7.0 | 7.2 | 7.5 | 8.0 | 8.6 | 8.7 | 8.9 | 9.0 | 10.x | 12.0|
  // Max SMEM (KB)|     96    |  64 | 164 | 100 | 164 | 100 |     228    | 100 |
  // valid dynamic shared memory sizes for different compute capabilities:
  // * 7.0 | 7.2 : 0, 8, 16, 32, 64, 96
  // * 7.5       : 0, 32, 64
  // * 8.0 | 8.7 : 0, 8, 16, 32, 64, 100, 132, 164
  // * 8.6 | 8.9 : 0, 8, 16, 32, 64, 100
  // * 9.0 | 10.x: 0, 8, 16, 32, 64, 100, 132, 164, 196, 228
  // * 12.0      : 0, 8, 16, 32, 64, 100
  constexpr int BLK_M = 64;
  constexpr int BLK_N = 64;

  // TMA is used for K/V loading
  constexpr bool KV_USE_TMA = false;

  assert(params.n_heads % params.n_kv_heads == 0 &&
         "n_heads must be divisible by n_kv_heads");
  const int group_size = params.n_heads / params.n_kv_heads;

  using TileShape = Shape<Int<BLK_M>, Int<BLK_N>, Int<kHeadDim>>;

  using Block = FmhaBlock<TileShape, Dtype, LOCAL>;

  using CollectiveMainloop = Sm120CollectiveFMhaWs<TileShape,
                                                   Dtype,
                                                   EVEN_K,
                                                   ALIBI,
                                                   SOFT_CAP,
                                                   LOCAL,
                                                   KV_USE_TMA>;
  using CollectiveEpilogue = Sm120CollectiveEpilogue<TileShape, Dtype, EVEN_K>;

  // TODO: support persistent kernels
  using TileScheduler = SingleTileScheduler;

  // TODO: pass in max_q_len and max_kv_len for variable length
  // MNKL: (Q K D ((KH G), B))
  using ProblemShape =
      cute::tuple<int, int, int, cute::tuple<cute::tuple<int, int>, int>>;
  ProblemShape problem_shape = make_tuple(
      params.q_len,
      params.kv_len,
      params.head_dim,
      make_tuple(make_tuple(params.n_kv_heads, group_size), params.batch_size));

  using AttnKernel = Sm120KernelFmhaWs<ProblemShape,  // (Q, K, D, ((KH G), B))
                                       Block,
                                       CollectiveMainloop,
                                       CollectiveEpilogue,
                                       TileScheduler>;

  // TODO: convert params to Kernel Args
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
      // Block arguments
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
      // mainloop arguments
      .mainloop =
          {
              .sliding_window = params.sliding_window,
              .logits_soft_cap = params.logits_soft_cap,
              .sm_scale = params.sm_scale,
              .alibi_slopes_ptr = params.alibi_slopes_ptr,
          },
      // epilogue arguments
      .epilogue = {},
  };

  auto attn_params =
      AttnKernel::to_underlying_arguments(attn_args, /*workspace*/ nullptr);

  auto mha_kernel = detail::device_kernel<AttnKernel>;

  const auto smem_size = AttnKernel::kSharedStorageSize;
  if (smem_size >= 48 * 1024) {
    cudaFuncSetAttribute(
        mha_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  }

  const dim3 grid = AttnKernel::get_grid_shape(attn_params);
  const dim3 block = AttnKernel::get_block_shape();

  mha_kernel<<<grid, block, smem_size, stream>>>(attn_params);
  // TODO: check launch status
}

}  // namespace llm
