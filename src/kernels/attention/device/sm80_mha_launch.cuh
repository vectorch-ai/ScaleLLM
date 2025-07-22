#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cute/layout.hpp>
#include <cute/tensor.hpp>

#include "collective/sm80_collective_epilogue.cuh"
#include "collective/sm80_collective_mha.cuh"
#include "common/tile_scheduler.cuh"
#include "kernel/sm80_kernel_mha.cuh"

namespace llm {

namespace detail {
/// Generic kernel template.
template <typename Operator, typename Params>
__global__ __launch_bounds__(Operator::kMmaThreads) void device_kernel(
    __grid_constant__ const Params params,
    __grid_constant__ const typename Operator::TileSchedulerParams
        scheduler_params) {
  extern __shared__ char smem[];
  Operator op;
  op(params, scheduler_params, smem);
}
}  // namespace detail

template <typename Dtype,
          int HEAD_DIM,
          bool EVEN_K,
          bool ALIBI,
          bool SOFT_CAP,
          bool LOCAL,
          typename Params>
void sm80_launch_mha_kernel(const Params& params, cudaStream_t stream) {
  const auto batch_size = params.batch_size;
  const auto n_kv_heads = params.n_kv_heads;
  const auto max_q_packed_len = params.max_q_len * params.group_size;

  // TODO: tune block shape MNK based on the head dim and smem size
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

  using TileShape = Shape<Int<BLK_M>, Int<BLK_N>, Int<HEAD_DIM>>;
  using CollectiveMainloop =
      Sm80CollectiveMha<TileShape, Dtype, EVEN_K, ALIBI, SOFT_CAP, LOCAL>;
  using CollectiveEpilogue = Sm80CollectiveEpilogue<TileShape, Dtype, EVEN_K>;

  // TODO: support persistent kernels
  using TileScheduler = SingleTileScheduler;

  const auto m_blocks = cute::ceil_div(max_q_packed_len, BLK_M);
  typename TileScheduler::Params scheduler_params{
      .batch_size = batch_size, .m_blocks = m_blocks, .n_kv_heads = n_kv_heads};
  using AttnKernel =
      Sm80KernelMha<CollectiveMainloop, CollectiveEpilogue, TileScheduler>;

  auto mha_kernel = detail::device_kernel<AttnKernel, Params>;

  const auto smem_size = AttnKernel::kSharedStorageSize;
  if (smem_size >= 48 * 1024) {
    cudaFuncSetAttribute(
        mha_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  }

  const dim3 grid = AttnKernel::get_grid_shape(scheduler_params);
  const dim3 block = AttnKernel::get_block_shape();

  mha_kernel<<<grid, block, smem_size, stream>>>(params, scheduler_params);
  // TODO: check launch status
}

}  // namespace llm
