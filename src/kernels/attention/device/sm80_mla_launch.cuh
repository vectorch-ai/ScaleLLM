#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cute/layout.hpp>
#include <cute/tensor.hpp>

#include "collective/sm80_collective_mla.cuh"
#include "collective/sm80_collective_mla_epilogue.cuh"
#include "common/tile_scheduler.cuh"
#include "kernel/sm80_kernel_mla.cuh"

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

template <int ARC, int HEAD_DIM, int ROPE_HEAD_DIM>
CUTE_HOST_DEVICE constexpr auto tile_shape_selector() {
  using namespace cute;
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
  //   constexpr int stages = 1;
  if constexpr (HEAD_DIM <= 128) {
    return Shape<_64, _64, _128>{};
  } else if constexpr (HEAD_DIM <= 256) {
    return Shape<_64, _32, _128>{};
  } else if constexpr (HEAD_DIM <= 512) {
    return Shape<_64, _16, _128>{};
  } else {
    return Shape<_64, _16, _128>{};
  }
}

template <int ARC, int HEAD_DIM, int ROPE_HEAD_DIM>
CUTE_HOST_DEVICE constexpr auto stages_selector() {
  using namespace cute;
  if constexpr (HEAD_DIM <= 128) {
    return 2;
  } else if constexpr (HEAD_DIM <= 256) {
    return 2;
  } else if constexpr (HEAD_DIM <= 512) {
    return 1;
  } else {
    return 1;
  }
}

}  // namespace detail

template <typename Dtype, int HEAD_DIM, int ROPE_HEAD_DIM, typename Params>
void sm80_launch_mla_kernel(const Params& params, cudaStream_t stream) {
  const auto batch_size = params.batch_size;
  const auto max_q_packed_len = params.max_q_len * params.group_size;

  constexpr int stages = detail::stages_selector<80, HEAD_DIM, ROPE_HEAD_DIM>();
  using TileShape =
      decltype(detail::tile_shape_selector<80, HEAD_DIM, ROPE_HEAD_DIM>());
  using CollectiveMainloop =
      Sm80CollectiveMla<stages, TileShape, Dtype, HEAD_DIM, ROPE_HEAD_DIM>;
  using CollectiveEpilogue =
      Sm80CollectiveMlaEpilogue<TileShape, Dtype, HEAD_DIM>;

  // TODO: support persistent kernels
  using TileScheduler = SingleTileScheduler;

  constexpr int BLK_M = get<0>(TileShape{});

  const auto m_blocks = cute::ceil_div(max_q_packed_len, BLK_M);
  typename TileScheduler::Arguments scheduler_args{
      batch_size, m_blocks, /*n_kv_heads=*/1};
  auto scheduler_params =
      TileScheduler::to_underlying_arguments(scheduler_args);

  using AttnKernel =
      Sm80KernelMla<CollectiveMainloop, CollectiveEpilogue, TileScheduler>;

  auto mla_kernel = detail::device_kernel<AttnKernel, Params>;

  const auto smem_size = AttnKernel::kSharedStorageSize;
  if (smem_size >= 48 * 1024) {
    cudaFuncSetAttribute(
        mla_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  }

  const dim3 grid = AttnKernel::get_grid_shape(scheduler_args);
  const dim3 block = AttnKernel::get_block_shape();

  mla_kernel<<<grid, block, smem_size, stream>>>(params, scheduler_params);
  //   TODO: check launch status
}

}  // namespace llm
